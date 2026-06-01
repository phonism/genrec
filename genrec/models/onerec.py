"""
OneRec: Generative Recommender with RQ-VAE + SFT + GRPO pipeline.

Independent implementation (does not inherit LCRec).
Wraps a Qwen2 LLM backbone with codebook tokens and GRPO generation methods.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DynamicCache,
)
from typing import Optional, Dict, List, Callable, Tuple


class OneRec(nn.Module):
    """
    OneRec model based on Qwen2 LLM backbone.

    Supports:
    - SFT training with codebook tokens
    - GRPO generation (sample multiple completions per prompt)
    - Log probability computation for GRPO loss
    """

    def __init__(self, pretrained_path: str, torch_dtype: str = "auto") -> None:
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained_path)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_path, torch_dtype=torch_dtype,
        )
        # Fix checkpoints saved via OneRec.save_pretrained() with accelerator.save —
        # they may have "model.model." prefix instead of "model." due to the OneRec
        # wrapper. Detect and reload with corrected keys if needed.
        import safetensors.torch, glob, os
        sf_files = glob.glob(os.path.join(pretrained_path, "*.safetensors"))
        if sf_files:
            sd = safetensors.torch.load_file(sf_files[0])
            if any(k.startswith("model.model.") for k in sd.keys()):
                fixed = {k.replace("model.model.", "model.", 1).replace("model.lm_head.", "lm_head.", 1): v
                         for k, v in sd.items()}
                self.model.load_state_dict(fixed, strict=False)

    def gradient_checkpointing_enable(self, use_reentrant=True):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': use_reentrant}
        )

    def add_codebook_tokens(self, num_codebooks: int, codebook_size: int):
        """Add <C{i}_{j}> special tokens to the tokenizer and resize embeddings."""
        new_tokens = [f"<C{i}_{j}>" for i in range(num_codebooks) for j in range(codebook_size)]
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            self.model.config.vocab_size = len(self.tokenizer)

    def add_item_sep_token(self) -> int:
        """Add <|item_sep|> special token for CLM format. Returns the token ID."""
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|item_sep|>"]}
        )
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.vocab_size = len(self.tokenizer)
        self.item_sep_token_id = self.tokenizer.convert_tokens_to_ids("<|item_sep|>")
        return self.item_sep_token_id

    def tokenize_sft_format(
        self,
        prompt: str,
        response: str = "",
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompt + response in SFT format."""
        prompt_ids = self.tokenizer(prompt).input_ids
        response_ids = self.tokenizer(response).input_ids
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        input_ids = torch.LongTensor([input_ids]).to(device)
        return {
            "input_ids": input_ids,
            "prompt_seq_length": len(prompt_ids),
            "attention_mask": torch.ones_like(input_ids),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_pretrained(self, save_dir: str, **kwargs):
        # Explicitly pass the HF model's state dict to avoid double "model." prefix
        # when accelerator.save is used as save_function
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = self.model.state_dict()
        self.model.save_pretrained(save_dir, **kwargs)
        self.tokenizer.save_pretrained(save_dir)

    def load_pretrained(self, load_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            load_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loaded checkpoint from {load_dir}")

    @torch.no_grad()
    def generate_constrained_beam(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        position_mask: torch.Tensor,    # [num_steps, vocab_size] bool
        code_map: torch.Tensor,         # [num_codebooks, vocab_size] -> code value
        num_codebooks: int,
        beam_width: int = 10,
        topk: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom constrained beam search with KV cache reuse.

        Returns:
            sem_ids: [B, topk, num_codebooks] predicted semantic IDs
            scores:  [B, topk] log-probability scores
        """
        B, L = input_ids.shape
        device = input_ids.device
        num_steps = num_codebooks + 1  # codebook tokens + EOS

        # Caller is expected to keep lookup tables on device (see
        # ConstrainedDecodingHelper.to()); these guards are cheap no-ops when
        # already on-device and avoid an H2D copy per evaluation batch.
        if position_mask.device != device:
            position_mask = position_mask.to(device)
        if code_map.device != device:
            code_map = code_map.to(device)

        # --- Prefill: one forward pass to get KV cache + first logits ---
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]  # [B, V]
        past_kv = out.past_key_values

        # Apply step-0 mask
        mask_0 = position_mask[0].unsqueeze(0)  # [1, V]
        logits = logits.masked_fill(~mask_0, float('-inf'))
        log_probs = F.log_softmax(logits, dim=-1)  # [B, V]

        # Select top beam_width tokens per sample
        topk_scores, topk_ids = log_probs.topk(beam_width, dim=-1)  # [B, beam_width]

        # Expand for beams: [B, beam_width]
        beam_scores = topk_scores  # [B, beam_width]
        beam_tokens = topk_ids.unsqueeze(-1)  # [B, beam_width, 1]

        # Expand KV cache: each layer (key, value) from [B, H, L, D] -> [B*beam_width, H, L, D]
        past_kv = self._expand_past_kv(past_kv, beam_width)

        # Expand attention_mask for beams
        # [B, L] -> [B*beam_width, L]
        attn_mask = attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(B * beam_width, L)

        # --- Autoregressive steps 1..num_steps-1 ---
        for step in range(1, num_steps):
            # Next input token: last generated token for each beam
            next_input = beam_tokens[:, :, -1].reshape(B * beam_width, 1)  # [B*bw, 1]

            # Update attention mask
            attn_mask = torch.cat([attn_mask, torch.ones(B * beam_width, 1, device=device, dtype=attn_mask.dtype)], dim=-1)

            out = self.model(input_ids=next_input, attention_mask=attn_mask, past_key_values=past_kv, use_cache=True)
            logits = out.logits[:, -1, :]  # [B*bw, V]
            past_kv = out.past_key_values

            # Apply step mask
            mask_s = position_mask[step].unsqueeze(0)  # [1, V]
            logits = logits.masked_fill(~mask_s, float('-inf'))
            log_probs = F.log_softmax(logits, dim=-1)  # [B*bw, V]
            log_probs = log_probs.view(B, beam_width, -1)  # [B, bw, V]

            # For non-EOS steps, expand beams
            if step < num_codebooks:
                # Candidate scores: [B, bw, V] -> select top beam_width from bw*V
                candidate_scores = beam_scores.unsqueeze(-1) + log_probs  # [B, bw, V]
                candidate_scores = candidate_scores.view(B, -1)  # [B, bw*V]
                top_scores, top_indices = candidate_scores.topk(beam_width, dim=-1)  # [B, bw]

                # Decode beam and token indices
                beam_idx = top_indices // log_probs.size(-1)  # [B, bw] which beam
                token_idx = top_indices % log_probs.size(-1)  # [B, bw] which token

                # Reorder beams
                beam_scores = top_scores  # [B, bw]
                # Gather previous tokens from selected beams
                prev_tokens = torch.gather(
                    beam_tokens, 1,
                    beam_idx.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
                )  # [B, bw, step]
                beam_tokens = torch.cat([prev_tokens, token_idx.unsqueeze(-1)], dim=-1)  # [B, bw, step+1]

                # Reorder KV cache
                reorder_idx = (torch.arange(B, device=device).unsqueeze(1) * beam_width + beam_idx).view(-1)  # [B*bw]
                past_kv = self._reorder_past_kv(past_kv, reorder_idx)
                attn_mask = attn_mask[reorder_idx]
            else:
                # EOS step: just pick the EOS score and add to beam scores
                allowed_mask = position_mask[step]  # [V]
                eos_token_idx = allowed_mask.nonzero(as_tuple=False)[0, 0]
                eos_log_prob = log_probs[:, :, eos_token_idx]  # [B, bw]
                beam_scores = beam_scores + eos_log_prob

        # --- Extract sem_ids from beam_tokens using code_map ---
        # beam_tokens: [B, beam_width, num_codebooks] (excluding EOS)
        # code_map: [num_codebooks, V]
        sem_ids = torch.zeros(B, beam_width, num_codebooks, dtype=torch.long, device=device)
        for c in range(num_codebooks):
            token_ids = beam_tokens[:, :, c]  # [B, bw]
            sem_ids[:, :, c] = code_map[c][token_ids]  # lookup

        # Sort by score descending and take topk
        sorted_idx = beam_scores.argsort(dim=-1, descending=True)  # [B, bw]
        sorted_idx_k = sorted_idx[:, :topk]  # [B, topk]
        sem_ids = torch.gather(sem_ids, 1, sorted_idx_k.unsqueeze(-1).expand(-1, -1, num_codebooks))
        scores = torch.gather(beam_scores, 1, sorted_idx_k)

        return sem_ids, scores

    @staticmethod
    def _expand_past_kv(past_kv, beam_width):
        """Expand KV cache from [B, ...] to [B*beam_width, ...]."""
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx in range(len(past_kv)):
                layer = past_kv.layers[layer_idx]
                key = layer.keys
                value = layer.values
                new_key = key.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(-1, *key.shape[1:])
                new_value = value.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(-1, *value.shape[1:])
                new_cache.update(new_key, new_value, layer_idx)
            return new_cache
        expanded = []
        for layer_kv in past_kv:
            expanded.append(tuple(
                t.unsqueeze(1).expand(-1, beam_width, -1, -1, -1)
                 .reshape(t.size(0) * beam_width, *t.shape[1:])
                for t in layer_kv
            ))
        return tuple(expanded)

    @staticmethod
    def _reorder_past_kv(past_kv, reorder_idx):
        """Reorder KV cache according to beam selection indices."""
        if isinstance(past_kv, DynamicCache):
            past_kv.reorder_cache(reorder_idx)
            return past_kv
        return tuple(
            tuple(t.index_select(0, reorder_idx) for t in layer_kv)
            for layer_kv in past_kv
        )

    @torch.no_grad()
    def generate_topk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 3,
        beam_width: int = 10,
        topk: Optional[int] = None,
        allowed_token_fn: Optional[Callable[[int], bool]] = None,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[List[Tuple[torch.Tensor, float]]]:
        """Batched beam search for evaluation."""
        batch_size = input_ids.size(0)
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        topk = topk or beam_width
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id

        beams = [[(input_ids[i], 0.0, False)] for i in range(batch_size)]

        for _ in range(max_new_tokens):
            new_beams = []
            for b in range(batch_size):
                candidates = []
                for seq, score, finished in beams[b]:
                    if finished:
                        candidates.append((seq, score, True))
                        continue

                    attn = attention_mask[b].unsqueeze(0)
                    out = self.model(input_ids=seq.unsqueeze(0), attention_mask=attn)
                    logits = out.logits[0, -1] / temperature
                    log_probs = F.log_softmax(logits, dim=-1)

                    next_scores, next_tokens = torch.topk(log_probs, beam_width)
                    for tok, tok_logp in zip(next_tokens.tolist(), next_scores.tolist()):
                        if allowed_token_fn and not allowed_token_fn(tok):
                            continue
                        new_seq = torch.cat([seq, torch.tensor([tok], device=device)])
                        new_score = score + tok_logp
                        new_finished = (tok == eos_token_id)
                        candidates.append((new_seq, new_score, new_finished))
                if not candidates:
                    candidates = beams[b]
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:beam_width])
            beams = new_beams

            if all(all(finished for (_, _, finished) in beam) for beam in beams):
                break

        final_result = []
        for beam in beams:
            beam.sort(key=lambda x: x[1], reverse=True)
            final_result.append([(seq, score) for seq, score, _ in beam[:topk]])
        return final_result

    @torch.no_grad()
    def generate_for_grpo(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int = 16,
        max_new_tokens: int = 4,
        temperature: float = 1.0,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        use_beam_search: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple completions per prompt for GRPO.

        Args:
            input_ids: [B, L] prompt token ids
            attention_mask: [B, L] attention mask
            num_generations: G completions per prompt
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            prefix_allowed_tokens_fn: constrained decoding function
            use_beam_search: if True, use beam search (MiniOneRec style);
                            if False, use sampling (OpenOneRec style)

        Returns:
            generated_ids: [B*G, L+T] full sequences (prompt + generated)
            gen_log_probs: [B*G, T] log probs of generated tokens
        """
        B, L = input_ids.shape
        device = input_ids.device

        if use_beam_search:
            # Stochastic beam search (MiniOneRec style): do_sample=True
            # for diverse candidates across iterations
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_beams=num_generations,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            # Sampling: diverse candidates with exploration
            expanded_ids = input_ids.repeat_interleave(num_generations, dim=0)
            expanded_mask = attention_mask.repeat_interleave(num_generations, dim=0)
            gen_kwargs = dict(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        if prefix_allowed_tokens_fn is not None:
            gen_kwargs['prefix_allowed_tokens_fn'] = prefix_allowed_tokens_fn

        generated_ids = self.model.generate(**gen_kwargs)  # [B*G, L+T']

        # Pad to consistent length (L + max_new_tokens)
        target_len = L + max_new_tokens
        if generated_ids.size(1) < target_len:
            pad = torch.full(
                (generated_ids.size(0), target_len - generated_ids.size(1)),
                self.tokenizer.pad_token_id, device=device, dtype=generated_ids.dtype
            )
            generated_ids = torch.cat([generated_ids, pad], dim=1)
        elif generated_ids.size(1) > target_len:
            generated_ids = generated_ids[:, :target_len]

        # Compute log probs for generated tokens
        gen_log_probs = self.compute_log_probs(generated_ids, prompt_len=L)

        return generated_ids, gen_log_probs

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probs for the generated portion.

        Args:
            input_ids: [B, L+T] full sequences
            prompt_len: length of prompt (tokens before this are not scored)
            attention_mask: optional attention mask

        Returns:
            log_probs: [B, T] log probabilities of generated tokens
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L+T, V]

        # Shift: logits[t] predicts input_ids[t+1]
        # We want log_probs for positions prompt_len to end
        shift_logits = logits[:, prompt_len - 1:-1, :]  # [B, T, V]
        shift_labels = input_ids[:, prompt_len:]  # [B, T]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Mask out padding
        pad_mask = (shift_labels != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * pad_mask

        return token_log_probs
