"""
NoteLLM: A Retrievable Large Language Model for Note Recommendation
https://arxiv.org/pdf/2403.01744
"""
import torch
from torch import nn
import torch.distributed as dist
import gin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers import cache_utils
from transformers.loss import loss_utils
from typing import Union, Optional, Dict, List, Callable, Tuple, Sequence


class DynamicCache(cache_utils.DynamicCache):
    """
    DynamicCache with device caching
    """

    def __init__(self, obj: cache_utils.DynamicCache):
        super().__init__()
        self.key_cache, self.value_cache = obj.key_cache, obj.value_cache
        self.fixed_seq_length = obj.get_seq_length()
        self.device_dict = {self.key_cache[0].device: self}

    def to(self, device: torch.device) -> "DynamicCache":
        """
        Get the DynamicCache object for the specified device.
        """
        obj = self.device_dict.get(device)
        if obj is None:
            obj = self.device_dict[device] = cache_utils.DynamicCache()
            obj.key_cache = [k.to(device) for k in self.key_cache]
            obj.value_cache = [v.to(device) for v in self.value_cache]
        obj.crop(self.fixed_seq_length)
        return obj


@gin.configurable 
class Query2Embedding(nn.Module):
    """Model"""

    def __init__(
        self,
        pretrained_path: str,
        freeze_lm: bool = False,
        item_token: str = "[EMB]",
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        gradient_checkpointing: bool = True,
        tau: float = 3.0,
        alpha: float = 0.01,
        hardneg_r: float = 0.1
    ):
        super().__init__()

        self.tau = nn.Parameter(torch.tensor(tau))
        self.alpha, self.hardneg_r = alpha, hardneg_r
        self.item_token = item_token
        # Auto classes keep the implementation compatible with both the
        # original Qwen2 backbone and the repository's locally mirrored Qwen3
        # checkpoints.
        tokenizer_kwargs = {}
        if pad_token is not None:
            tokenizer_kwargs["pad_token"] = pad_token
        if eos_token is not None:
            tokenizer_kwargs["eos_token"] = eos_token
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_path, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("NoteLLM tokenizer needs either a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_path)
        if freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False
        previous_vocab_size = len(self.tokenizer)
        source_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_tokens([item_token], special_tokens=True)
        self.emb_id = self.tokenizer.convert_tokens_to_ids(item_token)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.vocab_size = len(self.tokenizer)
        if self.emb_id >= previous_vocab_size:
            # LoRA freezes the base embedding table.  Seed the newly-added
            # marker from a real special token instead of leaving it as a
            # permanently-random vector.
            if source_token_id is None or source_token_id >= previous_vocab_size:
                source_token_id = 0
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings().weight
                input_embeddings[self.emb_id].copy_(input_embeddings[source_token_id])
                output_layer = self.model.get_output_embeddings()
                if output_layer is not None and output_layer.weight.data_ptr() != input_embeddings.data_ptr():
                    output_layer.weight[self.emb_id].copy_(
                        output_layer.weight[source_token_id]
                    )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def tokenize(self,
        query: list[str],
        category: list[str] = None,
        score: Optional[list[float]] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            query (list[str]): query texts.
            category (list[str]): category texts.
            score (list[float]): score.
        Returns:
            Dict[str, torch.Tensor]: tokenized texts.
        """
        original_truncation_side = self.tokenizer.truncation_side
        if truncation:
            # [EMB] is deliberately the final token. Left truncation preserves
            # both that marker and the user's most recent interactions.
            self.tokenizer.truncation_side = "left"
        try:
            token = self.tokenizer(
                query,
                category,
                padding=True,
                truncation=truncation,
                max_length=max_length,
                return_token_type_ids=category is not None,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.truncation_side = original_truncation_side

        # The paper uses the hidden state immediately BEFORE [EMB].  With a
        # causal decoder, the marker state has already consumed the marker
        # itself and is not the advertised compressed-note representation.
        emb_mask = token.input_ids == self.emb_id
        if not emb_mask.any(dim=1).all():
            raise ValueError(
                "every NoteLLM input must contain the embedding marker; "
                "increase max_length or keep the marker at the end"
            )
        marker_idx = emb_mask.int().argmax(1, keepdim=True)
        if (marker_idx == 0).any():
            raise ValueError("the embedding marker needs a preceding content token")
        token["emb_token_idx"] = marker_idx - 1
        
        if category is not None:
            token["labels"] = token.input_ids.where(token.pop("token_type_ids").bool(), -100)

        # build hardneg label
        if score is not None and len(score) > 0:
            token["hardneg"] = torch.tensor(score) < self.hardneg_r
        return token

    def tokenize_with_targets(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize the joint GCL+CSFT prompt with response-only labels.

        Decoder-only tokenizers do not reliably expose segment IDs for text
        pairs (Qwen returns all-zero token types).  Build each sequence
        explicitly so only the category/topic target contributes to the causal
        language-model loss, while left-truncating prompt content keeps both
        the recent text and the [EMB] compression boundary.
        """
        if len(prompts) != len(targets):
            raise ValueError("prompts and targets must have the same length")
        if max_length <= 1:
            raise ValueError("max_length must be greater than one")

        input_rows = []
        label_rows = []
        embedding_positions = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("NoteLLM CSFT requires an EOS token")

        for prompt, target in zip(prompts, targets):
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=True
            ).input_ids
            target_ids = self.tokenizer(
                str(target).strip(), add_special_tokens=False
            ).input_ids + [eos_id]
            if len(target_ids) >= max_length:
                target_ids = target_ids[: max_length - 1] + [eos_id]
            prompt_budget = max_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
            if self.emb_id not in prompt_ids:
                raise ValueError(
                    "CSFT truncation removed [EMB]; keep the marker at the end "
                    "of the compression prompt"
                )
            marker_idx = prompt_ids.index(self.emb_id)
            if marker_idx == 0:
                raise ValueError("the embedding marker needs a preceding content token")

            input_rows.append(torch.tensor(prompt_ids + target_ids, dtype=torch.long))
            label_rows.append(torch.tensor(
                [-100] * len(prompt_ids) + target_ids, dtype=torch.long
            ))
            embedding_positions.append(marker_idx - 1)

        row_lengths = [row.numel() for row in input_rows]
        input_ids = nn.utils.rnn.pad_sequence(
            input_rows, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = nn.utils.rnn.pad_sequence(
            label_rows, batch_first=True, padding_value=-100
        )
        # Qwen commonly uses EOS as its padding token.  Deriving the mask via
        # ``input_ids != pad_id`` would therefore hide every real EOS target
        # from attention as well as the padding.  Preserve lengths explicitly.
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row_length in enumerate(row_lengths):
            attention_mask[row_index, :row_length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "emb_token_idx": torch.tensor(
                embedding_positions, dtype=torch.long
            ).unsqueeze(1),
            "labels": labels,
        }

    @staticmethod
    def _gather_with_local_grad(embeddings: torch.Tensor) -> torch.Tensor:
        """Gather cross-rank negatives while retaining this rank's gradient."""
        if not dist.is_available() or not dist.is_initialized():
            return embeddings
        gathered = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, embeddings.detach())
        gathered[dist.get_rank()] = embeddings
        return torch.cat(gathered, dim=0)

    def paired_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Paper GCL over both directions and every non-self in-batch note.

        Training rows are interleaved ``q0,p0,q1,p1,...``.  The positive for
        row ``i`` is therefore ``i xor 1``; all other local and cross-rank rows
        are negatives, matching Equation (2) and the official implementation.
        """
        if embeddings.ndim != 2 or embeddings.size(0) % 2:
            raise ValueError("GCL requires an even number of paired embeddings")
        local_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        if dist.is_available() and dist.is_initialized():
            local_size = torch.tensor(
                [local_embeddings.size(0)], device=local_embeddings.device
            )
            sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, local_size)
            if any(int(size.item()) != local_embeddings.size(0) for size in sizes):
                raise ValueError(
                    "cross-rank GCL requires equal local batch sizes; use drop_last=True"
                )
            rank_offset = dist.get_rank() * local_embeddings.size(0)
        else:
            rank_offset = 0

        all_embeddings = self._gather_with_local_grad(local_embeddings)
        # Each rank owns only its local anchors.  DDP then averages those local
        # means into the exact global-anchor objective; evaluating the complete
        # global matrix on every rank would introduce an extra 1/world-size
        # gradient scaling.
        logits = local_embeddings @ all_embeddings.T
        logits = logits * self.tau.exp()
        local_rows = torch.arange(local_embeddings.size(0), device=embeddings.device)
        labels = rank_offset + (local_rows ^ 1)
        if item_ids is not None:
            item_ids = torch.as_tensor(
                item_ids, device=embeddings.device, dtype=torch.long
            ).reshape(-1)
            if item_ids.numel() != local_embeddings.size(0):
                raise ValueError("item_ids must align with paired embeddings")
            if dist.is_available() and dist.is_initialized():
                gathered_ids = [
                    torch.empty_like(item_ids) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_ids, item_ids)
                all_item_ids = torch.cat(gathered_ids, dim=0)
            else:
                all_item_ids = item_ids
            duplicate_item = item_ids.unsqueeze(1).eq(all_item_ids.unsqueeze(0))
            # A designated pair remains positive even in a pathological
            # self-transition; only additional copies are false negatives.
            duplicate_item[local_rows, labels] = False
            logits = logits.masked_fill(
                duplicate_item, torch.finfo(logits.dtype).min
            )
        logits[local_rows, rank_offset + local_rows] = torch.finfo(logits.dtype).min
        return nn.functional.cross_entropy(logits, labels)

    def _causal_lm(self) -> PreTrainedModel:
        """Return the underlying causal LM, unwrapping PEFT when present."""
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        return self.model

    def get_embedding(self, input_ids, attention_mask, emb_token_idx, past_key_values):
        """
        get embedding
        """
        causal_lm = self._causal_lm()
        outputs = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        last_hidden_state, outputs["last_hidden_state"] = outputs["last_hidden_state"], None

        # extract embedding
        emb_token_idx = emb_token_idx.repeat(1, causal_lm.config.hidden_size).unsqueeze(1)
        outputs["sentence_embedding"] = last_hidden_state.gather(1, emb_token_idx).squeeze(1)
        
        norm_embedding = nn.functional.normalize(outputs.sentence_embedding, p=2, dim=1)
        return norm_embedding, last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        emb_token_idx: torch.LongTensor,
        labels: torch.LongTensor = None,
        hardneg: torch.BoolTensor = None,
        pair_item_ids: torch.LongTensor = None,
        past_key_values: DynamicCache = None,
        return_loss: bool = True,
    ):
        """
        forward
        """
        if past_key_values is not None:
            past_key_values = past_key_values.to(input_ids.device)
            if past_key_values.key_cache[0].shape[0] != input_ids.shape[0]:
                past_key_values.batch_select_indices([0] * input_ids.shape[0])

            attention_mask = torch.cat([
                torch.ones(
                    attention_mask.shape[0],
                    past_key_values.get_seq_length(),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ), attention_mask
            ], dim=1)


        outputs = {}
        outputs["sentence_embedding"], last_hidden_state = self.get_embedding(
            input_ids,
            attention_mask,
            emb_token_idx,
            past_key_values
        )

        if not return_loss:
            return {"sentence_embedding": outputs["sentence_embedding"]}

        # Generative contrastive learning: both members of every related pair
        # are anchors, and negatives include all ranks when DDP is active.
        cl_loss = self.paired_contrastive_loss(
            outputs["sentence_embedding"], item_ids=pair_item_ids
        )

        if hardneg is not None:
            # Retain the diagnostic in outputs.  The paper's published GCL
            # objective does not replace positive pairs using this legacy mask.
            outputs["hardneg"] = hardneg

        if labels is None or (labels < 0).all():
            outputs["loss"] = cl_loss
            return outputs

        # category generation loss
        causal_lm = self._causal_lm()
        logits = causal_lm.lm_head(last_hidden_state)
        gen_loss = loss_utils.ForCausalLMLoss(
            logits=logits,
            labels=labels,
            vocab_size=causal_lm.config.vocab_size
        )
        outputs["loss"] = (cl_loss + gen_loss * self.alpha) / (1 + self.alpha)
        return outputs

    def save_pretrained(self, save_dir: str, **kwargs):
        """
        Save both tokenizer and model weights.

        Args:
            save_dir (str): directory to save the model and tokenizer.
            **kwargs: additional arguments to pass to the tokenizer and model.
        """
        self.model.save_pretrained(save_dir, **kwargs)
        self.tokenizer.save_pretrained(save_dir)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Support two input methods:
        1. Tensor: directly as input_ids
        2. Dict: containing all parameters needed for model generation (e.g. input_ids, attention_mask)
        Additional parameters are passed to model.generate() through kwargs
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        elif not isinstance(inputs, dict):
            raise ValueError("inputs must be Tensor or Dict[str, Tensor]")
        
        # call model generate
        return self.model.generate(**inputs, **kwargs)

    @staticmethod
    def compute_metrics(topk: int = 5, batch_size: int = 64, save_file: str = ''):
        """Return an explicitly diagnostic in-batch retrieval metric callback.

        This callback only ranks each query against the positives present in the
        same evaluation batch.  It is useful for debugging representation
        learning, but it is *not* a full-catalog recommendation metric and must
        not be reported alongside Recall/NDCG from the other recommenders.

        Use :meth:`compute_full_catalog_metrics` for benchmark evaluation.
        """

        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        def compute_topk_acc(eval_prediction: EvalPrediction):
            """Compute recall against only the paired candidates in each batch."""
            if isinstance(eval_prediction.predictions, tuple):
                pred, hardneg = eval_prediction.predictions
                pred = torch.tensor(pred)
                hardneg = torch.as_tensor(hardneg, dtype=torch.bool)
                pred1, pred2 = pred[::2][~hardneg], pred[1::2][~hardneg]
            else:
                pred = torch.tensor(eval_prediction.predictions)
                pred1, pred2 = pred[::2], pred[1::2]

            if pred1.shape != pred2.shape:
                raise ValueError(
                    "NoteLLM metric expects paired query/item embeddings with "
                    f"matching shapes, got {tuple(pred1.shape)} and {tuple(pred2.shape)}"
                )
            if pred1.shape[0] == 0:
                raise ValueError("NoteLLM metric received no evaluable embedding pairs")

            correct_topk = 0
            for i in range(0, pred1.shape[0], batch_size):
                sim = torch.mm(
                    nn.functional.normalize(pred1[i:i + batch_size], p=2, dim=1),
                    nn.functional.normalize(pred2[i:i + batch_size], p=2, dim=1).T,
                )
                block_topk = min(topk, sim.shape[1])
                topk_idx = sim.topk(block_topk, dim=1).indices
                true_idx = torch.arange(sim.shape[0], device=topk_idx.device).unsqueeze(1)
                correct_topk += (topk_idx == true_idx).any(dim=1).sum().item()
            topk_acc = correct_topk / pred1.shape[0]

            if save_file:
                with open(save_file, "a") as f:
                    print(topk_acc, file=f)
            return {f"diagnostic_in_batch_recall@{topk}": topk_acc}
        return compute_topk_acc

    @staticmethod
    @torch.no_grad()
    def compute_full_catalog_metrics(
        query_embeddings: torch.Tensor,
        catalog_embeddings: torch.Tensor,
        target_item_ids: torch.Tensor,
        catalog_item_ids: Optional[torch.Tensor] = None,
        excluded_item_ids: Optional[torch.Tensor] = None,
        topks: Sequence[int] = (1, 5, 10),
        query_batch_size: int = 256,
    ) -> Dict[str, float]:
        """Compute exact item-level Recall/NDCG against the full catalog.

        ``catalog_embeddings`` must contain exactly one row per candidate item.
        ``catalog_item_ids`` identifies those rows; when omitted, row indices are
        treated as item IDs.  Each query has one ground-truth item.
        """

        if query_embeddings.ndim != 2 or catalog_embeddings.ndim != 2:
            raise ValueError("query_embeddings and catalog_embeddings must be rank-2")
        if query_embeddings.shape[1] != catalog_embeddings.shape[1]:
            raise ValueError("query and catalog embedding dimensions must match")
        if catalog_embeddings.shape[0] == 0:
            raise ValueError("catalog_embeddings must contain at least one item")
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")

        target_item_ids = torch.as_tensor(
            target_item_ids, device=query_embeddings.device, dtype=torch.long
        ).reshape(-1)
        if target_item_ids.shape[0] != query_embeddings.shape[0]:
            raise ValueError("target_item_ids must contain one ID per query")

        if catalog_item_ids is None:
            catalog_item_ids = torch.arange(
                catalog_embeddings.shape[0], device=query_embeddings.device
            )
        else:
            catalog_item_ids = torch.as_tensor(
                catalog_item_ids, device=query_embeddings.device, dtype=torch.long
            ).reshape(-1)
        if catalog_item_ids.shape[0] != catalog_embeddings.shape[0]:
            raise ValueError("catalog_item_ids must contain one ID per catalog row")
        if torch.unique(catalog_item_ids).numel() != catalog_item_ids.numel():
            raise ValueError("catalog_item_ids must be unique for item-level evaluation")
        if not torch.isin(target_item_ids, catalog_item_ids).all():
            raise ValueError("every target_item_id must be present in the catalog")

        if excluded_item_ids is not None:
            excluded_item_ids = torch.as_tensor(
                excluded_item_ids, device=query_embeddings.device, dtype=torch.long
            ).reshape(-1)
            if excluded_item_ids.shape[0] != query_embeddings.shape[0]:
                raise ValueError("excluded_item_ids must contain one ID per query")
            if not torch.isin(excluded_item_ids, catalog_item_ids).all():
                raise ValueError("every excluded_item_id must be present in the catalog")
            if torch.eq(excluded_item_ids, target_item_ids).any():
                raise ValueError("a target item cannot also be excluded")

        clean_topks = tuple(sorted(set(int(k) for k in topks)))
        if not clean_topks or clean_topks[0] <= 0:
            raise ValueError("topks must contain positive integers")
        available_items = catalog_embeddings.shape[0] - int(excluded_item_ids is not None)
        if available_items <= 0:
            raise ValueError("full-catalog evaluation needs an unexcluded candidate")
        max_k = min(clean_topks[-1], available_items)

        catalog = nn.functional.normalize(
            catalog_embeddings.to(
                device=query_embeddings.device, dtype=query_embeddings.dtype
            ),
            p=2,
            dim=1,
        )
        recall_hits = {k: 0.0 for k in clean_topks}
        ndcg_sum = {k: 0.0 for k in clean_topks}

        for start in range(0, query_embeddings.shape[0], query_batch_size):
            stop = start + query_batch_size
            queries = nn.functional.normalize(query_embeddings[start:stop], p=2, dim=1)
            scores = torch.mm(queries, catalog.T)
            if excluded_item_ids is not None:
                scores.masked_fill_(
                    catalog_item_ids.unsqueeze(0).eq(
                        excluded_item_ids[start:stop].unsqueeze(1)
                    ),
                    float("-inf"),
                )
            top_indices = scores.topk(max_k, dim=1).indices
            top_items = catalog_item_ids[top_indices]
            matches = top_items.eq(target_item_ids[start:stop].unsqueeze(1))

            for requested_k in clean_topks:
                effective_k = min(requested_k, max_k)
                matches_at_k = matches[:, :effective_k]
                hit = matches_at_k.any(dim=1)
                recall_hits[requested_k] += hit.sum().item()
                if hit.any():
                    ranks = matches_at_k.float().argmax(dim=1)[hit].float() + 1.0
                    ndcg_sum[requested_k] += (1.0 / torch.log2(ranks + 1.0)).sum().item()

        denominator = query_embeddings.shape[0]
        if denominator == 0:
            raise ValueError("query_embeddings must contain at least one query")
        metrics = {}
        for k in clean_topks:
            metrics[f"recall@{k}"] = recall_hits[k] / denominator
            metrics[f"ndcg@{k}"] = ndcg_sum[k] / denominator
        return metrics
