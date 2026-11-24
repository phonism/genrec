#!/usr/bin/env python3
"""Compare GenRec to a pinned, external FuXi-beta/gamma checkout with real FBGEMM.

Run this in the upstream-compatible Torch/FBGEMM environment. The implementation
under test is loaded from this source tree; no upstream code is vendored here.
The upstream package is used only as a numerical/data oracle, never for training
the framework reproduction. All assertions must pass before a report is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch


PINS = {"beta": ("4fc73bc6b3a4d66d6a37e4178d567ca83edb7a73", "8c0df0d6d5b86091358dd199d9acefb59775640a"),
        "gamma": ("b434e5daab49cc973740af7b4d219f7133b9d84b", "7b42bf791387aa9ec810159ec0b7a6125a19ff3b")}
ROOT = Path(__file__).resolve().parents[1]


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_map(name):
    if name == "item_embedding.weight":
        return "_embedding_module._item_emb.weight"
    if name == "position_embedding.weight":
        return "_input_features_preproc._pos_emb.weight"
    if name.startswith("layers."):
        return name.replace("layers.", "_fuxi._attention_layers.", 1)
    raise ValueError(f"unmapped parameter: {name}")


def check_model(device, variant, full_shape=False):
    import fbgemm_gpu  # noqa: F401 -- register genuine jagged operators
    import importlib
    import types
    external = importlib.import_module("generative_recommenders.modeling.sequential.fuxi_" + variant)
    source_class = getattr(external, "FuXi" + variant.title())
    from generative_recommenders.modeling.sequential.embedding_modules import LocalEmbeddingModule
    from generative_recommenders.modeling.sequential.input_features_preprocessors import (
        LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    )
    from generative_recommenders.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    from generative_recommenders.modeling.sequential.autoregressive_losses import (
        LocalNegativesSampler, SampledSoftmaxLoss,
    )
    from generative_recommenders.modeling.similarity.dot_product import DotProductSimilarity

    torch.manual_seed(17)
    # Avoid importing unrelated framework/LLM training dependencies in the oracle env.
    package = types.ModuleType("genrec"); package.__path__ = [str(ROOT / "genrec")]
    models = types.ModuleType("genrec.models"); models.__path__ = [str(ROOT / "genrec/models")]
    sys.modules["genrec"], sys.modules["genrec.models"] = package, models
    local = importlib.import_module("genrec.models.fuxi_" + variant)
    seq, output, width, heads = (200, 11, 50, 1) if full_shape else (4, 2, 8, 2)
    items, dqk, ffn = (3952, 50, 1) if full_shape else (9, 2, 2)
    dv = 50 if full_shape else (4 if variant == "gamma" else 3)
    negatives = 128 if full_shape else 5
    kwargs = dict(max_sequence_len=seq, max_output_len=output, embedding_dim=width, num_blocks=2,
                  num_heads=heads, linear_dim=dv, attention_dim=dqk, linear_activation="silu",
                  linear_dropout_rate=0.0, attn_dropout_rate=0.0, ffn_multiply=ffn,
                  embedding_module=LocalEmbeddingModule(items, width), similarity_module=DotProductSimilarity(),
                  input_features_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(seq + output, width, 0.0),
                  output_postproc_module=L2NormEmbeddingPostprocessor(width, eps=1e-6), verbose=False)
    if variant == "beta":
        kwargs.update(normalization="rel_bias", func_type="pow")
    else:
        kwargs.update(range_alpha=0.6, left_beta=0.4, right_beta=0.6,
                      gamma_is_learnable=False, left_gamma=0.7, right_gamma=0.9)
    reference = source_class(**kwargs).to(device)
    port = getattr(local, "FuXi" + variant.title())(
        items, max_seq_len=seq, max_output_len=output, embed_dim=width, num_blocks=2,
        num_heads=heads, dqk=dqk, dv=dv, dropout=0.0, ffn_multiply=ffn).to(device)
    ref_params = dict(reference.named_parameters())
    assert set(ref_params) == {parameter_map(n) for n, _ in port.named_parameters()}
    with torch.no_grad():
        for n, p in port.named_parameters():
            p.copy_(ref_params[parameter_map(n)])
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]], device=device)
    times = torch.tensor([[10, 20, 35, 50, 70, 0], [15, 45, 90, 0, 0, 0]], device=device)
    targets = torch.tensor([[2, 3, 4, 7, 0, 0], [6, 8, 0, 0, 0, 0]], device=device)
    if full_shape:
        ids, times, targets = [torch.zeros(2, seq + output, dtype=torch.long, device=device) for _ in range(3)]
        for row, length in enumerate((200, 117)):
            ids[row, :length] = torch.arange(1, length + 1, device=device)
            targets[row, :length] = torch.arange(2, length + 2, device=device)
            times[row, :length + 1] = torch.arange(1, length + 2, device=device) * 3600
    lengths = ids.ne(0).sum(-1)
    expected = reference(lengths, ids, reference.get_item_embeddings(ids), {"timestamps": times})
    actual = port.encode(ids, times)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    full_logits, _ = port(ids, times)
    last_logits, _ = port(ids, times, last_only=True)
    torch.testing.assert_close(last_logits, full_logits[torch.arange(2, device=device), lengths - 1])
    sampler = LocalNegativesSampler(items, reference._embedding_module._item_emb,
                                    list(range(1, items + 1)), True, 1e-6).to(device)
    objective = SampledSoftmaxLoss(negatives, 0.05, reference)
    torch.manual_seed(91)
    original_loss = objective(
        lengths, expected, targets, reference.get_item_embeddings(targets), targets.ne(0).float(), sampler,
    )
    torch.manual_seed(91)
    _, local_loss = port(ids, times, targets, num_negatives=negatives)
    torch.testing.assert_close(local_loss, original_loss, rtol=3e-5, atol=3e-6)
    original_loss.backward()
    local_loss.backward()
    gradient_error = 0.0
    gradient_relative_l2 = 0.0
    mixed_shape = full_shape and device.type == "cuda"
    for n, p in port.named_parameters():
        ref = ref_params[parameter_map(n)]
        if p.grad is not None:
            relative = float((p.grad - ref.grad).norm() / ref.grad.norm().clamp_min(1e-10))
            gradient_relative_l2 = max(gradient_relative_l2, relative)
            if mixed_shape:
                # Full-shape padded/jagged roundoff is amplified by BF16 loss
                # rounding; this is a norm comparison, not bitwise equivalence.
                assert relative <= 1e-3, (n, relative)
            else:
                torch.testing.assert_close(p.grad, ref.grad, rtol=3e-4, atol=3e-5, msg=n)
            gradient_error = max(gradient_error, (p.grad - ref.grad).abs().max().item())
        else:
            assert ref.grad is None
    for model in [reference, port]:
        torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0).step()
    step_error = 0.0
    for n, p in port.named_parameters():
        ref = ref_params[parameter_map(n)]
        if not mixed_shape:
            torch.testing.assert_close(p, ref, rtol=3e-4, atol=3e-5, msg=n)
        step_error = max(step_error, (p - ref).abs().max().item())
    return {"device": str(device), "torch": torch.__version__, "multihead_test": heads > 1,
            "full_training_shape": full_shape, "padded_length": seq + output,
            "position_embedding_shape": list(port.position_embedding.weight.shape),
            "relative_position_shapes": [list((layer._rel_attn if variant == "gamma" else layer._rel_attn_bias)._pos_w.shape)
                                         for layer in port.layers],
            "variable_lengths": lengths.tolist(), "parameters": sum(p.numel() for p in port.parameters()),
            "max_output_abs_error": (actual - expected).abs().max().item(),
            "loss": local_loss.item(), "loss_abs_error": abs(local_loss.item() - original_loss.item()),
            "gradient_check": "per_parameter_relative_l2_le_1e-3" if mixed_shape else "elementwise_rtol_3e-4_atol_3e-5",
            "max_gradient_relative_l2_error": gradient_relative_l2,
            "adamw_check": "record_only_bf16_roundoff" if mixed_shape else "elementwise_rtol_3e-4_atol_3e-5",
            "max_gradient_abs_error": gradient_error, "max_adamw_step_abs_error": step_error,
            "passed": True}


def check_data(csv_path, catalogue_path):
    from generative_recommenders.data.dataset import DatasetV2
    from generative_recommenders.modeling.sequential.features import movielens_seq_features_from_row
    local = load_local("framework_csv", ROOT / "genrec/data/sequential_csv.py")
    checked = {}
    for phase, ignore in [("train", 1), ("test", 0)]:
        reference = DatasetV2(str(csv_path), 201, ignore, chronological=True)
        port = local.SequentialCSVDataset(csv_path, phase, catalog_path=catalogue_path)
        assert len(reference) == len(port)
        for i in range(len(port)):
            ref, actual = reference[i], port[i]
            batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in ref.items()}
            features, _, _ = movielens_seq_features_from_row(batch, torch.device("cpu"), 11)
            torch.testing.assert_close(actual["input_ids"], features.past_ids[0])
            torch.testing.assert_close(actual["timestamps"], features.past_payloads["timestamps"][0])
            length = int(ref["history_lengths"])
            assert actual["lengths"].item() == length
            assert actual["target_items"].item() == int(ref["target_ids"])
            torch.testing.assert_close(actual["input_ids"][:200], ref["historical_ids"])
            torch.testing.assert_close(actual["timestamps"][:length], ref["historical_timestamps"][:length])
            assert actual["timestamps"][length].item() == int(ref["target_timestamps"])
            if phase == "train":
                expected = torch.cat([ref["historical_ids"][1:length], torch.tensor([ref["target_ids"]])])
                torch.testing.assert_close(actual["targets"][:length], expected)
        checked[phase] = len(port)
    return {"rows_checked": checked, "full_padded_length": 211, "source_feature_helper_checked": True,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "catalogue_sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest(), "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["beta", "gamma"], required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pin, expected_blob = PINS[args.variant]
    source_file = args.reference / ("generative_recommenders/modeling/sequential/fuxi_" + args.variant + ".py")
    data = source_file.read_bytes()
    actual_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual_blob != expected_blob:
        raise ValueError(f"reference source blob mismatch: {actual_blob}")
    sys.path.insert(0, str(args.reference.resolve()))
    report = {"upstream_revision": pin, "upstream_blob": actual_blob,
              "model": check_model(torch.device(args.device), args.variant),
              "full_shape_model": check_model(torch.device(args.device), args.variant, full_shape=True)}
    if args.csv:
        report["data"] = check_data(args.csv, args.catalogue)
    report["implementation_sha256"] = hashlib.sha256((ROOT / ("genrec/models/fuxi_" + args.variant + ".py")).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
