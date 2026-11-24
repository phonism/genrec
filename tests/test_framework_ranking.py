"""Data contracts and independent upstream parity for the framework CTR port."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from genrec.data.criteo_ranking import CriteoRankingDataset
from genrec.models.rankmixer import RankElastor, RankMixer


@pytest.fixture
def feature_map():
    return {"dataset_id": "tiny", "num_fields": 3, "total_features": 13,
            "input_length": 3, "labels": ["label"], "features": [
                {"I1": {"source": "", "type": "numeric"}},
                {"C1": {"source": "", "type": "categorical", "vocab_size": 7, "padding_idx": 0}},
                {"C2": {"source": "", "type": "categorical", "vocab_size": 5, "padding_idx": 0}},
            ]}


def test_rankmixer_rejects_the_incompatible_paper_shape(feature_map):
    with pytest.raises(ValueError, match="token_dim % token_num"):
        RankMixer(feature_map, embedding_dim=20, token_num=15, token_dim=26)


def test_ctr_reader_preserves_training_vocabulary_and_column_order(tmp_path, feature_map):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "feature_map.json").write_text(json.dumps(feature_map))
    table = pa.table({"label": [0, 1, 0], "C2": [1, 2, 3], "I1": [0.2, 0.4, 0.6], "C1": [2, 4, 6]})
    pq.write_table(table, tmp_path / "train.parquet")
    ds = CriteoRankingDataset(tmp_path, max_rows=2)
    assert len(ds) == 2
    assert ds.feature_names == ["I1", "C1", "C2"]
    torch.testing.assert_close(ds[0]["features"], torch.tensor([0.2, 2, 1]))
    pq.write_table(pa.table({"I1": [0.1], "C1": [7], "C2": [1], "label": [0]}), tmp_path / "valid.parquet")
    with pytest.raises(ValueError, match="training vocabulary"):
        CriteoRankingDataset(tmp_path, "valid")


@pytest.mark.parametrize("model_class", [RankMixer, RankElastor])
def test_public_upstream_output_loss_gradient_and_adam_step(tmp_path, feature_map, model_class):
    pytest.importorskip("fuxictr")
    default = Path(__file__).resolve().parents[1] / "reference/upstream-rankelastor-68e5f29"
    source = Path(os.environ.get("GENREC_RANKELASTOR_REFERENCE", str(default)))
    if not (source / "src/UniMixer.py").exists():
        pytest.skip("set GENREC_RANKELASTOR_REFERENCE to the fixed external reference checkout")
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert pin == "68e5f29c9147b0c2263a1d28216b500940c057fe"
    sys.path.insert(0, str(source))
    from src.UniMixer import UniMixer
    from fuxictr.features import FeatureMap
    from fuxictr.utils import load_config

    path = tmp_path / "feature_map.json"
    path.write_text(json.dumps(feature_map))
    fm = FeatureMap("tiny", str(tmp_path))
    fm.load(str(path), {"embedding_dim": 4})
    full = model_class is RankElastor
    params = load_config(str(source / "config"), "UniMixer_criteo_dim_20" if full else "UniMixer_criteo_dim_20_base")
    params.update(dict(
        model_id="reference", model_root=str(tmp_path), gpu=-1, verbose=0,
        embedding_dim=4, Token_num=3, Token_dim=6, depth=2,
        mixing_type=999 if full else 2, pred_head="mlp", use_tokenizer=True,
        embedding_regularizer=1e-5,
        proj_param={"hidden_units": [5], "dropout": 0.0, "activation": "relu",
                    "trans_type": "matrix", "use_layer_norm": full},
        interaction_mlp_params={"hidden_units": [6], "dropout": 0.0,
                                "activation": "swiglu" if full else "gelu",
                                "expansion_rate": 3 if full else 1, "use_layer_norm": full},
    ))
    reference = UniMixer(fm, **params)
    port = model_class(feature_map, embedding_dim=4, token_num=3, token_dim=6,
                       num_blocks=2, ffn_hidden=6, tokenizer_hidden=5, tokenizer_dropout=0.0)
    port.load_state_dict(reference.state_dict(), strict=True)
    x = torch.tensor([[0.2, 1, 2], [0.7, 3, 1], [-0.4, 2, 4], [0.9, 4, 2]])
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    inputs = {name: x[:, i] for i, name in enumerate(["I1", "C1", "C2"])}
    inputs["label"] = y
    result = reference(inputs)
    logits, loss = port(x, y)
    torch.testing.assert_close(logits.sigmoid(), result["y_pred"], rtol=1e-6, atol=1e-7)
    original_loss = reference.compute_loss(result, y)
    torch.testing.assert_close(loss, original_loss, rtol=1e-6, atol=1e-7)
    original_loss.backward()
    loss.backward()
    ref_params = dict(reference.named_parameters())
    for name, p in port.named_parameters():
        torch.testing.assert_close(p.grad, ref_params[name].grad, rtol=2e-5, atol=2e-6, msg=name)
    for model in [reference, port]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        torch.optim.Adam(model.parameters(), lr=1e-3).step()
    for name, p in port.named_parameters():
        torch.testing.assert_close(p, ref_params[name], rtol=2e-5, atol=2e-6, msg=name)
