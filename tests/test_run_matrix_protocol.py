from scripts.run_matrix import build_cells


def test_matrix_separates_protocol_output_and_binds_dataset():
    cells = build_cells(
        models=["onerec_sft"],
        splits=["beauty"],
        seeds=[42],
        save_dir_template="out/{protocol}/{model}/{split}/{seed}",
        config_overrides={},
        evaluation_protocol="global_temporal",
    )

    assert cells[0].save_dir == "out/global_temporal/onerec_sft/beauty/42"
    assert (
        'AmazonOneRecSFTDataset.evaluation_protocol="global_temporal"'
        in cells[0].gin_overrides()
    )


def test_grpo_binds_training_and_evaluation_datasets():
    cell = build_cells(
        models=["onerec_grpo"],
        splits=["beauty"],
        seeds=[42],
        save_dir_template="out/{protocol}/{model}/{split}/{seed}",
        config_overrides={},
        evaluation_protocol="global_temporal",
    )[0]
    overrides = cell.gin_overrides()

    assert 'AmazonOneRecRLDataset.evaluation_protocol="global_temporal"' in overrides
    assert 'AmazonOneRecSFTDataset.evaluation_protocol="global_temporal"' in overrides
    assert (
        'train_grpo.sft_checkpoint_path="out/global_temporal/'
        'onerec_sft/beauty/42/checkpoint_best"'
        in overrides
    )
    assert cell.sft_checkpoint_path == (
        "out/global_temporal/onerec_sft/beauty/42/checkpoint_best"
    )


def test_grpo_reports_missing_matching_sft_checkpoint(tmp_path):
    cell = build_cells(
        models=["onerec_grpo"],
        splits=["sports"],
        seeds=[44],
        save_dir_template=str(
            tmp_path / "{protocol}" / "{model}" / "{split}" / "{seed}"
        ),
        config_overrides={},
        evaluation_protocol="leave_one_out",
    )[0]

    assert cell.missing_dependencies() == [cell.sft_checkpoint_path]

    checkpoint = tmp_path / "leave_one_out/onerec_sft/sports/44/checkpoint_best"
    checkpoint.mkdir(parents=True)
    assert cell.missing_dependencies() == []


def test_notellm_matrix_binds_exact_adapter_protocol():
    cell = build_cells(
        models=["notellm"],
        splits=["toys"],
        seeds=[44],
        save_dir_template="out/{protocol}/{model}/{split}/{seed}",
        config_overrides={},
        evaluation_protocol="global_temporal",
    )[0]

    assert cell.config_path == "config/notellm/amazon.gin"
    assert cell.trainer_module() == "genrec.trainers.notellm_trainer"
    assert (
        'AmazonNoteLLMDataset.evaluation_protocol="global_temporal"'
        in cell.gin_overrides()
    )


def test_rpg_uses_split_specific_configs_unless_overridden():
    cells = build_cells(
        models=["rpg"],
        splits=["beauty", "sports", "toys"],
        seeds=[42],
        save_dir_template="out/{protocol}/{model}/{split}/{seed}",
        config_overrides={},
    )

    assert [cell.config_path for cell in cells] == [
        "config/rpg/amazon.gin",
        "config/rpg/amazon_sports.gin",
        "config/rpg/amazon_toys.gin",
    ]

    overridden = build_cells(
        models=["rpg"],
        splits=["sports"],
        seeds=[42],
        save_dir_template="out/{protocol}/{model}/{split}/{seed}",
        config_overrides={"rpg": "config/custom_rpg.gin"},
    )
    assert overridden[0].config_path == "config/custom_rpg.gin"


def test_home_extension_has_54_isolated_cells_with_matching_sft_dependencies():
    from scripts.run_matrix import load_preset

    spec = load_preset("amazon2014_home_20261022")
    cells = [cell for protocol in spec["protocols"] for cell in build_cells(
        spec["models"], spec["splits"], spec["seeds"],
        "out/home_extension/{protocol}/{model}/{split}/seed{seed}", {}, protocol,
        "amazon2014_home_20261022")]
    assert len(cells) == len({c.save_dir for c in cells}) == 54
    assert all(c.split == "home" for c in cells)
    assert all(c.config_path == "config/rpg/amazon_home.gin" for c in cells if c.model == "rpg")
    sft_paths = {c.save_dir + "/checkpoint_best" for c in cells if c.model == "onerec_sft"}
    assert {c.sft_checkpoint_path for c in cells if c.model == "onerec_grpo"} == sft_paths
