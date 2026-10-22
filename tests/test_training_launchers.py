"""Exercise command routing without starting a model or requiring CUDA."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,args,executable,config", [
    ("train_rpg.sh", ["beauty", "1"], "python", "config/rpg/amazon.gin"),
    ("train_rpg.sh", ["sports", "1"], "python", "config/rpg/amazon_sports.gin"),
    ("train_rpg.sh", ["toys", "2"], "accelerate", "config/rpg/amazon_toys.gin"),
    ("train_onerec_grpo.sh", [], "python", "config/onerec/amazon/grpo.gin"),
    ("train_onerec_grpo.sh", ["beauty", "1", "--gin", "train_grpo.seed=44"],
     "python", "config/onerec/amazon/grpo.gin"),
])
def test_launchers_route_to_existing_configs_and_forward_arguments(
        tmp_path, script, args, executable, config):
    argv_file = tmp_path / "argv.json"
    for command in ("python", "accelerate"):
        shim = tmp_path / command
        shim.write_text(f"#!{sys.executable}\n"
                        "import json, os, sys\n"
                        "with open(os.environ['GENREC_TEST_ARGV'], 'w') as f:\n"
                        "    json.dump(sys.argv, f)\n")
        shim.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               GENREC_TEST_ARGV=str(argv_file))
    subprocess.run(["bash", f"scripts/{script}", *args], cwd=ROOT, env=env, check=True)
    actual = json.loads(argv_file.read_text())
    assert Path(actual[0]).name == executable
    assert config in actual and (ROOT / config).is_file()
    assert "--split" in actual
    if "--gin" in args:
        assert actual[-2:] == args[-2:]
    if executable == "accelerate":
        assert "--multi_gpu" in actual
        assert "--config_file" not in actual  # No dependency on a missing YAML.


def test_launcher_rejects_invalid_gpu_count_before_launch():
    result = subprocess.run(["bash", "scripts/train_onerec_grpo.sh", "beauty", "0"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "positive integer" in result.stderr
