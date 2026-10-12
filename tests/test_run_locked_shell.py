import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run-locked-179.sh"


@pytest.mark.skipif(shutil.which("flock") is None, reason="requires util-linux flock")
def test_gpu_handoff_waits_for_a_stale_cuda_process(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "nvidia-count"
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ! -e \"$FAKE_NVIDIA_COUNT\" ]]; then\n"
        "  touch \"$FAKE_NVIDIA_COUNT\"\n"
        "  echo '123, stale-trainer, 11840'\n"
        "fi\n"
    )
    fake_nvidia.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0",
            "GENREC_LOCK_DIR": str(tmp_path / "locks"),
            "GENREC_GPU_DRAIN_WAIT": "3",
            "GENREC_HOLD_MAX_MB": "0",
            "FAKE_NVIDIA_COUNT": str(state),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--gpus", "0", "--cell", "test.drain", "--", "true"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "waiting 2s for teardown" in result.stderr


@pytest.mark.skipif(shutil.which("flock") is None, reason="requires util-linux flock")
def test_gpu_handoff_still_rejects_a_persistent_occupant(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text("#!/usr/bin/env bash\necho '123, other-job, 4096'\n")
    fake_nvidia.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0",
            "GENREC_LOCK_DIR": str(tmp_path / "locks"),
            "GENREC_GPU_DRAIN_WAIT": "0",
            "GENREC_HOLD_MAX_MB": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--gpus", "0", "--cell", "test.reject", "--", "true"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 75
    assert "refusing to share" in result.stderr
