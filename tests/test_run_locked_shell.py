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


def _run_with_named_small_occupant(tmp_path, occupant_mask):
    """nvidia-smi names a live container PID whose own GPU mask is occupant_mask."""
    occupant = subprocess.Popen(
        ["sleep", "30"], env={**os.environ, "CUDA_VISIBLE_DEVICES": occupant_mask}
    )
    try:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_nvidia = fake_bin / "nvidia-smi"
        fake_nvidia.write_text(
            f"#!/usr/bin/env bash\necho '{occupant.pid}, python, 836'\n"
        )
        fake_nvidia.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CUDA_VISIBLE_DEVICES": "0",
                "GENREC_LOCK_DIR": str(tmp_path / "locks"),
                "GENREC_GPU_DRAIN_WAIT": "0",
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT), "--gpus", "0", "--cell", "test.collision", "--", "true"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        occupant.kill()
        occupant.wait()


@pytest.mark.skipif(
    shutil.which("flock") is None or not Path("/proc/self/environ").exists(),
    reason="requires util-linux flock and /proc",
)
def test_hold_pid_colliding_with_a_process_masked_to_another_gpu_is_allowed(tmp_path):
    result = _run_with_named_small_occupant(tmp_path, occupant_mask="3")
    assert result.returncode == 0, result.stderr
    assert "allowing auto-yield hold.py signature" in result.stderr


@pytest.mark.skipif(
    shutil.which("flock") is None or not Path("/proc/self/environ").exists(),
    reason="requires util-linux flock and /proc",
)
def test_small_named_process_on_the_same_gpu_still_blocks(tmp_path):
    result = _run_with_named_small_occupant(tmp_path, occupant_mask="0")
    assert result.returncode == 75
    assert "refusing to share" in result.stderr
