import subprocess
import sys


def test_control_plane_import_does_not_load_gpu_dependencies():
    script = """
import sys
from nanovllm.engine.scheduler import Scheduler
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'triton' not in sys.modules
assert 'flash_attn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_engine_module_import_does_not_load_optional_runtime_dependencies():
    script = """
import sys
from nanovllm.engine.llm_engine import LLMEngine
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'triton' not in sys.modules
assert 'flash_attn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
