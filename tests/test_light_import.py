"""The queue (run on a login node) must not import torch: a slow import from
the venv on Lustre stalled every runner tick (2026-11-03)."""

import subprocess
import sys
from pathlib import Path


def test_queue_import_does_not_load_torch():
    code = "import sys, fact.ops.queue; assert 'torch' not in sys.modules, 'torch imported'"
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lazy_package_attributes():
    import fact
    from fact.model import FaCT
    assert fact.FaCT is FaCT and fact.tiny_config is not None
