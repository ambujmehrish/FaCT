"""Provenance block for result files: which code, command and host produced them.

    from fact.provenance import provenance
    result["provenance"] = provenance()

The git commit is read from the checkout that holds this package (the job's
code), with a dirty flag when tracked files differ from that commit.
Failures to read git are recorded as such, never hidden.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    out = {"git_commit": commit or "unavailable",
           "git_dirty": (bool(status) if status is not None else None),
           "argv": list(sys.argv),
           "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "time_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "python": platform.python_version()}
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        out["torch"] = None
    return out
