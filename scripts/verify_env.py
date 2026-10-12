"""Offline-readiness verifier: prove the environment is correct BEFORE jobs.

Run on the login node after setup, and again inside a compute job (with
--expect-gpu) before the pilot - compute nodes have no internet, so a
missing cache there costs a queued allocation.

    source $FACT_BASE/fact_env.sh
    python scripts/verify_env.py --base $FACT_BASE [--expect-gpu] [--run-tests]

Checks:
  paths     every cache env var set and OUTSIDE $HOME; evidence of past
            $HOME leaks (~/.cache/huggingface|descript|torch >50MB) flagged
  space     free GB at base; warn when base is on $SCRATCH (purge policy)
  imports   every required package importable
  models    HF snapshots present locally (mimi, vocos[, xcodec2]) with
            local_files_only - no network touched; DAC weights file; whisper
  data      eval set (wav+txt pairs) and smoke data staged
  gpu       (--expect-gpu) CUDA visible + a tiny tensor op on device
Exit code 0 only if every FAIL-level check passes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []  # (level, name, detail)


def record(ok: bool, name: str, detail: str = "", warn_only: bool = False) -> None:
    level = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((level, name, detail))
    print(f"  [{level}] {name}" + (f" - {detail}" if detail else ""))


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def check_paths(home: Path) -> None:
    required = ["FACT_BASE", "FACT_CACHE", "HF_HOME", "XDG_CACHE_HOME",
                "PIP_CACHE_DIR", "TORCH_HOME"]
    for var in required:
        val = os.environ.get(var)
        if not val:
            record(False, f"env {var}", "not set - source fact_env.sh")
            continue
        p = Path(val).resolve()
        inside_home = p == home or home in p.parents
        record(not inside_home, f"env {var}",
               f"{p}" + (" (INSIDE $HOME!)" if inside_home else ""))
    for leak in [".cache/huggingface", ".cache/descript", ".cache/torch",
                 ".cache/pip"]:
        mb = dir_size_mb(home / leak)
        record(mb < 50, f"$HOME leak ~/{leak}", f"{mb:.0f} MB",
               warn_only=True)


def check_space(base: Path) -> None:
    usage = shutil.disk_usage(base)
    record(usage.free > 50e9, "free space at base",
           f"{usage.free / 1e9:.0f} GB free")
    scratch = os.environ.get("SCRATCH")
    if scratch and str(base.resolve()).startswith(str(Path(scratch).resolve())):
        record(True, "base on $SCRATCH",
               "purge policy applies - re-verify before long runs", warn_only=True)


def check_imports() -> None:
    required = ["torch", "torchaudio", "numpy", "yaml", "soundfile",
                "pyworld", "pesq", "pystoi", "fact"]
    optional = ["transformers", "dac", "vocos", "datasets", "whisper"]
    for mod in required:
        try:
            importlib.import_module(mod)
            record(True, f"import {mod}")
        except Exception as e:
            record(False, f"import {mod}", str(e))
    for mod in optional:
        try:
            importlib.import_module(mod)
            record(True, f"import {mod} (optional)")
        except Exception as e:
            record(True, f"import {mod} (optional)", f"missing: {e}", warn_only=True)


def check_models(cache: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        record(False, "huggingface_hub", str(e))
        return
    for repo, required in [("kyutai/mimi", True),
                           ("charactr/vocos-mel-24khz", True),
                           ("HKUSTAudio/xcodec2", False)]:
        try:
            path = snapshot_download(repo, local_files_only=True)
            record(True, f"offline snapshot {repo}", path)
        except Exception:
            record(False, f"offline snapshot {repo}",
                   "not cached - scripts/download_checkpoints.py",
                   warn_only=not required)
    dac_w = cache / "dac" / "weights_24khz.pth"
    record(dac_w.exists(), "DAC weights in $FACT_CACHE", str(dac_w))
    whisper_files = list((cache / "whisper").glob("*.pt")) if (cache / "whisper").exists() else []
    record(bool(whisper_files), "whisper cached (optional)",
           ", ".join(f.name for f in whisper_files) or "none", warn_only=True)


def check_data(base: Path) -> None:
    eval_dir = base / "data" / "eval" / "librispeech_test_clean"
    wavs = list(eval_dir.glob("*.wav")) if eval_dir.exists() else []
    txts = list(eval_dir.glob("*.txt")) if eval_dir.exists() else []
    record(len(wavs) >= 50 and len(txts) >= 50, "eval set staged",
           f"{len(wavs)} wavs / {len(txts)} transcripts in {eval_dir}")
    smoke = base / "smoke" / "data"
    n_smoke = len(list(smoke.glob("*.wav"))) if smoke.exists() else 0
    record(n_smoke > 0, "smoke data staged", f"{n_smoke} wavs in {smoke}")


def check_gpu() -> None:
    import torch
    ok = torch.cuda.is_available()
    detail = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}" if ok else "no CUDA"
    record(ok, "GPU visible", detail)
    if ok:
        x = torch.randn(64, 64, device="cuda")
        record(bool(torch.isfinite(x @ x).all()), "GPU tensor op")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("FACT_BASE"))
    ap.add_argument("--expect-gpu", action="store_true")
    ap.add_argument("--run-tests", action="store_true",
                    help="also run the pytest suite")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.base:
        raise SystemExit("--base or $FACT_BASE required (source fact_env.sh)")
    base = Path(args.base)
    cache = Path(os.environ.get("FACT_CACHE", base / "fact_cache"))
    home = Path.home().resolve()

    print("== paths =="); check_paths(home)
    print("== space =="); check_space(base)
    print("== imports =="); check_imports()
    print("== models (offline) ==")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # prove it works offline
    check_models(cache)
    print("== data =="); check_data(base)
    if args.expect_gpu:
        print("== gpu =="); check_gpu()
    if args.run_tests:
        import subprocess
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                            cwd=Path(__file__).resolve().parent.parent).returncode
        record(rc == 0, "pytest suite")

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    summary = {"pass": len(RESULTS) - len(fails) - len(warns),
               "warn": len(warns), "fail": len(fails),
               "results": [{"level": l, "name": n, "detail": d} for l, n, d in RESULTS]}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['pass']} pass / {summary['warn']} warn / {summary['fail']} fail")
    if fails:
        print("ENVIRONMENT NOT READY - fix FAILs before submitting jobs.")
        raise SystemExit(1)
    print("ENVIRONMENT READY" + (" (warnings above are advisory)" if warns else ""))


if __name__ == "__main__":
    main()
