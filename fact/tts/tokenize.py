"""Tokenize the training shards for the TTS LMs, one output file per shard.

    python -m fact.tts.tokenize --codec fact --fact-ckpt <runs>/main_c8_rd05/step_0400000.pt \
        --shards <data>/shards/train_v2 --out <data>/tts_tokens/fact --part 0 --num-parts 64
    python -m fact.tts.tokenize --codec mimi --mimi-streams 2 \
        --shards <data>/shards/train_v2 --out <data>/tts_tokens/mimi2 --part 0 --num-parts 64

Every codec reads the SAME shard items in the same order: FaCT from the
stored mel (exactly what it was trained on), Mimi from the original audio
(the shard's audio index, scripts/build_audio_index.py; each decoded file is
checked against the stored mel length). Output <out>/<shard stem>.pt is a
list aligned with the shard's items: {"codes": int16 (n_streams, T),
"text": int16 bytes+1} or None for an item the codec could not process
(the reason goes to <out>/<shard stem>.errors.json). The LM loader keeps
only items every compared codec has, so all LMs see the same utterances.
Shards already written are skipped (restart-safe; writes are atomic).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

FACT_ONLY_STREAMS = ("content", "prosody")


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def shard_paths(shards: Path, part: int, num_parts: int) -> list[Path]:
    paths = sorted(shards.glob("shard_*.pt"))
    if not paths:
        raise SystemExit(f"no shard_*.pt in {shards}")
    return paths[part::num_parts]


class FaCTCodec:
    """content + prosody indices of the frozen FaCT encoder, from stored mel."""

    n_streams = 2

    def __init__(self, ckpt: str, device: str):
        from ..train.stage_d_wave_decoder import load_fact_frozen
        self.model = load_fact_frozen(ckpt, device)
        if self.model.cfg.encoder.ssl_model:
            raise SystemExit("SSL-input FaCT models need audio; not supported here")
        bn = self.model.bottleneck
        if bn.variant != "factorized":
            raise SystemExit("needs a factorized FaCT model (content + prosody)")
        if bn.prosody_rate_divisor != 1:
            raise SystemExit("prosody at a lower rate than content is not supported (streams must align)")
        self.vocab_sizes = [int(bn.content_codebook_size), int(bn.prosody_codebook_size)]
        self.frame_rate_hz = self.model.cfg.token_rate_hz
        self.device = device
        self.s = self.model.cfg.encoder.frame_stack
        self.pad = self.model.cfg.audio.mel_mean

    @torch.no_grad()
    def encode_items(self, items: list[dict], paths=None, batch: int = 1) -> list:
        from ..data.preprocess import dequantize_mel
        mels = [dequantize_mel(it).float() for it in items]
        out: list = [None] * len(items)
        order = sorted(range(len(mels)), key=lambda i: mels[i].shape[0])
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            lens = [mels[i].shape[0] // self.s * self.s for i in idx]
            t = max(lens)
            x = torch.full((len(idx), t, mels[idx[0]].shape[1]), self.pad)
            for j, i in enumerate(idx):
                x[j, : lens[j]] = mels[i][: lens[j]]
            # Causal encoder: right padding never changes earlier tokens.
            enc = self.model.encode_mel(x.to(self.device))
            for j, i in enumerate(idx):
                n = lens[j] // self.s
                out[i] = torch.stack([enc.content_indices[j, :n], enc.prosody_indices[j, :n]]).short().cpu()
        return out


class MimiCodec:
    """First `n` RVQ codebooks of Mimi, from the original 24 kHz audio."""

    def __init__(self, n_streams: int, device: str, workers: int = 8):
        from ..baselines.mimi import MimiTokenStreams
        self.m = MimiTokenStreams(n_streams=n_streams, device=device)
        self.n_streams = n_streams
        self.vocab_sizes = list(self.m.vocab_sizes)
        self.frame_rate_hz = self.m.frame_rate_hz
        self.sr = self.m.sample_rate
        self.hop = int(round(self.sr / self.frame_rate_hz))
        self.workers = workers

    def _load(self, path: str, mel_frames: int, mel_hop: int, mel_sr: int):
        from ..data.preprocess import load_audio
        w = torch.from_numpy(load_audio(path, self.sr))
        expect = round(w.shape[0] * mel_sr / self.sr) // mel_hop + 1
        if not -1 <= expect - mel_frames <= 2:
            raise ValueError(f"audio index misaligned: {path} gives {expect} mel frames, shard has {mel_frames}")
        return w

    @torch.no_grad()
    def encode_items(self, items: list[dict], paths: list[str], mel_hop: int, mel_sr: int,
                     batch_seconds: float = 400.0, max_items: int = 1) -> tuple[list, dict]:
        errors: dict[int, str] = {}
        wavs: list = [None] * len(items)

        def load(i):
            try:
                return i, self._load(paths[i], items[i]["mel"].shape[0], mel_hop, mel_sr), None
            except Exception as e:  # recorded per item, never silently dropped
                return i, None, f"{type(e).__name__}: {str(e)[:200]}"
        with ThreadPoolExecutor(self.workers) as ex:
            for i, w, err in ex.map(load, range(len(items))):
                wavs[i] = w
                if err:
                    errors[i] = err
        out: list = [None] * len(items)
        ok = sorted((i for i in range(len(items)) if wavs[i] is not None), key=lambda i: wavs[i].shape[0])
        b = []
        for i in ok + [None]:
            if i is not None and (not b or (len(b) < max_items and
                                            (len(b) + 1) * wavs[i].shape[0] <= batch_seconds * self.sr)):
                b.append(i)
                continue
            if b:
                n_max = max(wavs[j].shape[0] for j in b)
                x = torch.zeros(len(b), n_max)
                for k, j in enumerate(b):
                    x[k, : wavs[j].shape[0]] = wavs[j]
                codes = self.m._encode(x)  # (B, n, T); causal, so right padding is harmless
                for k, j in enumerate(b):
                    n = -(-wavs[j].shape[0] // self.hop)
                    out[j] = codes[k, :, :n].short().cpu()
            b = [i] if i is not None else []
        return out, errors


def check_batching(codec, items, codes, plist, a) -> None:
    """Batched (padded) encoding must give exactly the per-item codes."""
    picked = [i for i in range(len(items)) if codes[i] is not None][: a.check_batching]
    for i in picked:
        if plist is None:
            single = codec.encode_items([items[i]])[0]
        else:
            single = codec.encode_items([items[i]], [plist[i]], a.mel_hop, a.mel_sr)[0][0]
        if single is None or not torch.equal(single, codes[i]):
            n = 0 if single is None else int((single != codes[i]).sum())
            raise SystemExit(f"batched and single-item codes differ for item {i} ({n} tokens): "
                             "padding leaks into the encoder")
    print(f"batching check: {len(picked)} items identical", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fact.tts.tokenize")
    ap.add_argument("--codec", choices=["fact", "mimi"], required=True)
    ap.add_argument("--fact-ckpt", default=None)
    ap.add_argument("--mimi-streams", type=int, default=2)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--audio-index", type=Path, default=None, help="default <shards>/audio_index")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", type=int, default=None, help="default $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--num-parts", type=int, default=1)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--mel-sr", type=int, default=24_000, help="sample rate of the stored mel")
    ap.add_argument("--mel-hop", type=int, default=480, help="hop of the stored mel")
    ap.add_argument("--batch", type=int, default=1,
                    help="items per encoder call. 1 (default) = exactly the single-utterance codes "
                         "the TTS eval computes for prompts; padded batches change Mimi's last "
                         "frame and flip rare FaCT FSQ boundary values in floating point")
    ap.add_argument("--check-batching", type=int, default=8,
                    help="re-encode this many items of the first shard one by one and require "
                         "identical codes (padding must not change tokens); 0 = off")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    part = a.part if a.part is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= part < a.num_parts:
        ap.error(f"part {part} not in [0, {a.num_parts})")
    if a.codec == "fact" and not (a.fact_ckpt and Path(a.fact_ckpt).is_file()):
        ap.error("--codec fact needs an existing --fact-ckpt")
    if not 1 <= a.mimi_streams <= 32:
        ap.error("--mimi-streams in 1..32")
    aidx = a.audio_index or a.shards / "audio_index"
    if a.codec == "mimi" and not aidx.is_dir():
        ap.error(f"{aidx} missing (scripts/build_audio_index.py)")
    paths = shard_paths(a.shards, part, a.num_parts)[: a.limit_shards]
    a.out.mkdir(parents=True, exist_ok=True)

    from ..config import FaCTConfig  # noqa: F401  (import check before the model loads)
    codec = FaCTCodec(a.fact_ckpt, a.device) if a.codec == "fact" else MimiCodec(a.mimi_streams, a.device)
    meta = {"codec": a.codec, "n_streams": codec.n_streams, "vocab_sizes": codec.vocab_sizes,
            "frame_rate_hz": codec.frame_rate_hz, "shards": str(a.shards),
            "fact_ckpt": a.fact_ckpt if a.codec == "fact" else None,
            "mimi_streams": a.mimi_streams if a.codec == "mimi" else None}
    mp = a.out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if old != meta:
            raise SystemExit(f"{mp} was written with different settings: {old}")
    else:
        tmp = mp.with_name(f"meta.json.{os.getpid()}.partial")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, mp)

    checked = False
    done = 0
    for p in paths:
        dst = a.out / p.name
        if dst.exists():
            continue
        items = torch.load(p, map_location="cpu", weights_only=True)
        errors: dict = {}
        if a.codec == "fact":
            codes = codec.encode_items(items, batch=a.batch)
        else:
            plist = (aidx / f"{p.stem}.txt").read_text(encoding="utf-8").splitlines()
            if len(plist) != len(items):
                raise SystemExit(f"{p.stem}: {len(plist)} audio paths for {len(items)} items")
            codes, errors = codec.encode_items(items, plist, a.mel_hop, a.mel_sr, max_items=a.batch)
        if not checked and a.check_batching and a.batch > 1:
            check_batching(codec, items, codes, plist if a.codec == "mimi" else None, a)
            checked = True
        rows = [None if c is None else {"codes": c, "text": it["text"].short()} for c, it in zip(codes, items)]
        if errors:
            (a.out / f"{p.stem}.errors.json").write_text(json.dumps({str(k): v for k, v in errors.items()}))
        atomic_save(rows, dst)
        done += 1
        print(f"{p.name}: {sum(r is not None for r in rows)}/{len(rows)} items", flush=True)
    print(f"part {part}/{a.num_parts}: {done} new shards ({len(paths)} assigned)")


if __name__ == "__main__":
    main()
