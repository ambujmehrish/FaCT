"""Build a static listening page (index.html) over a rendered sample folder.

    python scripts/make_sample_page.py --dir <samples dir>

Expects (any section may be missing):
  utterances.json                      from render_codec_samples.py
  codec/<system>/<utt>.wav             one column per system ("original" first)
  tts/<lm>/<utt>.wav, <utt>.prompt.wav eval_tts.py --save-audio
  prosody/<lm>/<utt>/{base,fixed_N,free_N}.wav   eval_tts_prosody.py --save-audio
Links are relative, so the folder can be copied and opened anywhere.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def audio(rel: str) -> str:
    return f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'


def cell(d: Path, rel: str) -> str:
    """<td> with a player for rel (relative to d), or '-' when the file is missing."""
    return "<td>" + (audio(rel) if (d / rel).exists() else "-") + "</td>"


def build(d: Path) -> str:
    meta = json.loads((d / "utterances.json").read_text(encoding="utf-8"))
    utts = meta["utterances"]
    parts = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>FaCT samples</title>",
             "<style>body{font-family:sans-serif;margin:16px;background:#fff;color:#111}"
             "table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #ccc;padding:4px;"
             "vertical-align:top;font-size:13px}audio{width:190px}.t{max-width:260px}</style>",
             "<h1>FaCT listening samples</h1>",
             f"<p>LibriSpeech test-clean, {len(utts)} utterances, {meta['sample_rate']} Hz, "
             f"RMS-matched to {meta['rms_dbfs']} dBFS. FaCT uses the speaker vector of another "
             "utterance of the same speaker.</p>"]
    codec = d / "codec"
    if codec.is_dir():
        systems = sorted((p.name for p in codec.iterdir() if p.is_dir()), key=lambda s: (s != "original", s))
        parts.append("<h2>Codec reconstruction</h2><table><tr><th>text</th>"
                     + "".join(f"<th>{html.escape(s)}</th>" for s in systems) + "</tr>")
        for u in utts:
            cells = "".join(cell(d, "codec/" + s + "/" + u["stem"] + ".wav") for s in systems)
            parts.append("<tr><td class='t'>" + html.escape(u["text"]) + "</td>" + cells + "</tr>")
        parts.append("</table>")
    tts = d / "tts"
    if tts.is_dir():
        lms = sorted(p.name for p in tts.iterdir() if p.is_dir())
        parts.append("<h2>Zero-shot TTS (prompt = another utterance of the same speaker)</h2>"
                     "<table><tr><th>text</th><th>prompt</th><th>ground truth</th>"
                     + "".join(f"<th>{html.escape(m)}</th>" for m in lms) + "</tr>")
        for u in utts:
            st = u["stem"]
            prompt = next(("tts/" + m + "/" + st + ".prompt.wav" for m in lms
                           if (d / "tts" / m / (st + ".prompt.wav")).exists()), "tts/-")
            cells = "".join(cell(d, "tts/" + m + "/" + st + ".wav") for m in lms)
            parts.append("<tr><td class='t'>" + html.escape(u["text"]) + "</td>" + cell(d, prompt)
                         + cell(d, "codec/original/" + st + ".wav") + cells + "</tr>")
        parts.append("</table>")
    pro = d / "prosody"
    if pro.is_dir():
        parts.append("<h2>Prosody control in TTS</h2><p>base = one sample; fixed = stream 0 held "
                     "(FaCT: content, Mimi-2: codebook 0), stream 1 resampled; free = fully resampled.</p>")
        for m in sorted(p.name for p in pro.iterdir() if p.is_dir()):
            parts.append(f"<h3>{html.escape(m)}</h3><table>")
            for u in utts:
                ud = pro / m / u["stem"]
                if not ud.is_dir():
                    continue
                names = sorted((p.stem for p in ud.glob("*.wav")),
                               key=lambda s: (s != "base", s.split("_")[0], s))
                rels = ["prosody/" + m + "/" + u["stem"] + "/" + n + ".wav" for n in names]
                parts.append("<tr><td class='t'>" + html.escape(u["text"]) + "</td>"
                             + "".join("<td>" + html.escape(n) + "<br>" + audio(r) + "</td>"
                                       for n, r in zip(names, rels)) + "</tr>")
            parts.append("</table>")
    return "\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args(argv)
    (a.dir / "index.html").write_text(build(a.dir), encoding="utf-8")
    print(f"wrote {a.dir / 'index.html'}")


if __name__ == "__main__":
    main()
