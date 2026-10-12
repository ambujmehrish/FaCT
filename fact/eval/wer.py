"""Word error rate with basic English text normalization (dependency-free).

For paper tables use the Whisper normalizer for exact comparability; this
module keeps the harness runnable without extra installs and is exact
Levenshtein on normalized words.
"""

from __future__ import annotations

import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> list[str]:
    text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip().split()


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def normalized_wer(ref_text: str, hyp_text: str) -> float:
    ref = normalize_text(ref_text)
    hyp = normalize_text(hyp_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return word_edit_distance(ref, hyp) / len(ref)
