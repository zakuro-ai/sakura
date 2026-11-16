"""Audio helpers shared by the speech-recognition backends.

A manifest row points at one clip: ``audio`` is a path (absolute, or relative to the dataset root) or
``<shard>#<member>`` with ``offset`` / ``size`` giving the clip's byte span inside an uncompressed tar shard
(Zakuro hub packaging). A span is one seek + one read. When the row carries ``sha256`` the bytes are verified
on every read: a replay on different audio must fail, not silently train on it (CONTRACTS §1.2).

``soundfile`` is imported lazily so ``import sakura.atelier`` works without it.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import unicodedata
from pathlib import Path
from typing import Any


def read_clip(
    root: str | Path,
    ref: str,
    offset: int | None = None,
    size: int | None = None,
    sha256: str | None = None,
    sample_rate: int = 16000,
) -> Any:
    """One mono float32 waveform (numpy array) from a file or a tar-shard byte span."""
    import soundfile as sf

    root = Path(root)
    if "#" in ref:
        shard, member = ref.split("#", 1)
        shard_path = root / shard
        if offset is not None and size is not None and offset == offset and size == size:  # NaN-safe
            with open(shard_path, "rb") as fh:
                fh.seek(int(offset))
                blob = fh.read(int(size))
        else:  # no byte span recorded: scan the tar
            with tarfile.open(shard_path) as tf:
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"{member} not in {shard_path}")
                blob = extracted.read()
    else:
        path = Path(ref)
        blob = (path if path.is_absolute() else root / path).read_bytes()
    if sha256 and hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{ref}: bytes do not match the manifest sha256 (corrupt or changed audio)")
    sound, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise ValueError(f"{ref}: expected {sample_rate} Hz, got {sr} Hz")
    if sound.ndim > 1:
        sound = sound.mean(axis=1)
    return sound


# --------------------------------------------------------------------------- scoring

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes), collapse whitespace: the comparison both the
    reference and the hypothesis go through, so a model is not penalised for Whisper's capitals/commas."""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def edit_distance(a: list[Any] | str, b: list[Any] | str) -> int:
    """Levenshtein distance (insert/delete/substitute = 1) over any two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(refs: list[str], hyps: list[str], normalize: bool = True) -> tuple[float, float]:
    """Corpus-level (WER, CER) as fractions: total edits / total reference length."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    w_err = w_len = c_err = c_len = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r, h = normalize_text(r), normalize_text(h)
        rw, hw = r.split(), h.split()
        w_err += edit_distance(rw, hw)
        w_len += len(rw)
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        c_err += edit_distance(rc, hc)
        c_len += len(rc)
    return (w_err / max(w_len, 1), c_err / max(c_len, 1))
