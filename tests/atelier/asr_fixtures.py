"""Synthetic speech corpora for the ASR backend tests: each character is a tone, so CTC/seq2seq can learn it."""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CHARS = "abc"


def _tone(seconds: float, freq: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False, dtype=np.float32)
    return 0.4 * np.sin(2 * np.pi * freq * t)


def _clip(text: str, rng: np.random.Generator) -> np.ndarray:
    """Each character is 0.12 s of its own tone: a task CTC can learn in a few epochs."""
    freqs = {"a": 400, "b": 900, "c": 1500, " ": 0}
    parts = [(_tone(0.12, freqs[c]) if freqs[c] else np.zeros(1920, np.float32)) for c in text]
    return np.concatenate(parts) + 0.01 * rng.standard_normal(sum(len(p) for p in parts)).astype(np.float32)


def _make_dataset(root: Path, n_train: int = 24, n_dev: int = 8, shards: bool = True) -> Path:
    import soundfile as sf

    rng = np.random.default_rng(0)
    texts = []
    for _ in range(n_train + n_dev):
        t = "".join(rng.choice(list(CHARS + " "), size=int(rng.integers(4, 8)))).strip()
        texts.append(t if t and " " in t else "ab c")  # always an inner space: the alphabet has one
    splits = ["train"] * n_train + ["dev"] * n_dev
    langs = ["xx" if i % 2 == 0 else "yy" for i in range(len(texts))]
    rows = []
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True)
    wavs = []
    for i, (text, split, lang) in enumerate(zip(texts, splits, langs)):
        p = audio_dir / f"{i:03d}.wav"
        sf.write(str(p), _clip(text, rng), 16000, subtype="PCM_16")
        wavs.append(p)
    if shards:
        shard = audio_dir / "all-000.tar"
        with tarfile.open(shard, "w") as tf:
            for p in wavs:
                tf.add(p, arcname=p.name)
        with tarfile.open(shard) as tf:
            infos = {m.name: m for m in tf.getmembers()}
        for i, (p, text, split, lang) in enumerate(zip(wavs, texts, splits, langs)):
            blob = p.read_bytes()
            rows.append({"audio": f"audio/all-000.tar#{p.name}", "transcript": text, "split": split,
                         "language": lang, "duration_ms": int(len(blob) / 32), "sample_rate": 16000,
                         "offset": infos[p.name].offset_data, "size": infos[p.name].size,
                         "sha256": hashlib.sha256(blob).hexdigest()})
    else:
        for p, text, split, lang in zip(wavs, texts, splits, langs):
            rows.append({"audio": f"audio/{p.name}", "transcript": text, "split": split,
                         "language": lang, "duration_ms": int(p.stat().st_size / 32)})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


def _spec_text(root: Path, **over) -> str:
    spec = {
        "atelier": 1, "name": "asr-test", "task": "speech_recognition",
        "data": {"uri": str(root), "format": "asr_manifest", "split": {"train": "train", "validation": "dev"}},
        "features": {"inputs": [{"name": "audio", "type": "audio"}],
                     "output": {"name": "transcript", "type": "text"}},
        "model": "deepspeech2", "hp": "speech_recognition/ctc-fast@1",
        "overrides": {"rnn_type": "nn.GRU", "rnn_hidden_size": 32, "rnn_hidden_layers": 1,
                      "epochs": 3, "batch_size": 8, "num_workers": 0, "learning_rate": 0.003,
                      "mixed_precision": False, "dispatch": "thread", "async_eval": False,
                      "checkpoint_every_s": None},
        "seed": 7, "export": ["safetensors"],
    }
    spec.update(over)
    return yaml.safe_dump(spec)


