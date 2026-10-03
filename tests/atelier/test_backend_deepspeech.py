"""speech_recognition: spec/registry/data are checked everywhere; the training e2e needs
asr-deepspeech (the backend's own venv) and is skipped without it."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sakura.atelier.backends import backend_name
from sakura.atelier.data import materialise
from sakura.atelier.registry import list_presets, load_models, resolve
from sakura.atelier.spec import load_spec

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
    import hashlib

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


# ----------------------------------------------------------- registry / spec / data

def test_registry_knows_speech_recognition():
    assert backend_name("speech_recognition", "deepspeech2") == "deepspeech"
    assert "deepspeech2" in load_models()
    assert "speech_recognition/ctc-fast@1" in list_presets("speech_recognition")


def test_spec_resolves(tmp_path):
    root = _make_dataset(tmp_path / "d")
    resolved = resolve(load_spec(_spec_text(root)))
    assert resolved.backend == "deepspeech"
    assert resolved.hyperparameters["rnn_hidden_size"] == 32  # override beat the preset
    assert resolved.hyperparameters["dispatch"] == "thread"


def test_only_safetensors_is_exportable(tmp_path):
    root = _make_dataset(tmp_path / "d")
    with pytest.raises(ValueError, match="cannot export"):
        resolve(load_spec(_spec_text(root, export=["onnx"])))


@pytest.mark.parametrize("shards", [True, False])
def test_materialise_asr_manifest(tmp_path, shards):
    root = _make_dataset(tmp_path / "d", shards=shards)
    data = materialise(load_spec(_spec_text(root)), tmp_path / "data")
    assert len(data.train) == 24 and len(data.validation) == 8
    assert data.labels[0] == "_" and " " in data.labels and set(data.labels[1:]) <= set(CHARS + " ")
    assert list(data.train["duration_ms"]) == sorted(data.train["duration_ms"])  # bucketing order
    assert data.extra["audio_root"] == str(root)
    assert len(data.manifest_sha256) == 64


def test_language_split_expression_and_pin_changes_with_selection(tmp_path):
    root = _make_dataset(tmp_path / "d")
    both = materialise(load_spec(_spec_text(root)), tmp_path / "a")
    xx = materialise(
        load_spec(_spec_text(root, data={"uri": str(root), "format": "asr_manifest",
                                         "split": {"train": "train@xx", "validation": "dev@xx"}})),
        tmp_path / "b")
    assert set(xx.train["language"]) == {"xx"} and len(xx.train) == 12
    assert xx.manifest_sha256 != both.manifest_sha256  # a different selection is a different dataset


def test_data_sha256_mismatch_fails_a_replay(tmp_path):
    root = _make_dataset(tmp_path / "d")
    data = materialise(load_spec(_spec_text(root)), tmp_path / "a")
    pinned = _spec_text(root, data={"uri": str(root), "format": "asr_manifest", "sha256": data.manifest_sha256,
                                    "split": {"train": "train", "validation": "dev"}})
    assert materialise(load_spec(pinned), tmp_path / "b").manifest_sha256 == data.manifest_sha256
    bad = pinned.replace(data.manifest_sha256, "a" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        materialise(load_spec(bad), tmp_path / "c")


def test_unknown_split_expression_is_an_error(tmp_path):
    root = _make_dataset(tmp_path / "d")
    spec = _spec_text(root, data={"uri": str(root), "format": "asr_manifest",
                                  "split": {"train": "train@zz", "validation": "dev"}})
    with pytest.raises(ValueError, match="selects no rows"):
        materialise(load_spec(spec), tmp_path / "a")


# ------------------------------------------------------------------- training e2e

# tests/conftest.py stubs `gnutools` with a MagicMock (sakura itself only needs RecNamespace);
# the real asr-deepspeech imports the real package, so drop the stubs before importing it.
import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

for _name in [m for m in sys.modules if m == "gnutools" or m.startswith("gnutools.")]:
    if isinstance(sys.modules[_name], MagicMock):
        del sys.modules[_name]

asr = pytest.importorskip("asr_deepspeech")
pytest.importorskip("safetensors")


def test_run_trains_exports_and_rescores(tmp_path):
    from sakura.atelier.__main__ import main

    root = _make_dataset(tmp_path / "d")
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(_spec_text(root), encoding="utf-8")
    out = tmp_path / "run"
    assert main(["run", str(spec_file), "--out", str(out), "--device", "cpu"]) == 0

    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "done" and report["backend"] == "deepspeech"
    assert 0.0 <= report["metrics"]["cer"] <= 1.5
    art = report["artifacts"][0]
    assert art["format"] == "safetensors"
    for name in ("model.safetensors", "model_config.json", "labels.json", "preprocess.json"):
        assert (out / "artifacts" / name).is_file()
    ev = report["runtime_eval"][0]
    assert ev["value"] is not None and ev["n"] == 8
    assert ev["matches_training"] is True, ev  # re-scored from the export alone

    lines = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    names = {(l["split"], l["name"]) for l in lines}
    assert {("train", "loss"), ("validation", "cer"), ("validation", "wer")} <= names
    assert sum(1 for l in lines if l["name"] == "cer") == 3  # one point per epoch


def test_predict_transcribes_from_the_artifact(tmp_path):
    from sakura.atelier.backends import load_backend
    from sakura.atelier.types import Artifact

    root = _make_dataset(tmp_path / "d")
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(_spec_text(root), encoding="utf-8")
    out = tmp_path / "run"
    from sakura.atelier.__main__ import main

    assert main(["run", str(spec_file), "--out", str(out), "--device", "cpu"]) == 0
    path = out / "artifacts" / "model.safetensors"
    art = Artifact(path=path, format="safetensors", sha256="", bytes=path.stat().st_size)
    res = load_backend("deepspeech").predict(art, {"file": str(root / "audio" / "000.wav")})
    assert isinstance(res["text"], str)


def test_resume_continues_for_more_epochs(tmp_path):
    from sakura.atelier.__main__ import main

    root = _make_dataset(tmp_path / "d")
    first = tmp_path / "first"
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(_spec_text(root), encoding="utf-8")
    assert main(["run", str(spec_file), "--out", str(first), "--device", "cpu"]) == 0
    spec2 = _spec_text(root, resume_from={"uri": str(first / "checkpoint")})
    spec2_file = tmp_path / "spec2.yaml"
    spec2_file.write_text(spec2, encoding="utf-8")
    second = tmp_path / "second"
    assert main(["run", str(spec2_file), "--out", str(second), "--device", "cpu"]) == 0
    epochs = sorted({json.loads(l)["step"] for l in (second / "metrics.jsonl").read_text().splitlines()
                     if json.loads(l)["name"] == "cer"})
    assert epochs == [3, 4, 5]  # 3 more epochs after the first run's 0..2


def test_a_run_whose_evaluation_fails_is_reported_failed(tmp_path, monkeypatch):
    import asr_deepspeech.trainers.deepspeech_trainer as T
    from sakura.atelier.__main__ import main

    def broken(*a, **k):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(T, "evaluate", broken)
    root = _make_dataset(tmp_path / "d")
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(_spec_text(root), encoding="utf-8")
    out = tmp_path / "run"
    assert main(["run", str(spec_file), "--out", str(out), "--device", "cpu"]) == 1
    report = json.loads((out / "report.json").read_text())
    # sync evaluation re-raises at once; async evaluation swallows per-epoch errors and the
    # backend's "no completed evaluation" guard then fails the run instead of reporting done
    assert report["status"] == "failed"
    assert "eval exploded" in report["error"] or "without a single completed" in report["error"]


def test_the_alphabet_keeps_its_space_end_to_end(tmp_path):
    from sakura.atelier.backends.deepspeech import _build_model

    model = _build_model(["_", " ", "a", "b"], {"rnn_type": "nn.GRU", "rnn_hidden_size": 8,
                                                "rnn_hidden_layers": 1}, {"sample_rate": 16000,
                                                "window_size": 0.02, "window_stride": 0.01})
    assert model.num_classes == 4


def test_backend_opts_out_of_global_deterministic_algorithms(monkeypatch):
    import torch

    from sakura.atelier.__main__ import _deterministic
    from sakura.atelier.backends import load_backend

    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: calls.append(a))
    assert load_backend("deepspeech").deterministic_algorithms is False
    _deterministic(1, load_backend("deepspeech").deterministic_algorithms)
    assert calls == []  # CTC must not be routed to cuDNN
    _deterministic(1)  # every other backend keeps the old behaviour
    assert calls == [(True,)]
