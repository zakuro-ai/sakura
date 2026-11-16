"""speech_recognition / whisper: helpers, routing and data are checked everywhere; the training e2e needs the
backend's own stack (transformers, peft, soundfile) and the openai/whisper-tiny weights, and is skipped without them."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from sakura.atelier.audio import edit_distance, error_rates, normalize_text, read_clip
from sakura.atelier.backends import backend_name
from sakura.atelier.backends import whisper as whisper_mod
from sakura.atelier.registry import list_presets, load_models, resolve
from sakura.atelier.spec import load_spec
from sakura.atelier.types import Artifact, MaterialisedData, RunDir
from tests.atelier.asr_fixtures import _make_dataset, _spec_text

WHISPER = {
    "model": "whisper-small",
    "hp": "speech_recognition/whisper-lora@1",
    "overrides": {
        "base_model": "openai/whisper-tiny", "language": "en", "epochs": 2, "batch_size": 4,
        "grad_accum": 1, "learning_rate": 0.001, "warmup_steps": 1, "num_workers": 0,
        "fp16": False, "eval_batch_size": 4, "max_new_tokens": 24, "lora_r": 8, "lora_alpha": 16,
    },
}


# ------------------------------------------------------------------ scoring helpers

def test_normalize_text():
    assert normalize_text("Hello,  World! It's OK.") == "hello world it's ok"


@pytest.mark.parametrize("a,b,d", [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "abc", 0), (["a", "b"], ["a"], 1)])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == d


def test_error_rates_are_corpus_level_fractions():
    wer, cer = error_rates(["a b c", "hello"], ["a x c", "hallo"])
    assert (wer, cer) == (0.5, 0.25)  # 2 word errors / 4 words, 2 char errors / 8 chars


def test_error_rates_normalise_case_and_punctuation():
    assert error_rates(["Hello, World."], ["hello world"]) == (0.0, 0.0)
    assert error_rates(["Hello, World."], ["hello world"], normalize=False)[0] > 0


# ---------------------------------------------------------------------------- clips

def test_read_clip_by_span_verifies_sha256_and_detects_corruption(tmp_path):
    pytest.importorskip("soundfile")
    root = _make_dataset(tmp_path / "d")
    import pandas as pd

    row = pd.read_csv(root / "manifest.csv").iloc[0]
    wave = read_clip(root, row["audio"], row["offset"], row["size"], row["sha256"])
    assert wave.ndim == 1 and len(wave) > 1000
    shard = root / "audio" / "all-000.tar"
    data = bytearray(shard.read_bytes())
    data[int(row["offset"]) + 100] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="do not match"):
        read_clip(root, row["audio"], row["offset"], row["size"], row["sha256"])
    read_clip(root, row["audio"], row["offset"], row["size"], None)  # unverified: not noticed


def test_read_clip_plain_path_and_tar_scan(tmp_path):
    pytest.importorskip("soundfile")
    root = _make_dataset(tmp_path / "d", shards=False)
    assert len(read_clip(root, "audio/000.wav")) > 1000
    root2 = _make_dataset(tmp_path / "e")
    with tarfile.open(root2 / "audio" / "all-000.tar") as tf:
        member = tf.getmembers()[0].name
    assert len(read_clip(root2, f"audio/all-000.tar#{member}")) > 1000  # no span: scans the tar


def test_read_clip_rejects_a_wrong_sample_rate(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    sf.write(str(tmp_path / "x.wav"), np.zeros(800, dtype="float32"), 8000)
    with pytest.raises(ValueError, match="16000"):
        read_clip(tmp_path, "x.wav")


# --------------------------------------------------------------------- registry/spec

def test_whisper_models_route_to_the_whisper_backend():
    assert backend_name("speech_recognition", "whisper-small") == "whisper"
    assert backend_name("speech_recognition", "whisper-large-v3-turbo") == "whisper"
    assert backend_name("speech_recognition", "deepspeech2") == "deepspeech"  # the task default is unchanged
    models = load_models()
    assert models["whisper-small"]["exports"] == ["safetensors", "ggml"]
    assert models["whisper-large-v3-turbo"]["exports"] == ["safetensors", "ggml"]
    assert "speech_recognition/whisper-lora@1" in list_presets("speech_recognition")


def test_whisper_spec_resolves_with_overrides(tmp_path):
    pytest.importorskip("soundfile")
    root = _make_dataset(tmp_path / "d")
    resolved = resolve(load_spec(_spec_text(root, **WHISPER)))
    assert resolved.backend == "whisper"
    hp = resolved.hyperparameters
    assert hp["lora_r"] == 8 and hp["language"] == "en" and hp["target_modules"] == ["q_proj", "v_proj"]


def test_a_whisper_model_cannot_use_the_ctc_preset_family_mismatch(tmp_path):
    pytest.importorskip("soundfile")
    root = _make_dataset(tmp_path / "d")
    resolved = resolve(load_spec(_spec_text(root, model="whisper-small", hp="speech_recognition/ctc-fast@1")))
    assert resolved.backend == "whisper"  # routing is by model; the preset only supplies hyperparameters


def test_collate_pads_labels_and_strips_the_start_token():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from sakura.atelier.backends.whisper import _collate

    batch = [
        {"features": np.zeros((4, 6), dtype="float32"), "labels": [50258, 5, 6, 7], "text": "a"},
        {"features": np.zeros((4, 6), dtype="float32"), "labels": [50258, 8], "text": "b"},
    ]
    out = _collate(50258)(batch)
    assert out["input_features"].shape == (2, 4, 6)
    assert out["labels"].tolist() == [[5, 6, 7], [8, -100, -100]]
    keep = _collate(999)(batch)  # a different start token: nothing stripped
    assert keep["labels"].shape == (2, 4)


# ------------------------------------------------------------------- training e2e

def _stack_or_skip():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("soundfile")
    pytest.importorskip("torch")
    from transformers import WhisperProcessor

    try:
        WhisperProcessor.from_pretrained("openai/whisper-tiny")
    except Exception as exc:  # offline CI, no cached weights
        pytest.skip(f"openai/whisper-tiny is not available: {type(exc).__name__}")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    _stack_or_skip()
    from sakura.atelier.__main__ import main

    tmp = tmp_path_factory.mktemp("whisper")
    root = _make_dataset(tmp / "d", n_train=16, n_dev=8)
    spec_file = tmp / "spec.yaml"
    spec_file.write_text(_spec_text(root, **WHISPER), encoding="utf-8")
    out = tmp / "run"
    rc = main(["run", str(spec_file), "--out", str(out), "--device", "cpu"])
    return rc, out, root, tmp


def test_run_trains_exports_and_rescores(trained):
    rc, out, _, _ = trained
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "done" and report["backend"] == "whisper"
    assert report["metrics"]["cer"] >= 0.0 and "wer" in report["metrics"]
    for name in ("model.safetensors", "config.json", "atelier_whisper.json", "tokenizer.json"):
        assert (out / "artifacts" / name).is_file(), name
    # the feature-extractor config is processor_config.json (transformers 5) or preprocessor_config.json (4)
    assert any((out / "artifacts" / n).is_file() for n in ("processor_config.json", "preprocessor_config.json"))
    ev = report["runtime_eval"][0]
    assert ev["value"] is not None and ev["n"] == 8
    assert ev["matches_training"] is True, ev  # re-scored from the export directory alone
    lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert {("train", "loss"), ("validation", "cer"), ("validation", "wer"), ("validation", "loss")} <= {
        (m["split"], m["name"]) for m in lines}
    assert sum(1 for m in lines if m["name"] == "cer") == 2  # one validation point per epoch


def test_the_exported_adapter_is_merged_not_a_peft_checkpoint(trained):
    _, out, _, _ = trained
    from safetensors import safe_open

    with safe_open(str(out / "artifacts" / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert keys and not any("lora_" in k for k in keys)  # dense weights: loadable without peft
    assert (out / "checkpoint" / "adapter" / "adapter_config.json").is_file()  # the adapter is kept for resume


def test_predict_transcribes_from_the_artifact(trained):
    _, out, root, _ = trained
    from sakura.atelier.backends import load_backend
    from sakura.atelier.types import Artifact

    path = out / "artifacts" / "model.safetensors"
    art = Artifact(path=path, format="safetensors", sha256="", bytes=path.stat().st_size)
    res = load_backend("whisper").predict(art, {"file": str(root / "audio" / "000.wav")})
    assert isinstance(res["text"], str)


def test_resume_continues_from_the_adapter(trained):
    rc, out, root, tmp = trained
    from sakura.atelier.__main__ import main

    spec2 = tmp / "spec2.yaml"
    spec2.write_text(_spec_text(root, **WHISPER, resume_from={"uri": str(out / "checkpoint")}), encoding="utf-8")
    second = tmp / "second"
    assert main(["run", str(spec2), "--out", str(second), "--device", "cpu"]) == 0
    assert json.loads((second / "report.json").read_text())["resumed_from"] == str(out / "checkpoint")


def test_resume_without_an_adapter_is_a_clear_error(trained, tmp_path):
    _, _, root, tmp = trained
    from sakura.atelier.__main__ import main

    empty = tmp_path / "ckpt"
    empty.mkdir()
    spec = tmp_path / "s.yaml"
    spec.write_text(_spec_text(root, **WHISPER, resume_from={"uri": str(empty)}), encoding="utf-8")
    assert main(["run", str(spec), "--out", str(tmp_path / "r"), "--device", "cpu"]) == 1
    assert "no adapter" in json.loads((tmp_path / "r" / "report.json").read_text())["error"]


def test_export_rejects_other_formats():
    from sakura.atelier.backends import load_backend
    from sakura.atelier.types import TrainResult

    with pytest.raises(ValueError, match="only 'safetensors' and 'ggml'"):
        load_backend("whisper").export(TrainResult(model=None, metrics={}, train_seconds=0.0), "gguf", None)  # type: ignore[arg-type]


def test_hashlib_is_used_for_artifact_digests(trained):
    _, out, _, _ = trained
    rep = json.loads((out / "report.json").read_text())["artifacts"][0]
    assert rep["sha256"] == hashlib.sha256((out / "artifacts" / "model.safetensors").read_bytes()).hexdigest()


# ------------------------------------------------------------------- ggml export (whisper.cpp)
# Runnable without whisper.cpp, transformers or a GPU: the tool-location and quant-mapping rules, the argv the
# backend hands whisper.cpp, and the "tool missing" paths are what the fleet cannot tell us about cheaply. The
# real convert + quantize + whisper-cli round trip needs the runner image (docs/atelier/envs/whisper.txt).

def test_ggml_quant_mapping_is_the_zak_forge_profile_table():
    # Literal copy of PROFILE_TYPES in zak-forge src/forge/runner/adapters/whispercpp.py.
    profile_types = {"WQ5_0": "q5_0", "WQ8_0": "q8_0"}
    assert whisper_mod.GGML_PROFILES == profile_types
    for profile, qtype in profile_types.items():
        assert whisper_mod._ggml_quant_type(profile) == qtype
    assert whisper_mod._ggml_quant_type(None) == "q5_0"  # the reference runner's default profile
    assert whisper_mod._ggml_quant_type("") == "q5_0"
    assert whisper_mod._ggml_quant_type("q8_0") == "q8_0"  # bare whisper-quantize names pass through
    assert whisper_mod._ggml_quant_type("Q5_0") == "q5_0"
    with pytest.raises(whisper_mod.GGMLExportError, match="neither a whisper-quantize type"):
        whisper_mod._ggml_quant_type("Q4_K_M")  # a llama-quantize name is not a whisper one
    with pytest.raises(whisper_mod.GGMLExportError, match="neither"):
        whisper_mod._ggml_quant_type("q5_0; rm -rf /")


def test_ggml_argv_mirror_the_reference_runner(tmp_path):
    import sys

    argv = whisper_mod._ggml_convert_argv(tmp_path / "convert-h5-to-ggml.py", tmp_path / "merged", tmp_path / "whisper",
                                          tmp_path / "out")
    assert argv == [sys.executable, str(tmp_path / "convert-h5-to-ggml.py"), str(tmp_path / "merged"),
                    str(tmp_path / "whisper"), str(tmp_path / "out")]
    assert whisper_mod._ggml_quantize_argv(tmp_path / "whisper-quantize", tmp_path / "f16.bin", tmp_path / "q.bin",
                                           "q5_0") == [str(tmp_path / "whisper-quantize"), str(tmp_path / "f16.bin"),
                                                       str(tmp_path / "q.bin"), "q5_0"]
    argv, outputs = whisper_mod._whisper_cli_argv(tmp_path / "whisper-cli", tmp_path / "m.bin",
                                                  [tmp_path / "a.wav", tmp_path / "b.wav"], "sn", tmp_path / "o", 3)
    assert argv[:10] == [str(tmp_path / "whisper-cli"), "-m", str(tmp_path / "m.bin"), "-l", "sn", "-t", "3",
                         "-nt", "-np", "-otxt"]
    assert argv[10:] == ["-f", str(tmp_path / "a.wav"), "-of", str(tmp_path / "o" / "00000"),
                         "-f", str(tmp_path / "b.wav"), "-of", str(tmp_path / "o" / "00001")]
    assert outputs == [tmp_path / "o" / "00000.txt", tmp_path / "o" / "00001.txt"]
    assert whisper_mod._whisper_cli_argv(tmp_path / "c", tmp_path / "m", [], None, tmp_path)[0][4] == "auto"


def _fitted(tmp_path, quant=None):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    (merged / whisper_mod.META).write_text(json.dumps({"language": "sn", "normalize": True}))
    return whisper_mod._Fitted(merged, {"language": "sn"}, 0, quant)


def _result(fitted):
    return type("R", (), {"model": fitted})()


def test_ggml_export_fails_precisely_when_whisper_cpp_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(whisper_mod.WHISPER_CPP_ENV, raising=False)
    out = RunDir(tmp_path / "run").ensure()
    with pytest.raises(whisper_mod.GGMLExportError, match="SAKURA_WHISPER_CPP_DIR is not set"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path)), "ggml", out)


def test_ggml_export_fails_precisely_when_the_checkout_is_missing_pieces(tmp_path, monkeypatch):
    cpp = tmp_path / "whisper.cpp"
    monkeypatch.setenv(whisper_mod.WHISPER_CPP_ENV, str(cpp))
    monkeypatch.setenv(whisper_mod.OPENAI_WHISPER_ENV, str(tmp_path / "whisper"))
    out = RunDir(tmp_path / "run").ensure()
    with pytest.raises(whisper_mod.GGMLExportError, match="convert-h5-to-ggml.py not found"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path)), "ggml", out)
    (cpp / "models").mkdir(parents=True)
    (cpp / "models" / "convert-h5-to-ggml.py").write_text("")
    with pytest.raises(whisper_mod.GGMLExportError, match="whisper-quantize not found -- build whisper.cpp"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path)), "ggml", out)
    (cpp / "build" / "bin").mkdir(parents=True)
    (cpp / "build" / "bin" / "whisper-quantize").write_text("")
    monkeypatch.delenv(whisper_mod.OPENAI_WHISPER_ENV)
    with pytest.raises(whisper_mod.GGMLExportError, match="SAKURA_OPENAI_WHISPER_DIR is not set"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path)), "ggml", out)


def _fake_whisper_cpp(tmp_path, monkeypatch, transcripts=None):
    """A whisper.cpp checkout whose tools are a recording subprocess.run: convert writes ggml-model.bin into
    its out dir, quantize writes its destination, whisper-cli writes one transcript per -of prefix."""
    cpp = tmp_path / "whisper.cpp"
    (cpp / "models").mkdir(parents=True)
    (cpp / "models" / "convert-h5-to-ggml.py").write_text("")
    (cpp / "build" / "bin").mkdir(parents=True)
    for b in ("whisper-quantize", "whisper-cli"):
        (cpp / "build" / "bin" / b).write_text("")
    monkeypatch.setenv(whisper_mod.WHISPER_CPP_ENV, str(cpp))
    monkeypatch.setenv(whisper_mod.OPENAI_WHISPER_ENV, str(tmp_path / "openai-whisper"))
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[1].endswith("convert-h5-to-ggml.py"):
            (Path(cmd[-1]) / "ggml-model.bin").write_bytes(b"f16" * 100)
        elif cmd[0].endswith("whisper-quantize"):
            Path(cmd[2]).write_bytes(b"q" * 64)
        elif cmd[0].endswith("whisper-cli"):
            prefixes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-of"]
            for i, prefix in enumerate(prefixes):
                Path(prefix + ".txt").write_text((transcripts or [""] * len(prefixes))[i] + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(whisper_mod.subprocess, "run", fake_run)
    return cpp, calls


@pytest.mark.parametrize("hp_quant,expected", [(None, "q5_0"), ("WQ8_0", "q8_0"), ("q8_0", "q8_0"), ("WQ5_0", "q5_0")])
def test_ggml_export_converts_then_quantises_at_the_hp_type(tmp_path, monkeypatch, hp_quant, expected):
    cpp, calls = _fake_whisper_cpp(tmp_path, monkeypatch)
    out = RunDir(tmp_path / "run").ensure()
    fitted = _fitted(tmp_path, hp_quant)
    art = whisper_mod.BACKEND.export(_result(fitted), "ggml", out)
    assert art.format == "ggml" and art.path == out.artifacts / "model.ggml.bin" and art.bytes == 64
    assert art.sha256 == hashlib.sha256(b"q" * 64).hexdigest()
    convert, quantize = calls
    assert convert[1] == str(cpp / "models" / "convert-h5-to-ggml.py")
    assert convert[2:] == [str(fitted.merged_dir), str(tmp_path / "openai-whisper"), str(out.checkpoint / "ggml")]
    f16 = out.checkpoint / "ggml" / "ggml-model.bin"
    assert quantize == [str(cpp / "build" / "bin" / "whisper-quantize"), str(f16), str(art.path), expected]
    assert not f16.exists()  # the intermediate is not kept
    assert art.extras["ggml_quant"] == expected and art.extras["footprint_bytes"] == 64
    assert "export_peak_rss_bytes" in art.extras
    assert (out.artifacts / whisper_mod.META).is_file()  # decoding settings travel with the ggml file


class _FakeTokenizer:
    """What a saved WhisperTokenizerFast answers: the full vocab (base + added) and the added part."""

    def get_vocab(self):
        return {"<|endoftext|>": 50257, "<|sn|>": 50300, "a": 64, "b": 65}

    def get_added_vocab(self):
        return {"<|endoftext|>": 50257, "<|sn|>": 50300}


def _fake_transformers(monkeypatch, loaded: list[str]):
    import sys
    import types

    mod = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            loaded.append(path)
            return _FakeTokenizer()

    mod.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)


def test_ggml_export_writes_the_vocab_files_the_converter_opens_by_name(tmp_path, monkeypatch):
    # transformers >= 5 saves a fast tokenizer as tokenizer.json only; convert-h5-to-ggml.py
    # opens vocab.json and added_tokens.json by name (staging job 377f97f9 died on the first).
    _, calls = _fake_whisper_cpp(tmp_path, monkeypatch)
    loaded: list[str] = []
    _fake_transformers(monkeypatch, loaded)
    fitted = _fitted(tmp_path)
    (fitted.merged_dir / "tokenizer.json").write_text("{}")
    seen_at_convert: dict[str, bool] = {}
    real_run = whisper_mod.subprocess.run

    def run_and_look(cmd, **kw):
        if str(cmd[1]).endswith("convert-h5-to-ggml.py"):
            seen_at_convert.update({n: (fitted.merged_dir / n).is_file() for n in whisper_mod.CONVERTER_TOKENIZER_FILES})
        return real_run(cmd, **kw)

    monkeypatch.setattr(whisper_mod.subprocess, "run", run_and_look)
    whisper_mod.BACKEND.export(_result(fitted), "ggml", RunDir(tmp_path / "run").ensure())
    assert seen_at_convert == {"vocab.json": True, "added_tokens.json": True}
    assert loaded == [str(fitted.merged_dir)]
    assert json.loads((fitted.merged_dir / "vocab.json").read_text()) == {"a": 64, "b": 65}
    assert json.loads((fitted.merged_dir / "added_tokens.json").read_text()) == {"<|endoftext|>": 50257, "<|sn|>": 50300}
    assert len(calls) == 2


def test_ggml_export_keeps_vocab_files_a_slow_tokenizer_already_wrote(tmp_path, monkeypatch):
    _fake_whisper_cpp(tmp_path, monkeypatch)
    loaded: list[str] = []
    _fake_transformers(monkeypatch, loaded)
    fitted = _fitted(tmp_path)
    (fitted.merged_dir / "vocab.json").write_text('{"x": 1}')
    (fitted.merged_dir / "added_tokens.json").write_text('{"y": 2}')
    whisper_mod.BACKEND.export(_result(fitted), "ggml", RunDir(tmp_path / "run").ensure())
    assert loaded == []  # nothing to derive: the tokenizer is never loaded
    assert (fitted.merged_dir / "vocab.json").read_text() == '{"x": 1}'


def test_whisper_cli_threads_follow_the_allowed_cpus_unless_overridden(monkeypatch):
    from sakura.atelier import ort

    monkeypatch.delenv("SAKURA_WHISPER_CPP_THREADS", raising=False)
    monkeypatch.setattr(ort, "allowed_threads", lambda: 3)
    assert whisper_mod._whisper_cli_threads() == 3  # the cpuset, not os.cpu_count()
    monkeypatch.setenv("SAKURA_WHISPER_CPP_THREADS", "6")
    assert whisper_mod._whisper_cli_threads() == 6
    monkeypatch.setenv("SAKURA_WHISPER_CPP_THREADS", "zero")
    assert whisper_mod._whisper_cli_threads() == 3


def test_ggml_rescore_caps_rows_only_under_the_notebook_check(monkeypatch):
    monkeypatch.delenv("ATELIER_REPLAY", raising=False)
    monkeypatch.delenv("SAKURA_WHISPER_CPP_CHECK_ROWS", raising=False)
    assert whisper_mod._replay_check_rows(256) == 256  # the engine's own run scores every clip
    monkeypatch.setenv("ATELIER_REPLAY", "check")
    assert whisper_mod._replay_check_rows(256) == 4  # the check only proves the cell runs
    assert whisper_mod._replay_check_rows(2) == 2
    monkeypatch.setenv("SAKURA_WHISPER_CPP_CHECK_ROWS", "7")
    assert whisper_mod._replay_check_rows(256) == 7


def test_whisper_cli_transcripts_survive_a_clip_cut_mid_codepoint(tmp_path, monkeypatch):
    # whisper-cli writes raw token bytes; a clip ending on a partial UTF-8 sequence
    # (job 0c56d8e1: 0xe1 without continuation) must not fail the whole re-score.
    import sys
    import types

    import numpy as np

    _fake_whisper_cpp(tmp_path, monkeypatch)
    # CI's pure-python job has no soundfile; the clip writer is not what is under test.
    fake_sf = types.ModuleType("soundfile")
    fake_sf.write = lambda path, wave, sr, subtype=None: Path(path).write_bytes(b"RIFF")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    real_run = whisper_mod.subprocess.run

    def run_and_corrupt(cmd, **kw):
        out = real_run(cmd, **kw)
        for i, a in enumerate(cmd):
            if a == "-of":
                Path(cmd[i + 1] + ".txt").write_bytes("mapokola ni bona".encode() + b" \xe1\n")
        return out

    monkeypatch.setattr(whisper_mod.subprocess, "run", run_and_corrupt)
    waves = [np.zeros(1600, dtype="float32")]
    texts = whisper_mod._whisper_cli_transcribe(tmp_path / "m.bin", waves, {"language": "sn"}, tmp_path / "w")
    assert texts == ["mapokola ni bona �"]


def test_ggml_export_rejects_an_unknown_quant_before_running_anything(tmp_path, monkeypatch):
    _, calls = _fake_whisper_cpp(tmp_path, monkeypatch)
    with pytest.raises(whisper_mod.GGMLExportError, match="ggml_quant='Q4_K_M'"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path, "Q4_K_M")), "ggml", RunDir(tmp_path / "run").ensure())
    assert calls == []


def test_ggml_export_surfaces_the_tools_stderr(tmp_path, monkeypatch):
    _fake_whisper_cpp(tmp_path, monkeypatch)
    monkeypatch.setattr(whisper_mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom: no mel filters"))
    with pytest.raises(whisper_mod.GGMLExportError, match="convert-h5-to-ggml.py failed: boom"):
        whisper_mod.BACKEND.export(_result(_fitted(tmp_path)), "ggml", RunDir(tmp_path / "run").ensure())


def test_artifact_report_entry_carries_the_ggml_extras(tmp_path):
    from sakura.atelier.__main__ import _artifact_entry

    out = RunDir(tmp_path / "run").ensure()
    art = Artifact(path=out.artifacts / "model.ggml.bin", format="ggml", sha256="x", bytes=64,
                   extras={"ggml_quant": "q5_0", "footprint_bytes": 64, "export_peak_rss_bytes": None, "bytes": 1})
    entry = _artifact_entry(art, out)
    untested = {"size_bytes": 64, "validated": False, "eval_rows": 0, "eval_acc": None, "eval_metric": None,
                "eval_value": None, "matches_training": None, "validation_reason": "not re-scored after export"}
    assert entry == {"path": "artifacts/model.ggml.bin", "format": "ggml", "sha256": "x", "bytes": 64,
                     "ggml_quant": "q5_0", "footprint_bytes": 64, "export_peak_rss_bytes": None, **untested}
    plain = Artifact(path=out.artifacts / "model.safetensors", format="safetensors", sha256="y", bytes=1)
    assert set(_artifact_entry(plain, out)) == {"path", "format", "sha256", "bytes", *untested}


def _ggml_data(tmp_path):
    import pandas as pd

    root = _make_dataset(tmp_path / "d", n_train=4, n_dev=6)
    df = pd.read_csv(root / "manifest.csv")
    return MaterialisedData(root=root, train=df[df.split == "train"], validation=df[df.split == "dev"].reset_index(drop=True),
                            manifest_sha256="", extra={"audio_root": str(root), "audio_col": "audio", "text_col": "transcript"})


def test_runtime_eval_ggml_without_whisper_cpp_reports_why_and_never_raises(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    monkeypatch.delenv(whisper_mod.WHISPER_CPP_ENV, raising=False)
    data = _ggml_data(tmp_path)
    art = Artifact(path=tmp_path / "model.ggml.bin", format="ggml", sha256="x", bytes=1)
    ev = whisper_mod.BACKEND.runtime_eval(art, data)
    assert (ev.format, ev.runtime, ev.metric, ev.value, ev.n, ev.matches_training) == ("ggml", "whisper.cpp", "cer", None, 0, None)
    assert "SAKURA_WHISPER_CPP_DIR is not set" in ev.reason
    cpp = tmp_path / "whisper.cpp"
    (cpp / "build" / "bin").mkdir(parents=True)
    monkeypatch.setenv(whisper_mod.WHISPER_CPP_ENV, str(cpp))
    ev = whisper_mod.BACKEND.runtime_eval(art, data)
    assert ev.value is None and "whisper-cli not found" in ev.reason


def test_runtime_eval_ggml_scores_cer_through_one_whisper_cli_process_capped_by_replay_rows(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    data = _ggml_data(tmp_path)
    refs = list(data.validation["transcript"])
    hyps = [refs[0], refs[1].upper(), "zzz"]  # 2 exact (one up to normalisation), 1 wrong
    cpp, calls = _fake_whisper_cpp(tmp_path, monkeypatch, transcripts=hyps)
    monkeypatch.setenv("ATELIER_REPLAY_EVAL_ROWS", "3")
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / whisper_mod.META).write_text(json.dumps({"language": "sn", "normalize": True}))
    art = Artifact(path=art_dir / "model.ggml.bin", format="ggml", sha256="x", bytes=1)
    ev = whisper_mod.BACKEND.runtime_eval(art, data)
    assert ev.reason is None, ev.reason
    assert ev.n == 3 and ev.matches_training is None  # the engine applies the ±0.05 quantised tolerance
    assert ev.value == pytest.approx(error_rates(refs[:3], hyps, True)[1]) and 0 < ev.value <= 1
    (cli,) = calls
    assert cli[:5] == [str(cpp / "build" / "bin" / "whisper-cli"), "-m", str(art.path), "-l", "sn"]
    assert cli.count("-f") == 3 and all(Path(cli[i + 1]).suffix == ".wav" for i, a in enumerate(cli) if a == "-f")


def test_runtime_eval_ggml_reports_a_failing_cli(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    data = _ggml_data(tmp_path)
    _fake_whisper_cpp(tmp_path, monkeypatch)
    monkeypatch.setattr(whisper_mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "failed to load model"))
    ev = whisper_mod.BACKEND.runtime_eval(Artifact(path=tmp_path / "m.bin", format="ggml", sha256="x", bytes=1), data)
    assert ev.value is None and "whisper-cli failed (exit 2): failed to load model" in ev.reason


def test_whisper_spec_can_request_ggml(tmp_path):
    pytest.importorskip("soundfile")
    root = _make_dataset(tmp_path / "d")
    resolved = resolve(load_spec(_spec_text(root, **WHISPER, export=["safetensors", "ggml"])))
    assert resolved.spec.export == ["safetensors", "ggml"]
    with pytest.raises(ValueError, match="cannot export"):
        resolve(load_spec(_spec_text(root, **WHISPER, export=["gguf"])))
