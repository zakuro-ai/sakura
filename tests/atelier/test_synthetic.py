"""``synthetic://`` data sources (SAK-0019): uri validation, deterministic generation of every kind, a
learnable signal in each, and the normal materialise path (split, manifest, sha256) on top.

Only numpy and pandas are needed: the generators write csv, jsonl, PNG and WAV by hand."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
import wave
import zlib

import numpy as np
import pytest

import sakura.atelier.__main__ as cli
from sakura.atelier import synthetic
from sakura.atelier.data import materialise
from sakura.atelier.spec import dump_spec, load_spec

#: kind -> (data.format, the rest of a spec that reads it)
SPECS = {
    "tabular": ("csv", """
task: tabular_classification
features:
  inputs: [{name: f0, type: number}, {name: f1, type: number}, {name: f2, type: number}, {name: f3, type: number}]
  output: {name: label, type: category}
model: hist_gradient_boosting
hp: tabular/hist_gradient_boosting@1
export: [joblib]
"""),
    "timeseries": ("csv", """
task: time_series_forecasting
features:
  inputs: [{name: value, type: timeseries}]
  output: {name: value, type: timeseries}
model: ludwig_forecaster
hp: time_series_forecasting/ecd@1
export: [onnx]
"""),
    "image": ("image_folder", """
task: image_classification
features:
  inputs: [{name: image, type: image}]
  output: {name: label, type: category}
model: resnet18
hp: image_classification/fast@1
export: [onnx]
"""),
    "audio": ("audio_folder", """
task: audio_classification
features:
  inputs: [{name: audio, type: audio}]
  output: {name: label, type: category}
model: stacked_cnn_audio
hp: audio_classification/fast@1
export: [onnx]
"""),
    "text": ("csv", """
task: text_classification
features:
  inputs: [{name: text, type: text}]
  output: {name: label, type: category}
model: ludwig_text
hp: text_classification/ecd@1
export: [onnx]
"""),
    "instruct": ("jsonl", """
task: text_generation
features:
  inputs: [{name: prompt, type: text}]
  output: {name: completion, type: text}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
"""),
    "preference": ("jsonl", """
task: text_generation
features:
  inputs: [{name: prompt, type: text}]
  output: {name: chosen, type: text}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
"""),
    "triples": ("csv", """
task: tabular_classification
features:
  inputs: [{name: head, type: category}, {name: relation, type: category}]
  output: {name: tail, type: category}
model: hist_gradient_boosting
hp: tabular/hist_gradient_boosting@1
export: [joblib]
"""),
    "asr": ("asr_manifest", """
task: speech_recognition
features:
  inputs: [{name: audio, type: audio}]
  output: {name: transcript, type: text}
model: deepspeech2
hp: speech_recognition/ctc-fast@1
export: [safetensors]
"""),
    "risk": ("csv", """
task: risk_simulation
model: scvar-copula
hp: risk_simulation/scvar@1
export: [json]
"""),
    "detection": ("yolo", """
task: object_detection
features:
  inputs: [{name: image, type: image}]
  output: {name: boxes, type: boxes}
model: rtdetr_r18
hp: object_detection/fast@1
export: [onnx]
"""),
    "image_regression": ("csv", """
task: image_regression
features:
  inputs: [{name: image, type: image}]
  output: {name: target, type: number}
model: siglip2_b16_256
hp: image_regression/siglip2@1
export: [onnx]
"""),
    "decision": ("jsonl", """
task: decision
model: hf-decision-lm
hp: decision/lora-forge@1
overrides: {base_model: Qwen/Qwen3.5-2B-Base}
export: [safetensors]
"""),
}
ROWS = {"risk": 240, "triples": 60, "decision": 60}
#: data.split per kind when it is not a validation fraction: risk and decision take none (the engine splits
#: a decision dataset by state), a YOLO tree names its own train / val directories
SPLITS = {"risk": "", "decision": "", "detection": ", split: {train: train, validation: val}"}
#: The yolo materialiser writes the absolute `path:` of its copy into data.yaml, which is pinned: a YOLO
#: dataset's manifest hash is that of one data dir (true of a file:// YOLO tree too). See the detection tests.
SAME_HASH_IN_ANY_DATA_DIR = sorted(k for k in SPECS if k != "detection")


def _uri(kind, seed=1, **params):
    query = "&".join(f"{k}={v}" for k, v in {"rows": ROWS.get(kind, 48), "seed": seed, **params}.items())
    return f"synthetic://{kind}?{query}"


def _spec_text(kind, uri):
    fmt, rest = SPECS[kind]
    split = SPLITS.get(kind, ", split: {validation_fraction: 0.25}")
    return f'atelier: 1\nname: syn-{kind}\ndata: {{uri: "{uri}", format: {fmt}{split}}}\nseed: 5{rest}'


def _tree_hash(path):
    """One hash over a generated file or directory: names and bytes."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(path.parent)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def test_every_data_format_kind_is_covered_here():
    assert set(SPECS) == set(synthetic.KINDS)


# --------------------------------------------------------------------------- the uri


def test_a_uri_is_parsed_into_the_kind_and_every_parameter_with_defaults():
    src = synthetic.parse_uri("synthetic://tabular?rows=50&seed=3&classes=4")
    assert src.kind == "tabular" and src.formats == ("csv",)
    assert src.params == {"rows": 50, "seed": 3, "task": "classification", "features": 4, "classes": 4}
    assert synthetic.parse_uri("synthetic://image").params == {"rows": 200, "seed": 0, "classes": 2, "size": 32}
    assert synthetic.is_synthetic("synthetic://text") and not synthetic.is_synthetic("file:///x")


@pytest.mark.parametrize("uri, match", [
    ("synthetic://video?rows=5", "unknown synthetic kind 'video'"),
    ("synthetic://?rows=5", "unknown synthetic kind"),
    ("synthetic://tabular?rowz=5", "no parameter 'rowz'"),
    ("synthetic://image?features=3", "no parameter 'features'"),
    ("synthetic://tabular?rows=5&rows=6", "given twice"),
    ("synthetic://tabular?rows=many", "rows must be a whole number"),
    ("synthetic://tabular?rows=-4", "rows must be a whole number"),
    ("synthetic://tabular?rows=0", "rows must be between 1 and"),
    ("synthetic://tabular?rows=9999999999", "rows must be between"),
    ("synthetic://tabular?task=ranking", "task must be one of"),
    ("synthetic://tabular?task=regression&classes=3", "classes does not apply"),
    ("synthetic://tabular?rows=3&classes=5", "at least classes"),
    ("synthetic://image?size=4", "size must be between 8 and 256"),
    ("synthetic://audio?classes=1", "classes must be between 2"),
    ("synthetic://triples?rows=5&entities=20", "rows must be between entities - 1"),
    ("synthetic://triples?rows=100&entities=3&relations=1", "rows must be between"),
    ("synthetic://risk?rows=12", "rows must be between 120"),
    ("synthetic://asr?words=40", "words must be between 2 and 16"),
    ("synthetic://tabular?rows", "bad query field"),
    ("synthetic://detection?size=16", "size must be between 32 and 640"),
    ("synthetic://detection?classes=1", "classes must be between 2 and 20"),
    ("synthetic://detection?rows=1", "rows must be between 2 and"),
    ("synthetic://detection?rows=3&classes=5", "at least classes"),
    ("synthetic://detection?ms=100", "no parameter 'ms'"),
    ("synthetic://image_regression?size=8", "size must be between 16 and 256"),
    ("synthetic://image_regression?classes=2", "no parameter 'classes'"),
    ("synthetic://decision?rows=49", "rows must be between 50 and 50000"),
    ("synthetic://decision?rows=50001", "rows must be between 50 and 50000"),
    ("synthetic://decision?options=1", "options must be between 2 and 26"),
    ("synthetic://decision?state=xml", "state must be one of"),
])
def test_a_bad_uri_says_what_is_wrong(uri, match):
    with pytest.raises(ValueError, match=match):
        synthetic.parse_uri(uri)


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_a_good_spec_validates_for_every_kind(kind):
    spec = load_spec(_spec_text(kind, _uri(kind)))
    assert spec.data.uri.startswith("synthetic://") and dump_spec(spec).count("synthetic://") == 1


def test_the_spec_refuses_an_unknown_kind_or_parameter():
    with pytest.raises(ValueError, match="unknown synthetic kind 'tabluar'"):
        load_spec(_spec_text("tabular", "synthetic://tabluar?rows=50"))
    with pytest.raises(ValueError, match="no parameter 'colour'"):
        load_spec(_spec_text("tabular", "synthetic://tabular?rows=50&colour=red"))


def test_the_spec_refuses_a_format_the_kind_is_not_written_as():
    with pytest.raises(ValueError, match="synthetic://tabular is written as data.format csv, not 'parquet'"):
        load_spec(_spec_text("tabular", _uri("tabular")).replace("format: csv", "format: parquet"))
    load_spec(_spec_text("risk", _uri("risk")).replace("format: csv", "format: parquet"))  # risk: either
    for kind, wrong in (("detection", "image_folder"), ("image_regression", "parquet"), ("decision", "csv")):
        right = SPECS[kind][0]
        with pytest.raises(ValueError, match=f"synthetic://{kind} is written as data.format {right}, not '{wrong}'"):
            load_spec(_spec_text(kind, _uri(kind)).replace(f"format: {right}", f"format: {wrong}"))


def test_validate_rejects_a_bad_synthetic_uri_without_running(tmp_path, capsys):
    good, bad = tmp_path / "good.yaml", tmp_path / "bad.yaml"
    good.write_text(_spec_text("tabular", _uri("tabular")))
    bad.write_text(_spec_text("tabular", "synthetic://tabular?rows=50&seeed=1"))
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(bad)]) == 1
    assert "no parameter 'seeed'" in capsys.readouterr().err
    assert sorted(tmp_path.iterdir()) == [bad, good]  # nothing was generated


# --------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_the_same_uri_gives_the_same_bytes_and_another_seed_does_not(kind, tmp_path):
    if kind == "risk":
        pytest.importorskip("scipy")
    fmt = SPECS[kind][0]
    a = synthetic.generate(_uri(kind), tmp_path / "a", fmt)
    b = synthetic.generate(_uri(kind), tmp_path / "b", fmt)
    other = synthetic.generate(_uri(kind, seed=2), tmp_path / "c", fmt)
    assert _tree_hash(a) == _tree_hash(b)
    assert _tree_hash(other) != _tree_hash(a)
    assert a.name == b.name == other.name  # the path does not depend on the parameters


@pytest.mark.parametrize("kind", SAME_HASH_IN_ANY_DATA_DIR)
def test_the_same_uri_gives_the_same_manifest_hash_through_materialise(kind, tmp_path):
    if kind == "risk":
        pytest.importorskip("scipy")
    spec = load_spec(_spec_text(kind, _uri(kind)))
    a = materialise(spec, tmp_path / "a")
    b = materialise(spec, tmp_path / "b")
    again = materialise(spec, tmp_path / "a")  # a replay into the same data dir regenerates the same files
    other = materialise(load_spec(_spec_text(kind, _uri(kind, seed=2))), tmp_path / "c")
    assert a.manifest_sha256 == b.manifest_sha256 == again.manifest_sha256
    assert other.manifest_sha256 != a.manifest_sha256
    # the pin the engine writes into resolved.yaml verifies on a replay
    pinned = spec.model_copy(update={"data": spec.data.model_copy(update={"sha256": a.manifest_sha256})})
    assert materialise(pinned, tmp_path / "d").manifest_sha256 == a.manifest_sha256
    with pytest.raises(ValueError, match="data.sha256 mismatch"):
        materialise(pinned.model_copy(update={"data": pinned.data.model_copy(update={"uri": _uri(kind, seed=2)})}),
                    tmp_path / "e")


def test_a_different_parameter_gives_different_data(tmp_path):
    a = synthetic.generate("synthetic://tabular?rows=40&seed=1", tmp_path / "a", "csv")
    b = synthetic.generate("synthetic://tabular?rows=40&seed=1&features=5", tmp_path / "b", "csv")
    assert a.read_bytes() != b.read_bytes()


def test_generating_again_replaces_what_was_there(tmp_path):
    synthetic.generate("synthetic://image?rows=12&seed=1&size=8", tmp_path, "image_folder")
    root = synthetic.generate("synthetic://image?rows=4&seed=1&size=8", tmp_path, "image_folder")
    assert len(list(root.rglob("*.png"))) == 4  # not 12: no file of the earlier dataset is left


def test_the_generators_module_imports_nothing_but_numpy():
    import subprocess
    import sys

    # Loaded by path: `import sakura` itself pulls in torch, which says nothing about this module.
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('synthetic_alone', {synthetic.__file__!r})\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['synthetic_alone'] = m; spec.loader.exec_module(m)\n"
            "m.parse_uri('synthetic://image?rows=4')\n"
            "print([n for n in ('PIL', 'soundfile', 'torch', 'pandas', 'scipy', 'sakura') if n in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# --------------------------------------------------------------------------- what each kind holds


def _rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def test_tabular_classification_is_balanced_and_separable(tmp_path):
    rows = _rows(synthetic.generate("synthetic://tabular?rows=300&seed=4&classes=3&features=5", tmp_path, "csv"))
    assert list(rows[0]) == ["f0", "f1", "f2", "f3", "f4", "label"] and len(rows) == 300
    labels = np.array([r["label"] for r in rows])
    assert sorted(set(labels)) == ["c0", "c1", "c2"] and all((labels == c).sum() == 100 for c in set(labels))
    x = np.array([[float(r[f"f{i}"]) for i in range(5)] for r in rows])
    # nearest class centroid, fitted on the first half, scored on the second: far above the 1/3 of chance
    centres = {c: x[:150][labels[:150] == c].mean(axis=0) for c in set(labels)}
    names = sorted(centres)
    pred = [names[int(np.argmin([np.linalg.norm(v - centres[c]) for c in names]))] for v in x[150:]]
    assert (np.array(pred) == labels[150:]).mean() > 0.7


def test_tabular_regression_is_a_noisy_linear_target(tmp_path):
    rows = _rows(synthetic.generate("synthetic://tabular?rows=200&seed=4&task=regression", tmp_path, "csv"))
    assert list(rows[0]) == ["f0", "f1", "f2", "f3", "target"]
    x = np.array([[float(r[f"f{i}"]) for i in range(4)] for r in rows])
    y = np.array([float(r["target"]) for r in rows])
    w, *_ = np.linalg.lstsq(x[:100], y[:100], rcond=None)
    assert 1 - ((y[100:] - x[100:] @ w) ** 2).sum() / ((y[100:] - y[100:].mean()) ** 2).sum() > 0.9  # R^2


def test_timeseries_is_one_series_in_time_order_with_a_period(tmp_path):
    rows = _rows(synthetic.generate("synthetic://timeseries?rows=240&seed=4&period=12", tmp_path, "csv"))
    assert [int(r["t"]) for r in rows] == list(range(240))
    v = np.array([float(r["value"]) for r in rows])
    detrended = v - np.polyval(np.polyfit(np.arange(240), v, 1), np.arange(240))
    assert np.corrcoef(detrended[:-12], detrended[12:])[0, 1] > 0.9  # repeats every 12 rows
    assert v[-24:].mean() > v[:24].mean()  # and trends up


def _png_pixels(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
    assert (depth, colour) == (8, 2)
    idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + 3 * width)
    return raw[:, 1:].reshape(height, width, 3)


def test_images_are_class_folders_of_pngs_separable_by_colour(tmp_path):
    root = synthetic.generate("synthetic://image?rows=30&seed=4&classes=3&size=16", tmp_path, "image_folder")
    assert sorted(d.name for d in root.iterdir()) == ["c0", "c1", "c2"]
    assert all(len(list((root / c).glob("*.png"))) == 10 for c in ("c0", "c1", "c2"))
    brightest = {}
    for c in ("c0", "c1", "c2"):
        pixels = np.stack([_png_pixels(p) for p in sorted((root / c).glob("*.png"))])
        assert pixels.shape == (10, 16, 16, 3)
        brightest[c] = int(pixels.reshape(-1, 3).mean(axis=0).argmax())  # the class's dominant channel
    assert brightest == {"c0": 0, "c1": 1, "c2": 2}  # red, green, blue


def test_a_generated_png_opens_in_an_imaging_library(tmp_path):
    image = pytest.importorskip("PIL.Image")
    root = synthetic.generate("synthetic://image?rows=2&seed=4&size=24", tmp_path, "image_folder")
    img = image.open(next(root.rglob("*.png")))
    assert img.size == (24, 24) and img.mode == "RGB"


def _wav(path):
    with wave.open(str(path)) as fh:
        assert (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) == (1, 2, 16_000)
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(float)


def _pitch(samples):
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / 16_000)[int(spectrum.argmax())])


def test_audio_is_class_folders_of_16k_wavs_with_one_pitch_per_class(tmp_path):
    root = synthetic.generate("synthetic://audio?rows=12&seed=4&classes=3&ms=250", tmp_path, "audio_folder")
    for c, hz in (("c0", 220.0), ("c1", 330.0), ("c2", 495.0)):
        clips = sorted((root / c).glob("*.wav"))
        assert len(clips) == 4
        for clip in clips:
            samples = _wav(clip)
            assert len(samples) == 4000 and abs(_pitch(samples) - hz) < 8


def test_text_classes_have_their_own_words(tmp_path):
    rows = _rows(synthetic.generate("synthetic://text?rows=120&seed=4&classes=3", tmp_path, "csv"))
    assert list(rows[0]) == ["text", "label"] and {r["label"] for r in rows} == {"c0", "c1", "c2"}
    vocab = {c: set(w for r in rows if r["label"] == c for w in r["text"].split()) for c in ("c0", "c1", "c2")}
    own = {c: vocab[c] - set.union(*(vocab[o] for o in vocab if o != c)) for c in vocab}
    assert all(len(words) >= 4 for words in own.values())
    # every text holds a word only its class uses
    assert all(set(r["text"].split()) & own[r["label"]] for r in rows)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_instruct_is_a_mapping_a_model_can_learn(tmp_path):
    rows = _jsonl(synthetic.generate("synthetic://instruct?rows=64&seed=4&keys=8", tmp_path, "jsonl"))
    assert all(set(r) == {"prompt", "completion"} for r in rows) and len(rows) == 64
    answer: dict[str, set] = {}
    for r in rows:
        key = r["prompt"].rstrip(":?").split()[-1]
        answer.setdefault(key, set()).add(r["completion"])
    assert len(answer) == 8 and all(len(v) == 1 for v in answer.values())  # one answer per key, always
    assert len({next(iter(v)) for v in answer.values()}) == 8  # and no two keys share one


def test_preference_rows_prefer_the_right_answer_over_another_keys(tmp_path):
    rows = _jsonl(synthetic.generate("synthetic://preference?rows=64&seed=4&keys=8", tmp_path, "jsonl"))
    assert all(set(r) == {"prompt", "chosen", "rejected"} and r["chosen"] != r["rejected"] for r in rows)
    chosen = {}
    for r in rows:
        chosen.setdefault(r["prompt"].rstrip(":?").split()[-1], set()).add(r["chosen"])
    assert all(len(v) == 1 for v in chosen.values())
    codes = {next(iter(v)) for v in chosen.values()}
    assert all(r["rejected"] in codes for r in rows)  # a plausible wrong answer, not noise
    # the same table as `instruct` at the same seed
    instruct = _jsonl(synthetic.generate("synthetic://instruct?rows=64&seed=4&keys=8", tmp_path / "i", "jsonl"))
    assert {r["completion"] for r in instruct} == codes


def test_triples_are_a_graph_with_multi_hop_paths(tmp_path):
    rows = _rows(synthetic.generate("synthetic://triples?rows=40&seed=4&entities=12&relations=3", tmp_path, "csv"))
    assert list(rows[0]) == ["head", "relation", "tail"] and len(rows) == 40
    triples = {(r["head"], r["relation"], r["tail"]) for r in rows}
    assert len(triples) == 40 and all(h != t for h, _, t in triples)
    assert {r for _, r, _ in triples} <= {"r0", "r1", "r2"}
    edges: dict[str, set] = {}
    for h, _, t in triples:
        edges.setdefault(h, set()).add(t)
    # e11 is 11 hops from e0 along the chain; a breadth-first walk reaches every entity
    seen, frontier, hops = {"e0"}, {"e0"}, 0
    while frontier:
        frontier = {t for h in frontier for t in edges.get(h, ())} - seen
        seen |= frontier
        hops += bool(frontier)
    assert seen == {f"e{i}" for i in range(12)} and hops >= 2


def test_asr_is_a_manifest_of_clips_with_one_tone_per_word(tmp_path):
    root = synthetic.generate("synthetic://asr?rows=10&seed=4&words=3", tmp_path, "asr_manifest")
    rows = _rows(root / "manifest.csv")
    assert list(rows[0]) == ["audio", "transcript", "duration_ms"] and len(rows) == 10
    for r in rows:
        words = r["transcript"].split()
        assert 1 <= len(words) <= 3 and set(words) <= {"alpha", "bravo", "charlie"} and "_" not in r["transcript"]
        samples = _wav(root / r["audio"])
        assert int(r["duration_ms"]) == round(1000 * len(samples) / 16_000) == 150 * len(words) + 50 * (len(words) - 1)
        first = {"alpha": 300.0, "bravo": 450.0, "charlie": 600.0}[words[0]]
        assert abs(_pitch(samples[:2400]) - first) < 10  # the first 150 ms is the first word's tone


def test_risk_delegates_to_the_risk_modules_own_generator(tmp_path):
    pytest.importorskip("scipy")
    from sakura.atelier import risk

    root = synthetic.generate("synthetic://risk?rows=240&seed=9", tmp_path / "a", "csv")
    risk.make_synthetic_dataset(tmp_path / "b", seed=9, months=240, fmt="csv")
    assert sorted(p.name for p in root.iterdir()) == ["exposure.csv", "factors.csv", "losses.csv"]
    assert all((root / n).read_bytes() == (tmp_path / "b" / n).read_bytes() for n in ("exposure.csv", "factors.csv", "losses.csv"))


def _boxes(path):
    return [(int(c), *(float(v) for v in rest)) for c, *rest in (line.split() for line in path.read_text().splitlines())]


def test_detection_is_a_yolo_tree_whose_labels_are_the_rectangles_drawn(tmp_path):
    root = synthetic.generate("synthetic://detection?rows=20&seed=4&classes=3&size=48", tmp_path, "yolo")
    assert sorted(p.name for p in root.iterdir()) == ["data.yaml", "images", "labels"]
    assert (root / "data.yaml").read_text() == "train: images/train\nval: images/val\nnames:\n  0: c0\n  1: c1\n  2: c2\n"
    stems = {split: sorted(p.stem for p in (root / "images" / split).glob("*.png")) for split in ("train", "val")}
    assert (len(stems["train"]), len(stems["val"])) == (16, 4)  # one image in five is val
    assert stems["train"] + stems["val"] == [f"{i:06d}" for i in range(20)]
    seen = set()
    for split in ("train", "val"):
        assert sorted(p.stem for p in (root / "labels" / split).glob("*.txt")) == stems[split]  # a label per image
        for stem in stems[split]:
            pixels = _png_pixels(root / "images" / split / f"{stem}.png").astype(float)
            assert pixels.shape == (48, 48, 3)
            boxes = _boxes(root / "labels" / split / f"{stem}.txt")
            assert 1 <= len(boxes) <= 3
            covered = np.zeros((48, 48), dtype=bool)
            for c, cx, cy, w, h in boxes:
                seen.add(c)
                x0, y0, x1, y1 = (round(v * 48) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                assert 0 <= x0 < x1 <= 48 and 0 <= y0 < y1 <= 48
                assert not covered[y0:y1, x0:x1].any()  # rectangles do not overlap
                covered[y0:y1, x0:x1] = True
                # the box is its class's colour: red, green, blue for three classes
                assert int(pixels[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0).argmax()) == c
            assert pixels[~covered].mean() < 80 < pixels[covered].max()  # the rest is the dark background
    assert seen == {0, 1, 2}


def test_detection_materialises_as_a_yolo_dataset_read_in_place(tmp_path):
    yaml = pytest.importorskip("yaml")
    spec = load_spec(_spec_text("detection", "synthetic://detection?rows=20&seed=2&classes=3&size=32"))
    data = materialise(spec, tmp_path / "data")
    assert data.labels == ["c0", "c1", "c2"]
    doc = yaml.safe_load(open(data.extra["yolo_data_yaml"]))
    assert doc["path"] == str((tmp_path / "data" / "yolo").resolve()) and doc["train"] == "images/train"
    # the generated tree is the materialised one: every file is pinned once
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    paths = [e["path"] for e in manifest]
    assert len(paths) == 41 and all(p.startswith("yolo/") for p in paths)  # data.yaml + 20 images + 20 labels
    # a replay into the same data dir gives the same hash; another seed does not. (Another data dir changes
    # data.yaml's absolute `path:`, hence the hash: the yolo materialiser's behaviour for any YOLO dataset.)
    assert materialise(spec, tmp_path / "data").manifest_sha256 == data.manifest_sha256
    other = load_spec(_spec_text("detection", "synthetic://detection?rows=20&seed=3&classes=3&size=32"))
    assert materialise(other, tmp_path / "data").manifest_sha256 != data.manifest_sha256


def test_limits_cap_a_synthetic_detection_dataset_with_list_files(tmp_path):
    spec = load_spec(_spec_text("detection", "synthetic://detection?rows=20&seed=2&size=32")
                     + "limits: {max_train_rows: 6, max_eval_rows: 2}\n")
    data = materialise(spec, tmp_path / "data")
    assert data.extra["limits_applied"] == {"max_train_rows": 6, "max_eval_rows": 2,
                                            "rows_before": {"train": 16, "validation": 4},
                                            "rows_after": {"train": 6, "validation": 2}}


def test_image_regression_is_a_table_of_images_whose_target_is_the_square_drawn(tmp_path):
    root = synthetic.generate("synthetic://image_regression?rows=24&seed=4&size=40", tmp_path, "csv")
    assert root.is_dir() and sorted(p.name for p in root.iterdir()) == ["images", "table.csv"]
    rows = _rows(root / "table.csv")
    assert list(rows[0]) == ["image", "target"] and len(rows) == 24
    assert [r["image"] for r in rows] == [f"images/{i:06d}.png" for i in range(24)]
    targets = np.array([float(r["target"]) for r in rows])
    assert targets.min() >= 0.2 and targets.max() <= 0.8 and len(set(targets)) > 5
    bright = []
    for r in rows:
        pixels = _png_pixels(root / r["image"]).astype(float)
        assert pixels.shape == (40, 40, 3)
        mask = pixels.mean(axis=2) > 130  # the square (about 220) against the background (about 40)
        assert mask[20, 20] and mask.sum() == round(float(r["target"]) * 40) ** 2  # centred, of that side
        bright.append(pixels.mean())
    assert np.corrcoef(bright, targets)[0, 1] > 0.9  # so the target can be read off the picture


def test_image_regression_materialises_through_the_image_table_path(tmp_path):
    spec = load_spec(_spec_text("image_regression", "synthetic://image_regression?rows=20&seed=2&size=16"))
    data = materialise(spec, tmp_path / "data")
    assert len(data.train) + len(data.validation) == 20 and list(data.train.columns) == ["image", "target"]
    assert all(str(p).endswith(".png") and open(p, "rb").read(4) == b"\x89PNG" for p in data.train["image"])
    assert data.train["target"].between(0.2, 0.8).all()
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert "table.csv" in [e["path"] for e in manifest] and len(manifest) == 21  # the table and every image


def _decision_ticket(state):
    if isinstance(state, dict):
        return state["category"], state["priority"], state["amount"], state["repeat"]
    fields = dict(part.split() for part in state.split("; "))
    return fields["category"], int(fields["priority"]), int(fields["amount"]), fields["repeat"] == "yes"


@pytest.mark.parametrize("state", ["json", "text"])
def test_decision_rows_are_typed_questions_answered_by_a_function_of_the_state(state, tmp_path):
    rows = _jsonl(synthetic.generate(f"synthetic://decision?rows=120&seed=4&options=4&state={state}", tmp_path, "jsonl"))
    assert len(rows) == 120 and all(set(r) == {"state", "questions", "answers"} for r in rows)
    assert all(isinstance(r["state"], dict if state == "json" else str) for r in rows)
    assert len({json.dumps(r["state"], sort_keys=True) for r in rows}) == 120  # no state twice
    questions = rows[0]["questions"]
    assert all(r["questions"] == questions for r in rows)
    assert {q: v["type"] for q, v in questions.items()} == {"queue": "choice", "urgency": "score", "escalate": "noul"}
    assert len(questions["queue"]["options"]) == 4 and questions["urgency"]["levels"] == ["low", "medium", "high"]
    queue_of: dict[str, set] = {}
    for r in rows:
        category, priority, amount, repeat = _decision_ticket(r["state"])
        assert set(r["answers"]) == {"queue", "urgency", "escalate"}
        queue_of.setdefault(category, set()).add(r["answers"]["queue"])
        assert r["answers"]["urgency"] == ("low" if priority <= 2 else "medium" if priority == 3 else "high")
        assert r["answers"]["escalate"] is (repeat and amount >= 5000)
    # one queue per category, no two categories share one, and every option is somebody's answer
    assert len(queue_of) == 4 and all(len(v) == 1 for v in queue_of.values())
    assert {next(iter(v)) for v in queue_of.values()} == set(questions["queue"]["options"])
    assert {r["answers"]["urgency"] for r in rows} == {"low", "medium", "high"}
    assert {r["answers"]["escalate"] for r in rows} == {True, False}


def test_decision_rows_are_accepted_whole_by_the_decision_task(tmp_path):
    from sakura.atelier import decision

    path = synthetic.generate("synthetic://decision?rows=80&seed=4", tmp_path, "jsonl")
    prepared = decision.prepare_rows(decision.load_rows(path), seed=5)
    assert prepared.stats["usable_rows"] == prepared.stats["states"] == 80
    assert prepared.stats["duplicates"] == prepared.stats["dropped_unanswered"] == 0
    assert prepared.stats["split_rows"] == {"eval": 8, "calibration": 8, "train": 64}
    kinds = {i.kind for row in prepared.train for i in decision.instances(row)}
    assert kinds == {"choice", "score", "noul"}


def test_decision_materialises_split_by_state_and_pins_the_one_file(tmp_path):
    spec = load_spec(_spec_text("decision", "synthetic://decision?rows=100&seed=2"))
    data = materialise(spec, tmp_path / "data")
    assert (len(data.train), len(data.validation), len(data.extra["calibration"])) == (80, 10, 10)
    assert data.extra["decision"]["stats"]["duplicates"] == 0
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert [e["path"] for e in manifest] == ["synthetic-decision.jsonl"]


# --------------------------------------------------------------------------- through materialise


def test_a_synthetic_table_flows_through_the_normal_split_and_manifest(tmp_path):
    spec = load_spec(_spec_text("tabular", "synthetic://tabular?rows=200&seed=2&classes=4"))
    data = materialise(spec, tmp_path / "data")
    assert len(data.train) + len(data.validation) == 200 and data.labels == ["c0", "c1", "c2", "c3"]
    assert data.extra["data_split"]["mode"] == "stratified_fraction"
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert [e["path"] for e in manifest] == ["synthetic-tabular.csv"]  # one pinned table, no stray copy
    assert manifest[0]["sha256"] == hashlib.sha256((tmp_path / "data" / "synthetic-tabular.csv").read_bytes()).hexdigest()


def test_synthetic_media_is_pinned_file_by_file(tmp_path):
    spec = load_spec(_spec_text("image", "synthetic://image?rows=20&seed=2&size=8"))
    data = materialise(spec, tmp_path / "data")
    assert len(data.train) + len(data.validation) == 20 and data.labels == ["c0", "c1"]
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert len(manifest) == 20 and all(e["path"].endswith(".png") for e in manifest)


def test_limits_cap_a_synthetic_dataset_like_any_other(tmp_path):
    spec = load_spec(_spec_text("text", "synthetic://text?rows=200&seed=2&classes=4")
                     + "limits: {max_train_rows: 16, max_eval_rows: 8}\n")
    data = materialise(spec, tmp_path / "data")
    assert (len(data.train), len(data.validation)) == (16, 8)
    assert set(data.train["label"]) == set(data.validation["label"]) == {"c0", "c1", "c2", "c3"}
    assert data.extra["limits_applied"]["rows_before"] == {"train": 152, "validation": 48}


def test_the_spec_object_the_caller_holds_keeps_the_synthetic_uri(tmp_path):
    spec = load_spec(_spec_text("instruct", _uri("instruct")))
    data = materialise(spec, tmp_path / "data")
    assert spec.data.uri == _uri("instruct")
    assert all(set(r) == {"prompt", "completion"} for r in data.train) and len(data.train) + len(data.validation) == 48
