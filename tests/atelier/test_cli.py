"""python -m sakura.atelier run|validate|presets -- the runner's whole
interface (CONTRACTS §2). No ludwig here: these exercise validate/presets and
run()'s failure paths, which must all work without a training stack."""

import json

from sakura.atelier.__main__ import main

SPEC = """
atelier: 1
name: tiny-tabular
task: tabular_classification
data:
  uri: file://{csv}
  format: csv
  split: {{validation_fraction: 0.25}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""


def _write_csv(path):
    import pandas as pd
    pd.DataFrame({"x": range(20), "y": ["a", "b"] * 10}).to_csv(path, index=False)


def test_validate_accepts_a_well_formed_spec(tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    _write_csv(csv_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC.format(csv=csv_path))

    assert main(["validate", str(spec_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_rejects_an_unconfirmed_guess(tmp_path):
    csv_path = tmp_path / "t.csv"
    _write_csv(csv_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC.format(csv=csv_path) + "guessed: [features.output.name]\n")

    assert main(["validate", str(spec_path)]) == 1


def test_presets_lists_known_ids(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "image_classification/fast@1" in out
    assert "tabular/ecd@1" in out


def test_presets_filters_by_task(capsys):
    main(["presets", "--task", "tabular_classification"])
    out = capsys.readouterr().out
    assert "tabular/ecd@1" in out and "image_classification/fast@1" not in out


def test_run_on_an_unparsable_spec_still_writes_a_failed_report(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text("not: [valid, atelier spec")
    out_dir = tmp_path / "out"

    rc = main(["run", str(spec_path), "--out", str(out_dir)])
    assert rc == 1
    report = json.loads((out_dir / "report.json").read_text())
    assert report["status"] == "failed" and report["error"]


def test_run_refuses_unconfirmed_guesses_with_a_failed_report(tmp_path):
    csv_path = tmp_path / "t.csv"
    _write_csv(csv_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC.format(csv=csv_path) + "guessed: [features.output.name]\n")
    out_dir = tmp_path / "out"

    rc = main(["run", str(spec_path), "--out", str(out_dir)])
    assert rc == 1
    report = json.loads((out_dir / "report.json").read_text())
    assert report["status"] == "failed"
    assert "unconfirmed" in report["error"]


def _write_report(run_dir, backend="fake", status="done", artifacts=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps({
        "status": status, "backend": backend,
        "artifacts": artifacts if artifacts is not None else [
            {"path": "artifacts/model.onnx", "format": "onnx", "sha256": "a" * 64, "bytes": 1},
        ],
    }))


def test_predict_prints_one_json_object_on_stdout(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write_report(run_dir)

    class FakeBackend:
        def predict(self, artifact, inputs):
            return {"top": [{"label": "cat", "score": 0.9}]}

    import sakura.atelier.__main__ as cli_module
    monkeypatch.setattr(cli_module, "load_backend", lambda name: FakeBackend())

    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps({"img": "some/path.png"}))

    rc = cli_module.main(["predict", str(run_dir), "--input", str(input_path)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"top": [{"label": "cat", "score": 0.9}]}


def test_predict_with_file_builds_a_file_input(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write_report(run_dir)
    raw_file = tmp_path / "pic.png"
    raw_file.write_bytes(b"fake-png")

    captured = {}

    class FakeBackend:
        def predict(self, artifact, inputs):
            captured["inputs"] = inputs
            return {"top": []}

    import sakura.atelier.__main__ as cli_module
    monkeypatch.setattr(cli_module, "load_backend", lambda name: FakeBackend())

    rc = cli_module.main(["predict", str(run_dir), "--file", str(raw_file)])
    assert rc == 0
    assert captured["inputs"] == {"file": str(raw_file)}


def test_predict_on_a_failed_run_reports_an_error(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_report(run_dir, status="failed")

    import sakura.atelier.__main__ as cli_module
    rc = cli_module.main(["predict", str(run_dir), "--input", str(tmp_path / "missing.json")])
    assert rc == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_predict_requires_either_input_or_file(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write_report(run_dir)

    class FakeBackend:
        def predict(self, artifact, inputs):
            return {}

    import sakura.atelier.__main__ as cli_module
    monkeypatch.setattr(cli_module, "load_backend", lambda name: FakeBackend())

    rc = cli_module.main(["predict", str(run_dir)])
    assert rc == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_run_on_an_unresolvable_spec_fails_before_materialising(tmp_path):
    csv_path = tmp_path / "t.csv"
    _write_csv(csv_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(SPEC.format(csv=csv_path).replace("model: ludwig_ecd", "model: not-a-model"))
    out_dir = tmp_path / "out"

    rc = main(["run", str(spec_path), "--out", str(out_dir)])
    assert rc == 1
    report = json.loads((out_dir / "report.json").read_text())
    assert report["status"] == "failed" and "unknown model" in report["error"]



def test_a_backend_that_prints_cannot_corrupt_the_predict_json(tmp_path, monkeypatch, capsys):
    # Ultralytics prints "Loading model.onnx for ONNX Runtime inference..." to
    # stdout; the runner parses stdout as the response.
    run_dir = tmp_path / "run"
    _write_report(run_dir)

    class ChattyBackend:
        def predict(self, artifact, inputs):
            print("Loading model.onnx for ONNX Runtime inference...")
            return {"boxes": [], "width": 4, "height": 4}

    import sakura.atelier.__main__ as cli_module
    monkeypatch.setattr(cli_module, "load_backend", lambda name: ChattyBackend())
    f = tmp_path / "x.jpg"
    f.write_bytes(b"jpg")
    assert cli_module.main(["predict", str(run_dir), "--file", str(f)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"boxes": [], "width": 4, "height": 4}
    assert "Loading model.onnx" in captured.err



def test_serve_loads_once_and_answers_each_line(tmp_path, monkeypatch, capsys):
    import io

    run_dir = tmp_path / "run"
    _write_report(run_dir)
    loads = []

    class CountingBackend:
        def predict(self, artifact, inputs):
            print("library chatter on stdout")  # must not reach the answers
            return {"echo": inputs}

    import sakura.atelier.__main__ as cli_module
    monkeypatch.setattr(cli_module, "load_backend", lambda name: loads.append(name) or CountingBackend())
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"x": 1}))
    monkeypatch.setattr("sys.stdin", io.StringIO(
        json.dumps({"input": str(a)}) + "\n" + json.dumps({"file": "/tmp/clip.wav"}) + "\n"
        + json.dumps({"nothing": 1}) + "\n"))
    assert cli_module.main(["serve", str(run_dir)]) == 0
    answers = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert answers[0] == {"echo": {"x": 1}}
    assert answers[1] == {"echo": {"file": "/tmp/clip.wav"}}
    assert "error" in answers[2]
    assert len(loads) == 1


def test_serve_on_a_failed_run_says_why_and_stops(tmp_path, capsys):
    import sakura.atelier.__main__ as cli_module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(json.dumps({"status": "failed", "error": "boom"}))
    assert cli_module.main(["serve", str(run_dir)]) == 1
    assert "not 'done'" in json.loads(capsys.readouterr().out)["error"]
