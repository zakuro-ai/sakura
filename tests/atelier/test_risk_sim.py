"""Task ``risk_simulation`` (model ``scvar-copula``, backend ``risk_sim``): the estimators, the spec rules,
and a whole run through the CLI -- report, ``results.json``, the bit-exact replay and the what-if ``predict``.

Everything runs for real (numpy/scipy, CPU, a few seconds): the dataset is the synthetic one from
``risk.make_synthetic_dataset``, written as csv so no parquet engine is needed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

import sakura.atelier.__main__ as cli  # noqa: E402
from sakura.atelier import risk  # noqa: E402
from sakura.atelier.backends import backend_name, load_backend  # noqa: E402
from sakura.atelier.data import materialise  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import Artifact  # noqa: E402

SPEC = """
atelier: 1
name: scvar-test
task: risk_simulation
model: scvar-copula
hp: risk_simulation/scvar@1
overrides: {{iterations: [2000, 8000]}}
data: {{uri: "{uri}", format: csv}}
export: [json]
seed: 39321
"""
LAYERS = [{"name": "hedging", "share": 0.25, "cap": 9.0}, {"name": "buffer", "share": 0.2, "cap": 5.0}]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("risk") / "ds"
    risk.make_synthetic_dataset(root, seed=7, fmt="csv")
    return root


@pytest.fixture(scope="module")
def tables(dataset):
    return risk.load_tables(risk.find_tables(dataset))


def _opt(**hp):
    return risk.risk_options({"iterations": [4000], "defence": LAYERS, **hp}, seed=11)


# ------------------------------------------------------------------ estimators


def test_the_same_seed_gives_the_same_bytes_and_another_seed_does_not(tables):
    a = risk.run_all(tables, _opt())
    b = risk.run_all(tables, _opt())
    assert risk.results_digest(a) == risk.results_digest(b)
    other = risk.run_all(tables, risk.risk_options({"iterations": [4000], "defence": LAYERS}, seed=12))
    assert risk.results_digest(other) != risk.results_digest(a)


def test_net_never_exceeds_gross_and_the_tail_grows_with_alpha(tables):
    r = risk.run_all(tables, _opt())
    for method in ("historical", "evt", "monte_carlo"):
        rows = [r[method][risk._key(a)] for a in r["alphas"]]
        for row in rows:
            assert 0 <= row["net"] <= row["gross"] + 1e-9
            assert row["net_es"] >= row["net"] - 1e-9
        assert [x["net"] for x in rows] == sorted(x["net"] for x in rows)


def test_delta_normal_understates_the_fat_tailed_simulation(tables):
    # The synthetic factors are Student-t with joint jumps: a normal fit to the net losses sits below the
    # copula at 99 %. This is the comparison the method table exists to show.
    r = risk.run_all(tables, _opt())
    assert r["delta_normal"]["a0.99"]["net"] < r["monte_carlo"]["a0.99"]["net"]


def test_defence_layers_apply_in_order_and_respect_their_caps():
    gross = np.array([0.0, 10.0, 100.0])
    total, per = risk.defence_cascade(gross, tuple(risk.DefenceLayer(**layer) for layer in LAYERS))
    assert per["hedging"].tolist() == [0.0, 2.5, 9.0]  # 25 % up to 9
    assert per["buffer"].tolist() == [0.0, 1.5, 5.0]  # 20 % of what hedging left, up to 5
    assert total.tolist() == [0.0, 4.0, 14.0]


def test_gpd_uses_the_exponential_limit_near_zero_shape():
    fit = {"u": 1.0, "xi": 0.0, "beta": 2.0, "n_excess": 10, "n": 100}
    var, es = risk.gpd_var_es(fit, 0.99)
    assert var == pytest.approx(1.0 - 2.0 * math.log(0.01 / 0.1))
    assert es == pytest.approx(var + 2.0)


def test_calibration_pins_the_gross_quantile_only_when_asked(tables):
    free = risk.run_all(tables, _opt())["monte_carlo"]
    pinned = risk.run_all(tables, _opt(calibrate_gross_q99=35.4))["monte_carlo"]
    assert pinned["a0.99"]["gross"] == pytest.approx(35.4, rel=1e-9)
    assert free["a0.99"]["gross"] != pytest.approx(35.4, rel=1e-3)


@pytest.mark.parametrize("hp, message", [
    ({"iteration": [1000]}, "unknown hyperparameters"),
    ({"alphas": [0.3]}, "alphas"),
    ({"iterations": [10]}, "iterations"),
    ({"defence": [{"name": "x", "share": 1.5, "cap": 1}]}, "share"),
])
def test_bad_hyperparameters_are_refused(hp, message):
    with pytest.raises(ValueError, match=message):
        risk.risk_options(hp, seed=0)


def test_a_dataset_without_its_tables_names_what_is_missing(tmp_path, dataset):
    (tmp_path / "losses.csv").write_bytes((dataset / "losses.csv").read_bytes())
    with pytest.raises(ValueError, match=r"missing tables \['factors', 'exposure'\]"):
        risk.find_tables(tmp_path)


# ------------------------------------------------------------------ spec


def test_the_spec_takes_no_features_and_no_split(dataset):
    spec = load_spec(SPEC.format(uri=dataset))
    assert spec.features is None and spec.data.split is None
    assert backend_name(spec.task, spec.model) == "risk_sim"
    with pytest.raises(ValueError, match="features does not apply to task risk_simulation"):
        load_spec(SPEC.format(uri=dataset) + "features: {inputs: [{name: a, type: number}], output: {name: y, type: number}}\n")
    with pytest.raises(ValueError, match="data.split does not apply to task risk_simulation"):
        load_spec(SPEC.format(uri=dataset).replace("format: csv}", "format: csv, split: {validation_fraction: 0.2}}"))


def test_materialise_pins_exactly_the_three_tables(tmp_path, dataset):
    data = materialise(load_spec(SPEC.format(uri=dataset)), tmp_path / "data")
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    paths = sorted(e["path"] for e in manifest["entries"]) if isinstance(manifest, dict) else sorted(e["path"] for e in manifest)
    assert paths == ["risk/exposure.csv", "risk/factors.csv", "risk/losses.csv"]
    assert set(data.train) == {"losses", "factors", "exposure"} and data.validation is None


# ------------------------------------------------------------------ the whole run


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, dataset) -> Path:
    tmp = tmp_path_factory.mktemp("run")
    (tmp / "spec.yaml").write_text(SPEC.format(uri=dataset))
    mp = pytest.MonkeyPatch()
    mp.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    try:
        assert cli.main(["run", str(tmp / "spec.yaml"), "--out", str(tmp / "out")]) == 0
    finally:
        mp.undo()
    return tmp / "out"


def test_the_report_has_every_method_and_a_matching_replay(run_dir):
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "done", report["error"]
    assert report["backend"] == "risk_sim" and report["task"] == "risk_simulation"
    assert {"historical_net_var_99", "delta_normal_net_var_99", "evt_net_var_99",
            "monte_carlo_net_var_99", "monte_carlo_net_se_99"} <= set(report["metrics"])
    assert set(report["evaluation"]["splits"]) == {"historical", "delta_normal", "evt", "monte_carlo"}
    (ev,) = [r for r in report["runtime_eval"] if r["format"] == "json"]
    assert ev["metric"] == "replay_match" and ev["value"] == 1.0 and ev["matches_training"] is True
    assert ev["n"] == 10_000
    results = json.loads((run_dir / "artifacts" / "results.json").read_text())
    assert results["results_sha256"] == report["provenance"]["results_sha256"]
    assert results["manifest"]["data_sha256"] == report["data_sha256"]
    assert [r["iterations"] for r in results["results"]["monte_carlo"]["runs"]] == [2000, 8000]
    # Nothing is trained: no training telemetry (the runner would log each point as an epoch), and the
    # Monte Carlo ladder is in the evaluation block instead.
    metrics_file = run_dir / "metrics.jsonl"
    assert not metrics_file.exists() or metrics_file.read_text() == ""
    ladder = report["evaluation"]["convergence"]
    assert [c["iterations"] for c in ladder] == [2000, 8000] and all(c["alpha"] == 0.99 for c in ladder)
    assert ladder[-1]["net_var"] == report["metrics"]["monte_carlo_net_var_99"]


def test_a_tampered_result_fails_the_replay(run_dir, tmp_path):
    results = json.loads((run_dir / "artifacts" / "results.json").read_text())
    results["results"]["monte_carlo"]["a0.99"]["net"] *= 0.5  # someone shaved the headline number
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    data = materialise(load_spec((run_dir / "resolved.yaml").read_text()), tmp_path / "data")
    ev = load_backend("risk_sim").runtime_eval(Artifact(path=path, format="json", sha256="", bytes=0), data)
    assert ev.value == 0.0 and ev.matches_training is False and "altered" in (ev.reason or "")
    # Re-digesting the edited numbers does not help: the replay recomputes them from the data.
    results["results_sha256"] = risk.results_digest(results["results"])
    path.write_text(json.dumps(results))
    ev = load_backend("risk_sim").runtime_eval(Artifact(path=path, format="json", sha256="", bytes=0), data)
    assert ev.value == 0.0 and "digest differs" in (ev.reason or "")


def test_a_replay_on_other_data_says_so(run_dir, tmp_path, dataset):
    other = tmp_path / "other"
    risk.make_synthetic_dataset(other, seed=8, fmt="csv")
    data = materialise(load_spec(SPEC.format(uri=other)), tmp_path / "data")
    artifact = Artifact(path=run_dir / "artifacts" / "results.json", format="json", sha256="", bytes=0)
    ev = load_backend("risk_sim").runtime_eval(artifact, data)
    assert ev.value == 0.0 and "data differs" in (ev.reason or "")


def test_the_report_carries_a_try_example_that_predict_accepts(run_dir, tmp_path, capsys):
    """The hub pre-fills Try it with `evaluation.try_example`: it must name a real factor and be answerable."""
    report = json.loads((run_dir / "report.json").read_text())
    example = report["evaluation"]["try_example"]
    assert example["alpha"] == 0.99 and example["iterations"] == 20_000
    ((factor, amount),) = example["spend"].items()
    assert factor == "fx" and amount == pytest.approx(86.46 * 1.2, abs=0.01)  # the synthetic data's largest spend
    path = tmp_path / "example.json"
    path.write_text(json.dumps(example))
    assert cli.main(["predict", str(run_dir), "--input", str(path)]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["iterations"] == 20_000 and answer["net_var"] >= 0


def test_results_json_is_self_contained(run_dir, tmp_path):
    """The delivered file carries its inputs and no runner path: copied anywhere, it predicts and replays."""
    text = (run_dir / "artifacts" / "results.json").read_text()
    delivered = json.loads(text)
    assert set(delivered["inputs"]) == {"losses", "factors", "exposure"}
    assert all("path" not in t for t in delivered["manifest"]["tables"].values())
    assert str(run_dir) not in text and str(run_dir.parent) not in text
    alone = tmp_path / "elsewhere" / "results.json"
    alone.parent.mkdir()
    alone.write_text(text)
    backend = load_backend("risk_sim")
    answer = backend.predict(Artifact(path=alone, format="json", sha256="", bytes=0), {"iterations": 5000})
    assert answer["iterations"] == 5000 and answer["gross_var"] > 0
    data = materialise(load_spec((run_dir / "resolved.yaml").read_text()), tmp_path / "data")
    assert backend.runtime_eval(Artifact(path=alone, format="json", sha256="", bytes=0), data).value == 1.0
    # Editing an embedded input is caught.
    delivered["inputs"]["exposure"]["data"][0][1] *= 2
    alone.write_text(json.dumps(delivered))
    ev = backend.runtime_eval(Artifact(path=alone, format="json", sha256="", bytes=0), data)
    assert ev.value == 0.0 and "inputs do not hash" in (ev.reason or "")


def test_predict_answers_a_what_if_on_the_exposure(run_dir, tmp_path, capsys):
    base, more = tmp_path / "base.json", tmp_path / "more.json"
    base.write_text("{}")
    more.write_text(json.dumps({"spend": {"cocoa": 60.0}, "iterations": 5000}))
    assert cli.main(["predict", str(run_dir), "--input", str(base)]) == 0
    a = json.loads(capsys.readouterr().out)
    assert cli.main(["predict", str(run_dir), "--input", str(more)]) == 0
    b = json.loads(capsys.readouterr().out)
    assert a["alpha"] == 0.99 and b["iterations"] == 5000
    assert b["gross_var"] > a["gross_var"]  # twice the cocoa spend, more at risk


def test_the_notebook_replays_the_run(tmp_path, dataset, monkeypatch):
    """The bundle's notebook shows the estimators by source and its check re-runs them (no repair loop)."""
    pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("SAKURA_ATELIER_NOTEBOOK", raising=False)
    monkeypatch.delenv("SAKURA_ATELIER_LLM_URL", raising=False)
    (tmp_path / "spec.yaml").write_text(SPEC.format(uri=dataset))
    assert cli.main(["run", str(tmp_path / "spec.yaml"), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["notebook_check"]["ok"] is True, report["notebook_check"]
    assert report["notebook_check"]["repairs"] == 0
    assert "notebook" in {a["format"] for a in report["artifacts"]}
