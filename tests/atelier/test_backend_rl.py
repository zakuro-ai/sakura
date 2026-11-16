"""The rl backend (task ``reinforcement_learning``): REINFORCE on a knowledge graph, on a CPU.

Small real trainings (a few seconds each) on ``synthetic://triples?rule=compose``, whose graph holds a
two-hop rule a policy can learn and a random walk cannot guess. The environment itself is tested in
``test_kg_env.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from sakura.atelier import kg  # noqa: E402
from sakura.atelier.backends import load_backend  # noqa: E402
from sakura.atelier.backends import rl  # noqa: E402
from sakura.atelier.data import materialise  # noqa: E402
from sakura.atelier.registry import resolve  # noqa: E402
from sakura.atelier.spec import load_spec  # noqa: E402
from sakura.atelier.types import MetricsSink, RunDir, read_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

SPEC = """
atelier: 1
name: kg-walk
task: reinforcement_learning
data: {{uri: "synthetic://triples?rows={rows}&seed=3&entities={entities}&relations=5&rule=compose", format: csv,
       split: {{validation_fraction: 0.15}}}}
features:
  inputs: [{{name: head, type: category}}, {{name: relation, type: category}}]
  output: {{name: tail, type: category}}
model: kg-path-policy
hp: reinforcement_learning/reinforce@1
export: [torchscript, onnx]
seed: {seed}
{more}
"""


def _train(tmp_path, name="run", seed=7, rows=110, entities=30, more="overrides: {steps: 40, batch_size: 64}"):
    """One training in this process. Returns (result, data, run dir, resolved spec)."""
    resolved = resolve(load_spec(SPEC.format(seed=seed, rows=rows, entities=entities, more=more)))
    out = RunDir(tmp_path / name).ensure()
    data = materialise(resolved.spec, out.root / "data")
    with MetricsSink(out.metrics_path) as sink:
        result = load_backend("rl").train(resolved, data, out, sink, None)
    return result, data, out, resolved


def _weights(result):
    return torch.cat([p.detach().reshape(-1) for p in result.model["policy"].parameters()])


# --------------------------------------------------------------------------- determinism


def test_the_same_seed_trains_the_same_policy(tmp_path):
    a, *_ = _train(tmp_path, "a")
    b, *_ = _train(tmp_path, "b")
    assert torch.equal(_weights(a), _weights(b)) and a.metrics == b.metrics
    points_a, _ = read_metrics(tmp_path / "a" / "metrics.jsonl")
    points_b, _ = read_metrics(tmp_path / "b" / "metrics.jsonl")
    strip = lambda points: [{k: v for k, v in p.items() if k != "t"} for p in points]  # noqa: E731
    assert strip(points_a) == strip(points_b) and points_a
    c, *_ = _train(tmp_path, "c", seed=8)
    assert not torch.equal(_weights(a), _weights(c))


# --------------------------------------------------------------------------- limits and the wrap-up signal


def test_max_steps_stops_the_loop_at_that_many_updates(tmp_path):
    result, _, out, _ = _train(tmp_path, more="overrides: {steps: 40, batch_size: 64, log_every: 2}\n"
                                              "limits: {max_steps: 5}")
    assert result.stopped_early == {"reason": "max_steps", "epoch": pytest.approx(5 / 2), "step": 5}
    assert result.provenance["updates"] == 5
    points, _ = read_metrics(out.metrics_path)
    rewards = [p["step"] for p in points if p["name"] == "mean_reward"]
    assert rewards == [1, 3, 4]  # every second update, then the shorter last interval
    assert [p["step"] for p in points if p["split"] == "validation"] == [1, 3, 4]


def test_a_cap_above_the_planned_steps_changes_nothing(tmp_path):
    result, *_ = _train(tmp_path, more="overrides: {steps: 6, batch_size: 64}\nlimits: {max_steps: 50}")
    assert result.stopped_early is None and result.provenance["updates"] == 6


def test_the_wrap_up_signal_stops_training_and_still_returns_a_policy(tmp_path, monkeypatch):
    monkeypatch.delenv("ATELIER_WRAP_UP_FILE", raising=False)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "wrap-up-requested").write_text(json.dumps({"reason": "time_budget"}))
    result, _, out, _ = _train(tmp_path)
    assert result.stopped_early == {"reason": "wrap_up", "epoch": 0.0, "step": 0}
    assert result.provenance["updates"] == 0 and set(result.metrics) >= {"hits_at_1", "random_walk_hits_at_1"}
    assert load_backend("rl").export(result, "torchscript", out).path.is_file()


def test_row_limits_set_the_number_of_queries(tmp_path):
    result, data, _, _ = _train(tmp_path, more="overrides: {steps: 3, batch_size: 64}\n"
                                               "limits: {max_train_rows: 20, max_eval_rows: 5}")
    assert len(data.train) == 20 and len(data.validation) == 5
    assert result.evaluation["splits"]["train"]["n"] == 20 and result.evaluation["splits"]["validation"]["n"] == 5
    assert result.evaluation["random_walk"]["n"] == 5
    assert result.evaluation["graph"]["triples"] == len(data.extra["kg"]["graph"]) > 20  # the graph is whole


def test_the_backend_refuses_to_resume(tmp_path):
    from sakura.atelier.types import Checkpoint

    resolved = resolve(load_spec(SPEC.format(seed=1, rows=110, entities=30, more="")))
    with pytest.raises(ValueError, match="does not resume"):
        load_backend("rl").train(resolved, None, RunDir(tmp_path), None, Checkpoint(tmp_path, "kg-path-policy", "x"))


# --------------------------------------------------------------------------- the policy and the search


def test_an_invalid_edge_gets_no_probability(tmp_path):
    result, *_ = _train(tmp_path, more="overrides: {steps: 2, batch_size: 64}")
    graph, policy = result.model["graph"], result.model["policy"]
    env = kg.PathEnv(graph, graph.triples[:16], 3)  # sixteen queries whose own edge is in the graph
    inputs, mask = rl._step_inputs(env, torch.full((16,), graph.stop_relation), torch.zeros(16, 64))
    assert not mask.all() and mask[:, kg.STOP].all()
    with torch.no_grad():
        logits, hidden = policy(*inputs)
    probabilities = torch.softmax(logits, dim=-1)
    assert float(probabilities[~mask].max()) == 0.0 and torch.allclose(probabilities.sum(1), torch.ones(16))
    assert hidden.shape == (16, 64)


def test_beam_search_keeps_the_best_paths_first_and_width_one_is_greedy(tmp_path):
    result, _, _, _ = _train(tmp_path)
    graph, policy = result.model["graph"], result.model["policy"]
    queries = graph.triples[:12]
    wide = rl.beam_search(policy, graph, queries, 3, 6, 64)
    greedy = rl.beam_search(policy, graph, queries, 3, 1, 64)
    assert wide["entity"].shape == (12, 6) and wide["relations"].shape == (12, 6, 3)
    assert (np.diff(wide["score"], axis=1) <= 1e-6).all()  # best first
    assert np.isfinite(wide["score"][:, 0]).all() and (wide["score"] <= 1e-6).all()  # log probabilities
    # a path's probabilities over all the paths of a query cannot sum to more than one
    assert (np.exp(wide["score"]).sum(axis=1) <= 1.0 + 1e-5).all()
    assert wide["entities"][:, :, -1].tolist() == wide["entity"].tolist()  # the path ends where the beam says
    assert (greedy["score"][:, 0] <= wide["score"][:, 0] + 1e-5).all()  # the greedy path never beats the best beam
    ranks = rl.ranking(policy, graph, queries, 3, 6, 64, 3)
    assert set(ranks) == {"hits_at_3", "mrr"} and 0.0 <= ranks["mrr"] <= 1.0
    assert ranks["mrr"] <= ranks["hits_at_3"] + 0.25  # a tail ranked fourth or lower adds at most 1/4


# --------------------------------------------------------------------------- export, runtime_eval, predict


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One policy trained long enough to have learned the rule: 150 updates, two hops (the rule's length)."""
    return _train(tmp_path_factory.mktemp("rl"), rows=210, entities=60,
                  more="overrides: {steps: 150, max_hops: 2}\nlimits: {max_eval_rows: 30}")


def test_the_policy_beats_a_random_walk(trained):
    result = trained[0]
    assert result.metrics["hits_at_1"] >= result.metrics["random_walk_hits_at_1"] + 0.3, result.metrics
    assert result.metrics["hits_at_3"] >= result.metrics["hits_at_1"] - 1e-9
    assert 0.0 < result.metrics["mrr"] <= 1.0
    assert result.evaluation["warnings"] == []
    assert result.evaluation["random_walk"]["hits_at_1"] == result.metrics["random_walk_hits_at_1"]


@pytest.mark.parametrize("fmt, runtime", [("torchscript", "torchscript"), ("onnx", "onnxruntime")])
def test_an_exported_policy_rescores_to_the_training_number(trained, fmt, runtime):
    if fmt == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    result, data, out, _ = trained
    backend = load_backend("rl")
    artifact = backend.export(result, fmt, out)
    assert artifact.format == fmt and artifact.bytes == artifact.path.stat().st_size
    ev = backend.runtime_eval(artifact, data)
    assert (ev.format, ev.runtime, ev.metric, ev.n) == (fmt, runtime, "hits_at_1", 30)
    assert ev.value == pytest.approx(result.metrics["hits_at_1"], abs=1e-9)  # the same walks, not just close
    assert ev.matches_training is None  # the engine compares, with the tolerance of CONTRACTS section 2.3


def test_graph_json_carries_the_vocabulary_and_the_triples(trained):
    result, data, out, _ = trained
    backend = load_backend("rl")
    extras = backend.extra_artifacts(result, out)
    assert [(a.format, a.path.name) for a in extras] == [("kg_graph", "graph.json")]
    doc = json.loads(extras[0].path.read_text())
    assert doc["schema"] == 1 and doc["entities"] == sorted(f"e{i}" for i in range(60))
    assert doc["relations"] == [f"r{i}" for i in range(5)] and doc["columns"] == ["head", "relation", "tail"]
    assert len(doc["triples"]) == len(data.extra["kg"]["graph"]) and doc["settings"]["max_hops"] == 2
    ev = backend.runtime_eval(extras[0], data)  # delivered beside the policy, not scored
    assert ev.value is None and "not a policy" in ev.reason


def test_predict_returns_the_tail_and_the_path_walked(trained):
    result, data, out, _ = trained
    backend = load_backend("rl")
    artifact = backend.export(result, "torchscript", out)
    graph = result.model["graph"]
    edges = {(graph.entities[h], graph.relations[r], graph.entities[t]) for h, r, t in graph.triples}
    edges |= {(t, r, h) for h, r, t in edges}  # an edge can be walked backwards
    hits = 0
    for head, relation, tail in data.validation.to_numpy()[:10]:
        answer = backend.predict(artifact, {"head": head, "relation": relation})
        assert set(answer) == {"tail", "path", "stopped", "log_prob"} and len(answer["path"]) <= 2
        at = head
        for hop in answer["path"]:  # every hop is an edge of the delivered graph
            assert (at, hop["relation"], hop["entity"]) in edges and isinstance(hop["inverse"], bool)
            at = hop["entity"]
        assert at == answer["tail"] and answer["stopped"] == (len(answer["path"]) < 2)
        hits += answer["tail"] == tail
    assert hits >= 3  # held-out triples, reached without their own edge
    ranked = backend.predict(artifact, {"inputs": {"head": "e0", "relation": "r2", "top_k": 3}})
    assert 1 <= len(ranked["candidates"]) <= 3 and ranked["candidates"][0]["tail"] == ranked["tail"]
    with pytest.raises(ValueError, match="unknown entity 'nobody'"):
        backend.predict(artifact, {"head": "nobody", "relation": "r0"})
    with pytest.raises(ValueError, match="unknown relation"):
        backend.predict(artifact, {"head": "e0", "relation": "r9"})


# --------------------------------------------------------------------------- the whole run, in a fresh process

E2E = """
atelier: 1
name: kg-smoke
task: reinforcement_learning
data: {uri: "synthetic://triples?rows=280&seed=3&entities=80&relations=5&rule=compose", format: csv,
       split: {validation_fraction: 0.15}}
features:
  inputs: [{name: head, type: category}, {name: relation, type: category}]
  output: {name: tail, type: category}
model: kg-path-policy
hp: reinforcement_learning/reinforce@1
export: [torchscript, onnx]
overrides: {max_hops: 2}  # the rule is two hops long
seed: 1
limits: {max_steps: 150, max_train_rows: 200, max_eval_rows: 40}
"""


def _cli(tmp_path, *args):
    env = {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    env.pop("ATELIER_WRAP_UP_FILE", None)
    return subprocess.run([sys.executable, "-m", "sakura.atelier", *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=900)


def _run(tmp_path, *extra):
    (tmp_path / "spec.yaml").write_text(E2E)
    assert _cli(tmp_path, "validate", "spec.yaml").stdout.strip() == "OK"
    proc = _cli(tmp_path, "run", "spec.yaml", "--out", "out", *extra)
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert proc.returncode == 0, f"exit {proc.returncode}: {report.get('error')}\n{proc.stderr[-3000:]}"
    return report, tmp_path / "out"


def test_a_capped_run_on_synthetic_triples_learns_the_rule(tmp_path):
    pytest.importorskip("onnxruntime")
    report, out = _run(tmp_path, "--no-notebook")
    assert report["status"] == "done" and report["backend"] == "rl" and report["task"] == "reinforcement_learning"
    assert report["limits_applied"] == {
        "max_steps": 150, "max_train_rows": 200, "max_eval_rows": 40,
        "rows_before": {"train": 238, "validation": 42}, "rows_after": {"train": 200, "validation": 40}}
    assert report["stopped_early"]["reason"] == "max_steps" and report["stopped_early"]["step"] == 150
    assert report["provenance"]["updates"] == 150  # the preset plans 1500

    # the policy against a uniform random walk on the same 40 held-out queries
    m = report["metrics"]
    assert m["hits_at_1"] >= m["random_walk_hits_at_1"] + 0.3, m
    assert m["random_walk_hits_at_1"] < 0.1 and m["hits_at_3"] >= m["hits_at_1"] and m["mrr"] > 0.3
    ev = report["evaluation"]
    assert ev["splits"]["validation"]["n"] == 40 and ev["random_walk"]["hits_at_1"] == m["random_walk_hits_at_1"]
    assert ev["graph"] == {"entities": 80, "relations": 5, "triples": 238, "max_hops": 2} and ev["warnings"] == []

    # the return curve: one point per logging interval, and it rises
    points = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    rewards = [p["value"] for p in points if p["name"] == "mean_reward" and p["split"] == "train"]
    assert len(rewards) == 6 and rewards[-1] >= rewards[0] + 0.3
    assert [p["step"] for p in points if p["name"] == "hits_at_1"] == list(range(24, 150, 25))

    # both exports reload in their own runtime and walk to the same answers
    assert [(r["format"], r["runtime"], r["metric"], r["n"]) for r in report["runtime_eval"]] == [
        ("torchscript", "torchscript", "hits_at_1", 40), ("onnx", "onnxruntime", "hits_at_1", 40)]
    assert all(r["matches_training"] is True and r["value"] == m["hits_at_1"] for r in report["runtime_eval"])
    formats = {a["format"]: a for a in report["artifacts"]}
    assert set(formats) == {"torchscript", "onnx", "kg_graph", "license"}
    assert formats["torchscript"]["validated"] and formats["onnx"]["eval_metric"] == "hits_at_1"
    assert report["license"]["spdx"] == "Apache-2.0" and formats["kg_graph"]["path"] == "artifacts/graph.json"
    resolved = yaml.safe_load((out / "resolved.yaml").read_text())
    assert resolved["limits"] == {"max_steps": 150, "max_train_rows": 200, "max_eval_rows": 40}
    assert resolved["data"]["split"]["strategy"] == "random" and resolved["hyperparameters"]["steps"] == 1500

    # predict, as the runner calls it
    (tmp_path / "ask.json").write_text(json.dumps(ev["try_example"]))
    for fmt in ("torchscript", "onnx"):
        proc = _cli(tmp_path, "predict", "out", "--input", "ask.json", "--format", fmt)
        answer = json.loads(proc.stdout)
        assert proc.returncode == 0 and answer["tail"].startswith("e") and isinstance(answer["path"], list), answer


def test_the_notebook_of_a_reinforcement_learning_run_replays_it(tmp_path):
    pytest.importorskip("nbclient")
    pytest.importorskip("onnxruntime")
    report, out = _run(tmp_path)
    assert report["notebook_check"]["ok"] is True, report["notebook_check"]
    assert (out / "artifacts" / "notebook.tar").is_file()
