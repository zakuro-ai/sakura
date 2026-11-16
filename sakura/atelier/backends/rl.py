"""rl: a path-walking policy on a knowledge graph (task ``reinforcement_learning``, model ``kg-path-policy``).

The environment is ``sakura.atelier.kg``: an episode starts at a query's head entity with the query's
relation, the agent follows one edge per hop or stops, and it is rewarded 1 when it ends on the query's
tail. The policy is small: entity and relation embeddings, a GRU cell over the path walked so far, and an
MLP whose output scores every outgoing edge of the current entity (a masked softmax over the valid ones).
It is trained with REINFORCE, a moving-average baseline and an entropy bonus, on a CPU, from random weights.

``train`` logs the mean reward of each logging interval (the return curve) and the validation ``hits_at_1``.
The final numbers are ``hits_at_1`` by greedy rollout, ``hits_at_<k>`` and ``mrr`` by beam search, and the
success rate of a uniform random walk on the same validation queries, so a reader can see whether the
policy learned anything. ``export`` writes the one-hop policy as ``torchscript`` or ``onnx``, with
``graph.json`` beside it (entities, relations and the graph's triples: the policy cannot walk without
them). ``runtime_eval`` reloads the exported file and re-scores ``hits_at_1`` in its own runtime.
``predict`` takes ``{"head": ..., "relation": ...}`` and returns the tail it reaches and the path it walked.

torch is imported inside the functions (``docs/atelier/envs/rl.txt``); the module itself imports without it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import numpy.typing as npt

from sakura.atelier import kg
from sakura.atelier.types import (Artifact, Checkpoint, MaterialisedData, MetricsSink, ResolvedSpec, RunDir,
                                  RuntimeEval, TrainResult)
from sakura.atelier.wrap_up import WrapUpWatch, max_steps_of

GRAPH = "graph.json"
FILES = {"torchscript": "policy.pt", "onnx": "policy.onnx"}
RUNTIMES = {"torchscript": "torchscript", "onnx": "onnxruntime"}
#: The inputs of one policy step, in order (the ONNX graph's input names).
STEP_INPUTS = ("current", "query", "last", "hidden", "cand_rel", "cand_ent", "mask")
#: (current, query, last, hidden, cand_rel, cand_ent, mask) -> (logits, hidden), torch tensors.
Step = Callable[..., Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _one_thread() -> Iterator[None]:
    """Run torch on one thread. The tensors here are a few kilobytes: splitting an operation across cores
    costs more than it saves, and on a busy machine the idle worker threads make a run many times slower."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


_POLICY: list[Any] = []


def policy_class() -> Any:
    """The policy network's class, built on first use so that importing this module needs no torch."""
    if _POLICY:
        return _POLICY[0]
    import torch
    from torch import nn

    class PathPolicy(nn.Module):
        """One hop: where the agent is, what it was asked and how it got here -> a score per outgoing edge."""

        def __init__(self, n_entities: int, n_relations: int, dim: int, hidden: int) -> None:
            super().__init__()
            self.entity = nn.Embedding(n_entities, dim)
            self.relation = nn.Embedding(n_relations, dim)  # as written, inverse, and the stop / start token
            self.history = nn.GRUCell(2 * dim, hidden)
            self.state = nn.Sequential(nn.Linear(hidden + 2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * dim))

        def forward(self, current: Any, query: Any, last: Any, hidden: Any, cand_rel: Any, cand_ent: Any,
                    mask: Any) -> tuple[Any, Any]:
            here = self.entity(current)
            hidden = self.history(torch.cat([self.relation(last), here], dim=-1), hidden)
            state = self.state(torch.cat([hidden, here, self.relation(query)], dim=-1))
            candidates = torch.cat([self.relation(cand_rel), self.entity(cand_ent)], dim=-1)
            logits = (candidates * state.unsqueeze(1)).sum(dim=-1)
            return logits.masked_fill(~mask, -1e9), hidden  # masked softmax: an invalid edge gets no mass

    _POLICY.append(PathPolicy)
    return PathPolicy


def _settings(hp: dict[str, Any]) -> dict[str, Any]:
    """The preset's values (``reinforcement_learning/reinforce@1``) with their types; defaults are the preset's."""
    return {
        "max_hops": int(hp.get("max_hops", 3)), "embedding_dim": int(hp.get("embedding_dim", 32)),
        "hidden_dim": int(hp.get("hidden_dim", 64)), "steps": int(hp.get("steps", 1500)),
        "batch_size": int(hp.get("batch_size", 128)), "learning_rate": float(hp.get("learning_rate", 0.003)),
        "entropy_weight": float(hp.get("entropy_weight", 0.02)), "baseline_decay": float(hp.get("baseline_decay", 0.9)),
        "grad_clip": float(hp.get("grad_clip", 5.0)), "beam_width": int(hp.get("beam_width", 8)),
        "hits_k": int(hp.get("hits_k", 3)), "log_every": int(hp.get("log_every", 25)),
        "random_rollouts": int(hp.get("random_rollouts", 20)), "max_out_degree": int(hp.get("max_out_degree", 200)),
    }


def _step_inputs(env: kg.PathEnv, last: Any, hidden: Any) -> tuple[tuple[Any, ...], Any]:
    """The seven inputs of one policy step for the environment's current state, and the valid-action mask."""
    import torch

    rel, ent, ok = env.actions()
    mask = torch.from_numpy(ok)
    return (torch.from_numpy(env.current), torch.from_numpy(env.queries[:, 1].copy()), last, hidden,
            torch.from_numpy(rel), torch.from_numpy(ent), mask), mask


def beam_search(step: Step, graph: kg.Graph, queries: npt.NDArray[Any], max_hops: int, width: int,
                hidden_dim: int) -> dict[str, npt.NDArray[Any]]:
    """Walk every query with the ``width`` most probable paths (``width`` 1 is the greedy rollout).

    Returns, per query and beam, best first: ``entity`` (where the path ends), ``score`` (its log
    probability, ``-inf`` for a beam that holds no path), and ``relations`` / ``entities`` (the edge
    followed and the entity reached at each hop; after a stop the stop relation repeats)."""
    import torch

    n = len(queries)
    env = kg.PathEnv(graph, np.repeat(queries, width, axis=0), max_hops)
    score = torch.full((n, width), float("-inf"))
    score[:, 0] = 0.0  # one path to start with: the others are empty until it branches
    hidden = torch.zeros(n * width, hidden_dim)
    last = torch.full((n * width,), graph.stop_relation, dtype=torch.long)
    relations = np.full((n * width, max_hops), graph.stop_relation, dtype=np.int64)
    entities = np.repeat(queries[:, 0], width)[:, None].repeat(max_hops, axis=1)
    offset = torch.arange(n)[:, None] * width
    with torch.no_grad():
        for hop in range(max_hops):
            inputs, mask = _step_inputs(env, last, hidden)
            logits, hidden = step(*inputs)
            logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, float("-inf"))
            slots = logp.shape[1]
            score, flat = (score.reshape(-1, 1) + logp).reshape(n, width * slots).topk(width, dim=1)
            source = (offset + flat // slots).reshape(-1)
            action = (flat % slots).reshape(-1).numpy().copy()
            action[~np.isfinite(score.reshape(-1).numpy())] = kg.STOP  # an empty beam just waits
            env.reindex(source.numpy())
            hidden, relations, entities = hidden[source], relations[source.numpy()], entities[source.numpy()]
            taken = env.step(action)
            relations[:, hop], entities[:, hop:] = taken, env.current[:, None]
            last = torch.from_numpy(taken)
    return {"entity": env.current.reshape(n, width), "score": score.numpy(),
            "relations": relations.reshape(n, width, max_hops), "entities": entities.reshape(n, width, max_hops)}


def hits_at_1(step: Step, graph: kg.Graph, queries: npt.NDArray[Any], max_hops: int, hidden_dim: int) -> float:
    """The share of queries whose greedy rollout (the most probable action at every hop) ends on the tail."""
    if len(queries) == 0:
        return 0.0
    found = beam_search(step, graph, queries, max_hops, 1, hidden_dim)["entity"][:, 0]
    return float((found == queries[:, 2]).mean())


def ranking(step: Step, graph: kg.Graph, queries: npt.NDArray[Any], max_hops: int, width: int, hidden_dim: int,
            k: int) -> dict[str, float]:
    """``hits_at_<k>`` and ``mrr`` from a beam search: the entities the beams end on, each ranked by its
    most probable path. A tail no beam reaches has no rank (reciprocal rank 0)."""
    if len(queries) == 0:
        return {f"hits_at_{k}": 0.0, "mrr": 0.0}
    beams = beam_search(step, graph, queries, max_hops, width, hidden_dim)
    hits, reciprocal = 0, 0.0
    for ends, scores, tail in zip(beams["entity"], beams["score"], queries[:, 2]):
        ranked = list(dict.fromkeys(int(e) for e, s in zip(ends, scores) if np.isfinite(s)))
        if int(tail) in ranked:
            rank = ranked.index(int(tail)) + 1
            hits += rank <= k
            reciprocal += 1.0 / rank
    return {f"hits_at_{k}": hits / len(queries), "mrr": reciprocal / len(queries)}


def _graph_of(data: MaterialisedData, max_out_degree: int) -> tuple[kg.Graph, tuple[str, str, str]]:
    info = data.extra["kg"]
    columns = (str(info["columns"][0]), str(info["columns"][1]), str(info["columns"][2]))
    names = kg.Graph.build(info["entities"], info["relations"], np.zeros((0, 3), dtype=np.int64))
    triples, _ = names.encode(info["graph"], columns)
    return kg.Graph.build(info["entities"], info["relations"], triples, max_out_degree), columns


def _graph_doc(model: dict[str, Any]) -> dict[str, Any]:
    graph: kg.Graph = model["graph"]
    return {"schema": 1, "entities": graph.entities, "relations": graph.relations,
            "triples": graph.triples.tolist(), "columns": list(model["columns"]), "settings": model["settings"]}


def _write_graph(model: dict[str, Any], out: RunDir) -> Path:
    out.artifacts.mkdir(parents=True, exist_ok=True)
    path = out.artifacts / GRAPH
    path.write_text(json.dumps(_graph_doc(model), sort_keys=True) + "\n", encoding="utf-8")
    return path


@lru_cache(maxsize=4)
def _load(path_str: str, fmt: str) -> tuple[Step, kg.Graph, dict[str, Any]]:
    """The delivered policy as a step function in its own runtime, with the graph and settings from the
    ``graph.json`` beside it. Memoised: the warm ``serve`` worker loads once."""
    import torch

    path = Path(path_str)
    doc = json.loads((path.parent / GRAPH).read_text(encoding="utf-8"))
    graph = kg.Graph.build(doc["entities"], doc["relations"], doc["triples"], int(doc["settings"]["max_out_degree"]))
    if fmt == "torchscript":
        module = torch.jit.load(str(path), map_location="cpu")  # type: ignore[no-untyped-call]
        module.eval()
        return module, graph, doc
    from sakura.atelier.ort import inference_session

    session = inference_session(path)  # the engine's one way to open a session (explicit thread counts)

    def step(*inputs: Any) -> tuple[Any, Any]:
        logits, hidden = session.run(None, {name: t.numpy() for name, t in zip(STEP_INPUTS, inputs)})
        return torch.from_numpy(logits), torch.from_numpy(hidden)

    return step, graph, doc


class RlBackend:
    name = "rl"
    tasks = frozenset({"reinforcement_learning"})

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        if resume is not None:
            raise ValueError("the rl backend does not resume from a checkpoint: a run trains from random weights")
        with _one_thread():
            return self.reinforce(spec, data, out, metrics)

    def reinforce(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink) -> TrainResult:
        """The training loop and the final scoring: what ``train`` runs."""
        import torch

        t0 = time.time()
        s = _settings(spec.hyperparameters)
        seed = int(spec.spec.seed)
        graph, columns = _graph_of(data, s["max_out_degree"])
        train_q, _ = graph.encode(data.train, columns)
        eval_q, _ = graph.encode(data.validation, columns)
        hops, hidden_dim = s["max_hops"], s["hidden_dim"]

        torch.manual_seed(seed)
        sampler = torch.Generator().manual_seed(seed)
        policy = policy_class()(len(graph.entities), graph.stop_relation + 1, s["embedding_dim"], hidden_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=s["learning_rate"])
        batch = min(s["batch_size"], len(train_q))
        per_epoch = max(1, -(-len(train_q) // batch))
        # limits.max_steps and the wrap-up signal (limits.time_budget_s raises it) are checked before
        # every policy-gradient update: the loop stops there and the policy so far is scored and exported.
        watch = WrapUpWatch(out.root, max_steps_of(spec))
        baseline, order, cursor = 0.0, torch.empty(0, dtype=torch.long), 0
        rewards: list[float] = []
        losses: list[float] = []
        done = 0

        def log(step: int) -> None:
            epoch = (step + 1) / per_epoch
            metrics.log("mean_reward", sum(rewards) / len(rewards), step=step, epoch=epoch)
            metrics.log("loss", sum(losses) / len(losses), step=step, epoch=epoch)
            policy.eval()
            metrics.log("hits_at_1", hits_at_1(policy, graph, eval_q, hops, hidden_dim), step=step,
                        split="validation", epoch=epoch)
            policy.train()
            rewards.clear()
            losses.clear()

        for step in range(s["steps"]):
            if watch.stop(step / per_epoch, step):
                break
            if cursor + batch > len(order):  # a new pass over the training queries, in a new order
                order, cursor = torch.randperm(len(train_q), generator=sampler), 0
            env = kg.PathEnv(graph, train_q[order[cursor:cursor + batch].numpy()], hops)
            cursor += batch
            hidden = torch.zeros(batch, hidden_dim)
            last = torch.full((batch,), graph.stop_relation, dtype=torch.long)
            log_prob, entropy = torch.zeros(batch), torch.zeros(batch)
            for _ in range(hops):
                inputs, _mask = _step_inputs(env, last, hidden)
                logits, hidden = policy(*inputs)
                logp = torch.log_softmax(logits, dim=-1)
                prob = logp.exp()
                action = torch.multinomial(prob, 1, generator=sampler).squeeze(1)
                log_prob = log_prob + logp.gather(1, action[:, None]).squeeze(1)
                entropy = entropy - (prob * logp).sum(dim=1)
                last = torch.from_numpy(env.step(action.numpy()))
            reward = torch.from_numpy(env.reward())
            # REINFORCE: raise the log probability of the paths that did better than the running average
            loss = -((reward - baseline) * log_prob).mean() - s["entropy_weight"] * entropy.mean()
            baseline = s["baseline_decay"] * baseline + (1.0 - s["baseline_decay"]) * float(reward.mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), s["grad_clip"])
            optimizer.step()
            done = step + 1
            rewards.append(float(reward.mean()))
            losses.append(float(loss.detach()))
            if done % s["log_every"] == 0:
                log(step)
        if rewards:  # the last, shorter interval
            log(done - 1)

        policy.eval()
        k = s["hits_k"]
        final = {"hits_at_1": hits_at_1(policy, graph, eval_q, hops, hidden_dim),
                 **ranking(policy, graph, eval_q, hops, s["beam_width"], hidden_dim, k)}
        random_hits = kg.random_walk(graph, eval_q, hops, s["random_rollouts"], seed)
        train_hits = hits_at_1(policy, graph, train_q[:max(len(eval_q), 256)], hops, hidden_dim)
        warnings = []
        if final["hits_at_1"] <= random_hits + 0.05:
            warnings.append(f"the policy's hits_at_1 ({final['hits_at_1']:.3f}) is not clearly above a random "
                            f"walk's ({random_hits:.3f}) on the same queries: it has learned little or nothing")
        if graph.truncated:
            warnings.append(f"{graph.truncated} entities have more than max_out_degree ({s['max_out_degree']}) "
                            "edges; the surplus edges are not walkable")
        head, relation, _ = (eval_q[0] if len(eval_q) else train_q[0])
        evaluation = {
            "splits": {"train": {"n": int(len(train_q)), "hits_at_1": train_hits},
                       "validation": {"n": int(len(eval_q)), **final}},
            "random_walk": {"hits_at_1": random_hits, "rollouts_per_query": s["random_rollouts"],
                            "n": int(len(eval_q))},
            "graph": {"entities": len(graph.entities), "relations": graph.n_relations,
                      "triples": int(len(graph.triples)), "max_hops": hops},
            "warnings": warnings,
            "try_example": {"head": graph.entities[int(head)], "relation": graph.relations[int(relation)]},
        }
        model = {"policy": policy, "graph": graph, "columns": columns, "settings": s}
        return TrainResult(
            model=model, metrics={**final, "random_walk_hits_at_1": random_hits}, train_seconds=time.time() - t0,
            provenance={"torch": torch.__version__, "parameters": sum(p.numel() for p in policy.parameters()),
                        "updates": done},
            evaluation=evaluation, stopped_early=watch.stopped_early)

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        import torch

        if fmt not in FILES:
            raise ValueError(f"reinforcement_learning exports {sorted(FILES)}, not {fmt!r}")
        model = result.model
        policy, graph = model["policy"], model["graph"]
        _write_graph(model, out)
        # One example step, two episodes wide. Batch size and the number of action slots are free
        # dimensions: the graph traced here has no shape baked in.
        env = kg.PathEnv(graph, np.zeros((2, 3), dtype=np.int64), 1)
        example, _ = _step_inputs(env, torch.full((2,), graph.stop_relation, dtype=torch.long),
                                  torch.zeros(2, model["settings"]["hidden_dim"]))
        path = out.artifacts / FILES[fmt]
        policy.eval()
        if fmt == "torchscript":
            torch.jit.trace(policy, example).save(str(path))  # type: ignore[no-untyped-call]
        else:
            free = {0: "batch", 1: "actions"}
            torch.onnx.export(policy, example, str(path), input_names=list(STEP_INPUTS),
                              output_names=["logits", "next_hidden"], opset_version=17, dynamo=False,
                              dynamic_axes={**{n: {0: "batch"} for n in ("current", "query", "last", "hidden",
                                                                          "next_hidden")},
                                            **{n: free for n in ("cand_rel", "cand_ent", "mask", "logits")}})
        return Artifact(path=path, format=fmt, sha256=_sha256(path), bytes=path.stat().st_size)

    def extra_artifacts(self, result: TrainResult, out: RunDir) -> list[Artifact]:
        """``graph.json``, delivered beside the policy: the vocabulary and the triples it walks on."""
        path = _write_graph(result.model, out)
        return [Artifact(path=path, format="kg_graph", sha256=_sha256(path), bytes=path.stat().st_size)]

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        """Reload the exported policy in its own runtime and walk the validation queries greedily on the
        graph delivered beside it. ``matches_training`` is left to the engine: this ``hits_at_1`` against
        the training one, within the tolerance of CONTRACTS §2.3."""
        if artifact.format not in FILES:  # graph.json, the licence: delivered beside the policy, not scored
            return RuntimeEval(format=artifact.format, runtime="none", metric="hits_at_1", value=None, n=0,
                               matches_training=None, reason=f"{artifact.format} is not a policy")
        step, graph, doc = _load(str(artifact.path), artifact.format)
        columns = (doc["columns"][0], doc["columns"][1], doc["columns"][2])
        queries, unknown = graph.encode(data.validation, columns)
        n = len(queries) + unknown
        settings = doc["settings"]
        with _one_thread():
            correct = hits_at_1(step, graph, queries, settings["max_hops"], settings["hidden_dim"]) * len(queries)
        return RuntimeEval(format=artifact.format, runtime=RUNTIMES[artifact.format], metric="hits_at_1",
                           value=correct / n if n else None, n=n, matches_training=None,
                           reason=None if n else "no validation queries",
                           provider="CPUExecutionProvider" if artifact.format == "onnx" else None)

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        """``{"head": name, "relation": name}`` -> the tail the greedy walk ends on and the path it took.
        ``"top_k": n`` adds the ``n`` best distinct tails of a beam search (``candidates``)."""
        if artifact.format not in FILES:
            raise ValueError(f"predict needs the policy ({sorted(FILES)}), not the {artifact.format!r} artifact")
        if isinstance(inputs, dict) and isinstance(inputs.get("inputs"), dict):
            inputs = inputs["inputs"]  # a hub body {"inputs": {...}}
        with _one_thread():
            return self._walk(artifact, inputs)

    def _walk(self, artifact: Artifact, inputs: Any) -> Any:
        step, graph, doc = _load(str(artifact.path), artifact.format)
        head, relation = str(inputs["head"]), str(inputs["relation"])
        if head not in graph.entities:
            raise ValueError(f"unknown entity {head!r}: the graph has {len(graph.entities)} entities")
        if relation not in graph.relations:
            raise ValueError(f"unknown relation {relation!r} (one of {graph.relations})")
        settings = doc["settings"]
        hops, hidden_dim = settings["max_hops"], settings["hidden_dim"]
        query = np.array([[graph.entities.index(head), graph.relations.index(relation), -1]], dtype=np.int64)
        walk = beam_search(step, graph, query, hops, 1, hidden_dim)
        path = []
        for rel, ent in zip(walk["relations"][0, 0], walk["entities"][0, 0]):
            if int(rel) == graph.stop_relation:
                break
            name, inverse = graph.relation_name(int(rel))
            path.append({"relation": name, "inverse": inverse, "entity": graph.entities[int(ent)]})
        answer = {"tail": graph.entities[int(walk["entity"][0, 0])], "path": path,
                  "stopped": len(path) < hops, "log_prob": float(walk["score"][0, 0])}
        top_k = int(inputs.get("top_k", 0))
        if top_k > 0:
            beams = beam_search(step, graph, query, hops, max(top_k, settings["beam_width"]), hidden_dim)
            best: dict[str, float] = {}
            for ent, score in zip(beams["entity"][0], beams["score"][0]):
                if np.isfinite(score):
                    best.setdefault(graph.entities[int(ent)], float(score))
            answer["candidates"] = [{"tail": t, "log_prob": s} for t, s in list(best.items())[:top_k]]
        return answer

    #: The notebook shows these by source, in pipeline order (notebook/build.py).
    NOTEBOOK_CELLS: tuple[Any, ...] = (kg.Graph.build, kg.PathEnv, kg.random_walk, policy_class, beam_search,
                                       hits_at_1, ranking)



BACKEND = RlBackend()
# train/export/runtime_eval anchor the notebook's sections; the environment and the policy come before `train`.
RlBackend.NOTEBOOK_CELLS = (*RlBackend.NOTEBOOK_CELLS, BACKEND.reinforce, BACKEND.train, BACKEND.export,
                            BACKEND.runtime_eval)
