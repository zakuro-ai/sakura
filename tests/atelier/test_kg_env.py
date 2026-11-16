"""The knowledge-graph path-walking environment (``sakura.atelier.kg``), its triples table and the
``rule=compose`` synthetic graph. numpy and pandas only: the policy's tests are in ``test_backend_rl.py``."""

from __future__ import annotations

import csv
import re

import numpy as np
import pytest

from sakura.atelier import kg, synthetic
from sakura.atelier.spec import load_spec

ENTITIES = ["a", "b", "c", "d"]
RELATIONS = ["likes", "knows"]
# a -likes-> b -knows-> c, a -knows-> c, c -likes-> d
TRIPLES = [(0, 0, 1), (1, 1, 2), (0, 1, 2), (2, 0, 3)]


def _graph(**kw):
    return kg.Graph.build(ENTITIES, RELATIONS, TRIPLES, **kw)


def _edges(env, episode=0):
    """The valid (relation id, next entity) pairs of one episode, the stop action left out."""
    rel, ent, ok = env.actions()
    return {(int(r), int(e)) for r, e, v in zip(rel[episode][1:], ent[episode][1:], ok[episode][1:]) if v}


# --------------------------------------------------------------------------- the graph and its actions


def test_every_triple_is_walkable_both_ways_and_stop_is_slot_zero():
    g = _graph()
    env = kg.PathEnv(g, np.array([[2, 0, -1]]), max_hops=3)  # at c, nothing masked
    # c: forward likes -> d; inverse knows (id 1 + 2 relations) back to a and to b
    assert _edges(env) == {(0, 3), (3, 0), (3, 1)}
    rel, ent, ok = env.actions()
    assert ok[0, kg.STOP] and rel[0, kg.STOP] == g.stop_relation == 4 and ent[0, kg.STOP] == 2
    assert g.relation_name(3) == ("knows", True) and g.relation_name(0) == ("likes", False)


def test_padding_slots_are_never_valid():
    g = _graph()
    env = kg.PathEnv(g, np.array([[3, 0, -1]]), max_hops=1)  # d has one edge, the table is wider
    rel, ent, ok = env.actions()
    assert ok.shape[1] == 4 and ok[0].tolist() == [True, True, False, False]


def test_the_query_edge_and_its_inverse_are_not_walkable():
    g = _graph()
    asked = np.array([[0, 1, 2]])  # (a, knows, c): the answer edge is in the graph
    at_head = kg.PathEnv(g, asked, max_hops=3)
    assert (1, 2) not in _edges(at_head) and _edges(at_head) == {(0, 1)}  # only a -likes-> b is left
    at_tail = kg.PathEnv(g, np.array([[2, 1, 0]]), max_hops=3)  # standing on c, asked (c, knows, a)
    assert (3, 0) in _edges(at_tail)  # a different query: the inverse edge c -> a is just an edge
    # the same walk for the query (a, knows, c): once on c, the way back along the query edge is closed
    env = kg.PathEnv(g, asked, max_hops=3)
    env.step(np.array([_slot(env, 0, 1)]))  # a -likes-> b
    env.step(np.array([_slot(env, 1, 2)]))  # b -knows-> c
    assert (3, 0) not in _edges(env) and (3, 1) in _edges(env)


def _slot(env, relation, entity, episode=0):
    rel, ent, ok = env.actions()
    return next(i for i in range(rel.shape[1]) if ok[episode, i] and (rel[episode, i], ent[episode, i]) == (relation, entity))


def test_a_masked_edge_cannot_be_taken():
    g = _graph()
    free = kg.PathEnv(g, np.array([[0, 1, -1]]), max_hops=1)
    direct = _slot(free, 1, 2)
    with pytest.raises(ValueError, match="not valid"):
        kg.PathEnv(g, np.array([[0, 1, 2]]), max_hops=1).step(np.array([direct]))


# --------------------------------------------------------------------------- episodes


def test_reward_is_one_only_on_the_tail():
    g = _graph()
    env = kg.PathEnv(g, np.array([[0, 1, 2], [0, 1, 2]]), max_hops=2)
    env.step(np.array([_slot(env, 0, 1, 0), kg.STOP]))  # the first walks on, the second stops on a
    env.step(np.array([_slot(env, 1, 2, 0), kg.STOP]))
    assert env.reward().tolist() == [1.0, 0.0]
    assert kg.PathEnv(g, np.array([[0, 1, -1]]), max_hops=1).reward().tolist() == [0.0]  # no known answer


def test_stopping_ends_the_episode():
    g = _graph()
    env = kg.PathEnv(g, np.array([[0, 0, 3]]), max_hops=3)
    assert env.step(np.array([kg.STOP])).tolist() == [g.stop_relation]
    assert env.done.tolist() == [True] and env.finished
    assert _edges(env) == set()  # only stop is left
    env.step(np.array([kg.STOP]))
    assert env.current.tolist() == [0]


def test_the_hop_limit_ends_the_episode():
    g = _graph()
    env = kg.PathEnv(g, np.array([[0, 0, 3]]), max_hops=2)
    env.step(np.array([_slot(env, 0, 1)]))
    assert not env.finished
    env.step(np.array([_slot(env, 1, 2)]))
    assert env.finished and env.hops == 2 and env.reward().tolist() == [0.0]  # d was one hop too far
    with pytest.raises(RuntimeError, match="max_hops"):
        env.step(np.array([kg.STOP]))


def test_max_out_degree_drops_the_surplus_edges_and_counts_them():
    g = _graph(max_out_degree=1)
    assert g.act_ok.shape[1] == 2 and g.truncated == 3  # a, b and c have more than one edge
    assert _graph().truncated == 0


def test_encode_leaves_out_rows_the_graph_does_not_know():
    g = _graph()
    rows = {"head": ["a", "zz", "b"], "relation": ["likes", "likes", "hates"], "tail": ["b", "a", "c"]}
    ids, unknown = g.encode(rows)
    assert ids.tolist() == [[0, 0, 1]] and unknown == 2


# --------------------------------------------------------------------------- the random-walk baseline


def test_the_random_walk_is_seeded_and_cannot_use_the_query_edge():
    g = _graph()
    queries = np.array([[0, 1, 2], [2, 0, 3]])
    first = kg.random_walk(g, queries, max_hops=3, rollouts=200, seed=5)
    assert first == kg.random_walk(g, queries, max_hops=3, rollouts=200, seed=5)
    assert 0.0 < first < 1.0
    # (c, likes, d) is d's only edge: with it closed d cannot be reached at all
    assert kg.random_walk(g, np.array([[2, 0, 3]]), max_hops=3, rollouts=200, seed=5) == 0.0
    assert kg.random_walk(g, np.zeros((0, 3), dtype=np.int64), 3, 10, 0) == 0.0


# --------------------------------------------------------------------------- synthetic://triples?rule=compose

COMPOSE = "synthetic://triples?rows=150&seed=4&entities=40&relations=5&rule=compose"


def _triples(uri, tmp_path):
    with open(synthetic.generate(uri, tmp_path, "csv"), newline="") as fh:
        return [(r["head"], r["relation"], r["tail"]) for r in csv.DictReader(fh)]


def test_the_default_triples_are_unchanged_by_the_rule_parameter(tmp_path):
    plain = synthetic.generate("synthetic://triples?rows=40&seed=4&entities=12", tmp_path / "a", "csv").read_bytes()
    named = synthetic.generate("synthetic://triples?rows=40&seed=4&entities=12&rule=none", tmp_path / "b", "csv")
    assert plain == named.read_bytes()
    assert synthetic.parse_uri("synthetic://triples").params["rule"] == "none"


def test_compose_writes_a_two_hop_rule(tmp_path):
    triples = _triples(COMPOSE, tmp_path)
    assert len(triples) == len(set(triples)) == 150 and all(h != t for h, _, t in triples)
    step = {r: {h: t for h, rel, t in triples if rel == r} for r in ("r0", "r1", "r2")}
    names = {f"e{i}" for i in range(40)}
    for r in step:  # each of the three maps every entity to exactly one other entity
        assert set(step[r]) == names and set(step[r].values()) == names
    assert all(step["r2"][x] == step["r1"][step["r0"][x]] for x in names)  # r2 = r0 then r1
    others = [rel for _, rel, _ in triples if rel not in step]
    assert len(others) == 30 and set(others) <= {"r3", "r4"}  # the rest never touch the rule's relations


def test_compose_is_deterministic(tmp_path):
    assert _triples(COMPOSE, tmp_path / "a") == _triples(COMPOSE, tmp_path / "b")
    assert _triples(COMPOSE, tmp_path / "a") != _triples(COMPOSE.replace("seed=4", "seed=5"), tmp_path / "c")


@pytest.mark.parametrize("uri, match", [
    ("synthetic://triples?rule=chain", "rule must be one of"),
    ("synthetic://triples?rule=compose&relations=2&rows=60", "needs relations >= 3"),
    ("synthetic://triples?rule=compose&entities=20&rows=59", "rows must be between 3 \\* entities"),
])
def test_a_bad_compose_uri_says_what_is_wrong(uri, match):
    with pytest.raises(ValueError, match=match):
        synthetic.parse_uri(uri)


# --------------------------------------------------------------------------- the spec and the triples table

SPEC = """
atelier: 1
name: kg
task: reinforcement_learning
data: {{uri: "{uri}", format: {fmt}, split: {split}}}
features:
  inputs: [{{name: head, type: {head}}}, {{name: relation, type: category}}]
  output: {{name: tail, type: category}}
model: kg-path-policy
hp: reinforcement_learning/reinforce@1
export: [torchscript]
seed: 3
{more}
"""


def _spec(uri=COMPOSE, fmt="csv", split="{validation_fraction: 0.2}", head="category", more=""):
    return load_spec(SPEC.format(uri=uri, fmt=fmt, split=split, head=head, more=more))


def test_the_spec_names_the_three_columns_as_category_features():
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import split_strategy

    spec = _spec()
    resolved = resolve(spec)
    assert resolved.backend == "rl" and resolved.hyperparameters["max_hops"] == 3
    assert split_strategy(spec) == "random"  # the tail entity is not a class to stratify by
    with pytest.raises(ValueError, match="exactly two category inputs"):
        _spec(head="text")
    with pytest.raises(ValueError, match="csv or parquet triples table"):
        _spec(uri="file:///x.jsonl", fmt="jsonl")
    # the triples table is not curated: the block is refused, not silently ignored
    with pytest.raises(ValueError, match="curation is not supported for task 'reinforcement_learning'"):
        _spec(more="curation: {exact_duplicates: true}")
    with pytest.raises(ValueError, match="features is required"):
        load_spec(re.sub(r"features:\n.*\n.*\n", "", SPEC.format(
            uri=COMPOSE, fmt="csv", split="{validation_fraction: 0.2}", head="category", more="")))


def test_materialise_holds_the_validation_triples_out_of_the_graph(tmp_path):
    from sakura.atelier.data import materialise

    data = materialise(_spec(), tmp_path)
    graph = data.extra["kg"]["graph"]
    assert len(data.train) == 120 and len(data.validation) == 30 and len(graph) == 120
    edges = set(map(tuple, graph.to_numpy()))
    assert not edges & set(map(tuple, data.validation.to_numpy()))
    assert data.extra["kg"]["columns"] == ["head", "relation", "tail"]
    assert data.extra["kg"]["entities"] == sorted(f"e{i}" for i in range(40))
    assert data.extra["data_split"]["mode"] == "fraction" and data.extra["data_split"]["train_rows"] == 120
    again = materialise(_spec(), tmp_path / "again")
    assert again.manifest_sha256 == data.manifest_sha256 and again.validation.equals(data.validation)


def test_row_limits_cap_the_queries_and_leave_the_graph_whole(tmp_path):
    from sakura.atelier.data import materialise

    data = materialise(_spec(more="limits: {max_train_rows: 50, max_eval_rows: 10}"), tmp_path)
    assert len(data.train) == 50 and len(data.validation) == 10
    assert len(data.extra["kg"]["graph"]) == 120  # every training triple is still an edge
    assert data.extra["limits_applied"] == {"max_train_rows": 50, "max_eval_rows": 10,
                                            "rows_before": {"train": 120, "validation": 30},
                                            "rows_after": {"train": 50, "validation": 10}}


def test_a_real_table_with_its_own_column_names_and_a_split_column(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")  # the table is parquet; the core test environment has no engine for it
    from sakura.atelier.data import materialise

    table = tmp_path / "kg.parquet"
    pd.DataFrame({"subject": ["1", "2", "1", "3"], "predicate": ["p", "p", "q", "q"],
                  "object": ["2", "3", "3", "1"], "part": ["train", "train", "valid", "test"]}).to_parquet(table)
    spec = load_spec(f"""
atelier: 1
name: kg
task: reinforcement_learning
data: {{uri: "file://{table}", format: parquet, split: {{column: part, train: train, validation: valid, exclude: [test]}}}}
features:
  inputs: [{{name: subject, type: category}}, {{name: predicate, type: category}}]
  output: {{name: object, type: category}}
model: kg-path-policy
hp: reinforcement_learning/reinforce@1
export: [onnx]
""")
    data = materialise(spec, tmp_path / "data")
    assert list(data.train.columns) == ["subject", "predicate", "object"]
    assert data.validation.to_numpy().tolist() == [["1", "q", "3"]]
    assert data.extra["kg"]["entities"] == ["1", "2", "3"] and data.extra["kg"]["relations"] == ["p", "q"]
    assert data.extra["data_split"]["excluded_rows"] == 1


def test_an_empty_cell_in_a_csv_triples_table_is_refused(tmp_path):
    from sakura.atelier.data import materialise

    table = tmp_path / "kg.csv"
    table.write_text("head,relation,tail\na,r,b\nb,r,\nc,r,a\na,r,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty cells"):  # not an entity named ""
        materialise(_spec(uri=f"file://{table}", split="{validation_fraction: 0.25}"), tmp_path / "data")
