"""Knowledge-graph path walking: the environment of task ``reinforcement_learning`` (backend ``rl``).

The dataset is one table of triples ``head, relation, tail``. Its training rows are both the graph the agent
walks on and the training queries; its validation rows are queries only, held out of the graph. An episode
starts at a query's head with the query's relation. At each hop the agent picks one outgoing edge of the
entity it stands on, or stops; an edge can be followed in both directions (every triple also gives the
inverse edge ``tail -> head``). The episode ends when the agent stops or after ``max_hops`` hops, and the
reward is 1 when it ends on the query's tail, else 0. For each query its own edge and that edge's inverse
are not walkable, so the answer cannot be read off the graph: it has to be reached by another path.

This is the multi-hop reasoning setup of MINERVA (Das et al., 2018), kept minimal. Nothing here imports
torch: the environment is numpy, the table loading pandas. The policy lives in ``backends/rl.py``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: The action slot that stops the episode. It is always valid.
STOP = 0
COLUMNS = ("head", "relation", "tail")


@dataclass(frozen=True)
class Graph:
    """The walkable graph as padded action tables. For entity ``e``, slot 0 is the stop action and slots
    1.. are its outgoing edges ``(relation, next entity)``, forward edges and inverse edges together, sorted.
    Relation ids: ``r`` in ``[0, R)`` is a relation as written, ``r + R`` its inverse, ``2R`` the stop action."""

    entities: list[str]
    relations: list[str]
    #: The distinct triples the tables were built from, as (head, relation, tail) ids, sorted.
    triples: npt.NDArray[Any]
    act_rel: npt.NDArray[Any]
    act_ent: npt.NDArray[Any]
    act_ok: npt.NDArray[Any]
    #: How many entities had more than ``max_out_degree`` edges and lost the surplus.
    truncated: int = 0

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    @property
    def stop_relation(self) -> int:
        return 2 * len(self.relations)

    @classmethod
    def build(cls, entities: list[str], relations: list[str], triples: Any, max_out_degree: int = 200) -> Graph:
        n_ent, n_rel = len(entities), len(relations)
        t = np.unique(np.asarray(triples, dtype=np.int64).reshape(-1, 3), axis=0)
        src = np.concatenate([t[:, 0], t[:, 2]])
        rel = np.concatenate([t[:, 1], t[:, 1] + n_rel])
        dst = np.concatenate([t[:, 2], t[:, 0]])
        order = np.lexsort((dst, rel, src))
        src, rel, dst = src[order], rel[order], dst[order]
        degree = np.bincount(src, minlength=n_ent)
        start = np.concatenate([[0], np.cumsum(degree)[:-1]])
        slot = np.arange(len(src)) - start[src]  # position of each edge among its entity's edges
        keep = slot < max_out_degree
        width = 1 + min(int(degree.max(initial=0)), max_out_degree)
        act_rel = np.full((n_ent, width), 2 * n_rel, dtype=np.int64)
        act_ent = np.tile(np.arange(n_ent, dtype=np.int64)[:, None], (1, width))  # padding stays in place
        act_ok = np.zeros((n_ent, width), dtype=bool)
        act_ok[:, STOP] = True
        act_rel[src[keep], 1 + slot[keep]] = rel[keep]
        act_ent[src[keep], 1 + slot[keep]] = dst[keep]
        act_ok[src[keep], 1 + slot[keep]] = True
        return cls(entities=list(entities), relations=list(relations), triples=t, act_rel=act_rel,
                   act_ent=act_ent, act_ok=act_ok, truncated=int((degree > max_out_degree).sum()))

    def encode(self, rows: Any, columns: tuple[str, str, str] = COLUMNS) -> tuple[npt.NDArray[Any], int]:
        """The rows of a triples table as (head, relation, tail) ids, and how many rows were left out
        because they name an entity or a relation this graph does not know."""
        ent = {name: i for i, name in enumerate(self.entities)}
        rel = {name: i for i, name in enumerate(self.relations)}
        heads, rels, tails = (list(map(str, rows[c])) for c in columns)
        ids = [(ent[h], rel[r], ent[t]) for h, r, t in zip(heads, rels, tails) if h in ent and r in rel and t in ent]
        return np.asarray(ids, dtype=np.int64).reshape(-1, 3), len(heads) - len(ids)

    def relation_name(self, relation_id: int) -> tuple[str, bool]:
        """``(name, inverse)`` of a relation id an edge was followed by."""
        return self.relations[relation_id % self.n_relations], relation_id >= self.n_relations


class PathEnv:
    """A batch of episodes, one per query row ``(head, relation, tail)``. A tail of -1 means the answer is
    not known (``predict``): nothing is masked and the reward is 0."""

    def __init__(self, graph: Graph, queries: npt.NDArray[Any], max_hops: int) -> None:
        self.graph = graph
        self.queries = np.asarray(queries, dtype=np.int64).reshape(-1, 3)
        self.max_hops = int(max_hops)
        self.current = self.queries[:, 0].copy()
        self.done = np.zeros(len(self.queries), dtype=bool)
        self.hops = 0

    @property
    def finished(self) -> bool:
        return self.hops >= self.max_hops or bool(self.done.all())

    def actions(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """``(relation, next entity, valid)`` per episode and action slot. The query's own edge and its
        inverse are never valid; an episode that has stopped can only stop again."""
        g, cur = self.graph, self.current
        head, relation, tail = self.queries[:, 0:1], self.queries[:, 1:2], self.queries[:, 2:3]
        rel, ent, ok = g.act_rel[cur], g.act_ent[cur], g.act_ok[cur].copy()
        here = cur[:, None]
        ok &= ~((here == head) & (rel == relation) & (ent == tail))
        ok &= ~((here == tail) & (rel == relation + g.n_relations) & (ent == head))
        ok[self.done, 1:] = False
        return rel, ent, ok

    def step(self, action: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Take one action slot per episode; returns the relation id each episode followed."""
        if self.hops >= self.max_hops:
            raise RuntimeError(f"the episode is over: max_hops ({self.max_hops}) reached")
        action = np.asarray(action, dtype=np.int64)
        rel, ent, ok = self.actions()
        rows = np.arange(len(action))
        if not ok[rows, action].all():
            raise ValueError("an action that is not valid was chosen")
        self.done = self.done | (action == STOP)
        self.current = ent[rows, action]
        self.hops += 1
        taken: npt.NDArray[Any] = rel[rows, action]
        return taken

    def reward(self) -> npt.NDArray[Any]:
        out: npt.NDArray[Any] = (self.current == self.queries[:, 2]).astype(np.float32)
        return out

    def reindex(self, source: npt.NDArray[Any]) -> None:
        """Episode ``i`` continues from the state of episode ``source[i]`` (beam search). The queries stay."""
        self.current, self.done = self.current[source], self.done[source]


def random_walk(graph: Graph, queries: npt.NDArray[Any], max_hops: int, rollouts: int, seed: int) -> float:
    """The success rate of an agent that picks uniformly among its valid actions (stop included), over
    ``rollouts`` episodes per query. This is the baseline a trained policy is read against."""
    if len(queries) == 0:
        return 0.0
    rng = np.random.Generator(np.random.PCG64(seed))
    env = PathEnv(graph, np.tile(queries, (max(1, rollouts), 1)), max_hops)
    while not env.finished:
        _, _, ok = env.actions()
        draw = rng.random(ok.shape)
        draw[~ok] = -1.0
        env.step(draw.argmax(axis=1))
    return float(env.reward().mean())


def materialise_triples(spec: Any, data_dir: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """The triples table of a ``reinforcement_learning`` spec: ``(train queries, validation queries, extra,
    notes)``. ``features`` names the three columns (two inputs: head then relation; the output: tail).

    The table is split by the ordinary rules (a split column, ``stratify_by``, a seeded random
    ``validation_fraction``, or row ranges). Every training row is an edge of the graph
    (``extra["kg"]["graph"]``); ``limits.max_train_rows`` / ``max_eval_rows`` cap the QUERIES of each side
    and leave the graph whole. Entities and relations are numbered in sorted order over the whole table."""
    import pandas as pd

    from sakura.atelier import data as D
    from sakura.atelier.spec import features_of, split_of, split_strategy
    from sakura.atelier.splitting import describe_split, split_by_column, split_stratified

    features = features_of(spec)
    columns = (features.inputs[0].name, features.inputs[1].name, features.output.name)
    split = split_of(spec.data)
    local = D._fetch_local(spec.data.uri, data_dir / "_download")
    if spec.data.format == "csv":
        df = pd.read_csv(local, dtype=str, keep_default_na=False)
    elif spec.data.format == "parquet":
        df = pd.read_parquet(local)
    else:
        raise ValueError(f"reinforcement_learning data must be a csv or parquet table, got {spec.data.format!r}")
    pinned = data_dir / local.name  # the manifest pins the exact table read
    if local.resolve() != pinned.resolve():
        shutil.copyfile(local, pinned)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"triples table columns {list(df.columns)} are missing {missing} named by spec.features")
    # a csv is read with keep_default_na=False, so its empty cells arrive as "" and not as NaN
    if df[list(columns)].isna().any().any() or (df[list(columns)].astype(str) == "").any().any():
        raise ValueError("triples table has empty cells in its head, relation or tail column")
    for c in columns:
        df[c] = df[c].astype(str)

    excluded, warnings = 0, None
    if split.column is not None:
        train, validation, excluded = split_by_column(df, split)
    elif split.stratify_by:
        assert split.validation_fraction is not None
        train, validation, warnings = split_stratified(df, split.validation_fraction, split.stratify_by, spec.seed)
    elif split.validation_fraction is not None:
        validation = df.sample(frac=split.validation_fraction, random_state=spec.seed)
        train = df.drop(validation.index)
    else:
        train, validation = D._row_slice(df, split.train_expr), D._row_slice(df, split.validation_expr)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError(f"the split leaves {len(train)} training and {len(validation)} validation triples; "
                         "both sides need at least one")
    train, validation = train[list(columns)], validation[list(columns)]
    notes: dict[str, Any] = {"data_split": describe_split(
        train, validation, split=split, output=columns[2], categorical=False, excluded=excluded,
        warnings=warnings, strategy=split_strategy(spec), stratify_by=split.stratify_by)}
    kept = pd.concat([train, validation])
    extra = {"kg": {"columns": list(columns), "graph": train,
                    "entities": sorted(set(kept[columns[0]]) | set(kept[columns[2]])),
                    "relations": sorted(set(kept[columns[1]]))}}
    train_queries, validation_queries = D._apply_row_limits(spec, train, validation, notes)
    return train_queries, validation_queries, extra, notes


__all__ = ["COLUMNS", "Graph", "PathEnv", "STOP", "materialise_triples", "random_walk"]
