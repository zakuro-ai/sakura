"""``synthetic://`` data sources: a small generated dataset for a run that must prove a use case in minutes.

``data.uri: synthetic://<kind>?rows=N&seed=S[&...]`` names a generator and its parameters. The engine
writes the dataset to disk in one of the existing ``data.format``s and then reads it through the normal
materialise path (split, manifest, sha256), exactly like a ``file://`` source (SAK-0019, CONTRACTS §1.4).

Generation is deterministic for ``(kind, parameters, seed)``: the same uri gives the same bytes, so the
same manifest hash. To keep that true on every backend venv the files are written by hand (csv and jsonl
as text, PNG through ``zlib``, WAV through the standard ``wave`` module): no pandas, PIL or soundfile
version is involved. ``risk`` is the exception: it delegates to ``risk.make_synthetic_dataset``.

Only numpy is imported here, so ``validate`` can check a ``synthetic://`` uri without any training stack.
The numbers mean nothing about any real data: each kind carries a signal a model can learn, no more.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import urllib.parse
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

SCHEME = "synthetic://"
SAMPLE_RATE = 16_000
MAX_ROWS = 1_000_000

#: ``rows`` and ``seed`` apply to every kind. An int parameter is ``(default, min, max)``; a choice
#: parameter is ``(default, (choices...))``.
_COMMON: dict[str, tuple[Any, ...]] = {"rows": (200, 1, MAX_ROWS), "seed": (0, 0, 2**32 - 1)}

#: kind -> the ``data.format``s it can be written as (the first is the natural one) and its own parameters.
KINDS: dict[str, dict[str, Any]] = {
    # csv, columns f0..f{K-1} and `label` (classification, values c0..) or `target` (regression)
    "tabular": {"formats": ("csv",), "params": {"task": ("classification", ("classification", "regression")),
                                                 "features": (4, 1, 256), "classes": (2, 2, 100)}},
    # csv, columns `t` (0..N-1) and `value`: one series, sine + trend + noise
    "timeseries": {"formats": ("csv",), "params": {"period": (24, 2, 10_000)}},
    # <class>/<i>.png, classes c0..: one coloured shape per image, colour and shape set by the class
    "image": {"formats": ("image_folder",), "params": {"classes": (2, 2, 20), "size": (32, 8, 256)}},
    # <class>/<i>.wav, classes c0..: a 16 kHz tone whose frequency is set by the class
    "audio": {"formats": ("audio_folder",), "params": {"classes": (2, 2, 12), "ms": (500, 100, 5000)}},
    # csv, columns `text` and `label`: every class has its own words, mixed with shared filler words
    "text": {"formats": ("csv",), "params": {"classes": (2, 2, 50)}},
    # jsonl {prompt, completion}: a fixed table of `keys` words, each with one code word to answer
    "instruct": {"formats": ("jsonl",), "params": {"keys": (8, 2, 500)}},
    # jsonl {prompt, chosen, rejected}: the same table; `rejected` is another key's code
    "preference": {"formats": ("jsonl",), "params": {"keys": (8, 2, 500)}},
    # csv head,relation,tail: a chain e0 -> e1 -> ... (multi-hop paths) plus random edges. `rule=compose`
    # writes a graph with a two-hop rule instead (r2 = r0 then r1), for the rl backend to learn.
    "triples": {"formats": ("csv",), "params": {"entities": (20, 3, 10_000), "relations": (3, 1, 100),
                                                 "rule": ("none", ("none", "compose"))}},
    # manifest.csv (audio, transcript, duration_ms) + clips/*.wav: one tone per word of the transcript
    "asr": {"formats": ("asr_manifest",), "params": {"words": (4, 2, 16)}},
    # the three risk_simulation tables; `rows` is the number of monthly factor returns
    "risk": {"formats": ("parquet", "csv"), "params": {}},
    # a YOLO tree: data.yaml, images/{train,val}/<i>.png, labels/{train,val}/<i>.txt (`class cx cy w h`,
    # normalised): one to three solid rectangles per image, coloured by class. One image in five is val.
    "detection": {"formats": ("yolo",), "params": {"classes": (2, 2, 20), "size": (64, 32, 640)}},
    # table.csv (image, target) + images/<i>.png: a centred bright square; target = its side / the image side
    "image_regression": {"formats": ("csv",), "params": {"size": (32, 16, 256)}},
    # jsonl {state, questions, answers}, the decision task's rows: a ticket (`state=json` an object,
    # `state=text` one line) and three questions whose answers are a function of it
    "decision": {"formats": ("jsonl",), "params": {"options": (3, 2, 26), "state": ("json", ("json", "text"))}},
}
#: detection needs an image in each of train and val; decision.MIN_USABLE_ROWS is 50 (not imported: numpy only)
_MIN_ROWS = {"risk": 120, "timeseries": 8, "detection": 2, "decision": 50}
#: decision states are distinct (the task dedupes rows and splits by state); this stays far below the
#: 2 * 5 * 10000 * 2 states the smallest `options` allows, so drawing distinct ones is quick.
_MAX_ROWS = {"decision": 50_000}
#: The directory a folder kind is written to when it is not ``synthetic-<kind>``. `yolo` is where the yolo
#: materialiser keeps its copy of a dataset: written there, the tree is read in place, not copied and pinned twice.
_ROOTS = {"detection": "yolo"}
_ASR_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
              "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa")


@dataclass(frozen=True)
class SyntheticSource:
    kind: str
    #: every parameter of the kind, defaults filled in (``rows`` and ``seed`` included)
    params: dict[str, Any]

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(KINDS[self.kind]["formats"])


def is_synthetic(uri: str) -> bool:
    return uri.startswith(SCHEME)


def parse_uri(uri: str) -> SyntheticSource:
    """``synthetic://<kind>?name=value&...`` -> the kind and its full parameter set.

    Raises ``ValueError`` naming what is wrong: an unknown kind, an unknown or repeated parameter, a value
    out of range, or a combination that cannot be generated."""
    if not is_synthetic(uri):
        raise ValueError(f"not a synthetic:// uri: {uri!r}")
    kind, _, query = uri[len(SCHEME):].partition("?")
    kind = kind.strip("/")
    if kind not in KINDS:
        raise ValueError(f"unknown synthetic kind {kind!r} (one of {', '.join(sorted(KINDS))})")
    known = {**_COMMON, **KINDS[kind]["params"]}
    given: dict[str, str] = {}
    for name, value in urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query)):
        if name not in known:
            raise ValueError(f"synthetic://{kind} has no parameter {name!r} (it takes {', '.join(sorted(known))})")
        if name in given:
            raise ValueError(f"synthetic://{kind}: parameter {name!r} is given twice")
        given[name] = value
    params: dict[str, Any] = {}
    for name, rule in known.items():
        raw = given.get(name)
        if len(rule) == 2:  # a choice
            choice = rule[0] if raw is None else raw
            if choice not in rule[1]:
                raise ValueError(f"synthetic://{kind}: {name} must be one of {list(rule[1])}, got {choice!r}")
            params[name] = choice
            continue
        default, low, high = rule
        if raw is not None and not raw.isdigit():
            raise ValueError(f"synthetic://{kind}: {name} must be a whole number, got {raw!r}")
        number = int(default if raw is None else raw)
        low = _MIN_ROWS.get(kind, low) if name == "rows" else low
        high = _MAX_ROWS.get(kind, high) if name == "rows" else high
        if not low <= number <= high:
            raise ValueError(f"synthetic://{kind}: {name} must be between {low} and {high}, got {number}")
        params[name] = number
    rows = params["rows"]
    if kind == "tabular" and params["task"] == "regression" and "classes" in given:
        raise ValueError("synthetic://tabular: classes does not apply to task=regression")
    if "classes" in params and rows < params["classes"] and not (kind == "tabular" and params["task"] == "regression"):
        raise ValueError(f"synthetic://{kind}: rows ({rows}) must be at least classes ({params['classes']})")
    if kind == "triples":
        entities, relations = params["entities"], params["relations"]
        most = entities * (entities - 1) * relations
        if params["rule"] == "compose":
            if relations < 3:
                raise ValueError("synthetic://triples: rule=compose needs relations >= 3 (r0, r1 and their "
                                 f"composition r2), got {relations}")
            # the other edges use r3.. when there are such relations, so they never contradict the rule
            most = 3 * entities + entities * (entities - 1) * (relations - 3 if relations > 3 else 3) // 2
            if not 3 * entities <= rows <= most:
                raise ValueError(f"synthetic://triples: with rule=compose rows must be between 3 * entities "
                                 f"({3 * entities}, the rule's triples) and {most}, got {rows}")
        elif not entities - 1 <= rows <= most:
            raise ValueError(f"synthetic://triples: rows must be between entities - 1 ({entities - 1}, the chain "
                             f"that gives multi-hop paths) and {most} (every distinct triple), got {rows}")
    return SyntheticSource(kind=kind, params=params)


def check(uri: str, fmt: str) -> SyntheticSource:
    """``parse_uri`` plus: the spec's ``data.format`` is one this kind is written as."""
    source = parse_uri(uri)
    if fmt not in source.formats:
        raise ValueError(f"synthetic://{source.kind} is written as data.format "
                         f"{' or '.join(source.formats)}, not {fmt!r}")
    return source


def generate(uri: str, dest: Path, fmt: str) -> Path:
    """Write the dataset ``uri`` names under ``dest`` and return what ``data.uri`` should point at: the
    table file for csv / jsonl kinds, the dataset directory for the others. Anything a previous call left
    at that path is replaced, so the result depends on the uri alone."""
    source = check(uri, fmt)
    dest.mkdir(parents=True, exist_ok=True)
    rng = _rng(source)
    p = source.params
    if source.kind in _TABLES:
        name, writer = _TABLES[source.kind]
        path = dest / name
        path.write_text(writer(p, rng), encoding="utf-8")
        return path
    root = dest / _ROOTS.get(source.kind, f"synthetic-{source.kind}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if source.kind == "risk":
        from sakura.atelier import risk

        risk.make_synthetic_dataset(root, seed=p["seed"], months=p["rows"], fmt=fmt)
    else:
        _FOLDERS[source.kind](p, rng, root)
    return root


def _rng(source: SyntheticSource) -> np.random.Generator:
    # The kind is part of the stream, so `tabular` and `text` at the same seed are unrelated draws.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [source.params["seed"], zlib.crc32(source.kind.encode("ascii"))])))


# ------------------------------------------------------------------ words

_SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def _word(i: int) -> str:
    """A pronounceable made-up word, distinct for every ``i`` below 4900."""
    return _SYLLABLES[(i // len(_SYLLABLES)) % len(_SYLLABLES)] + _SYLLABLES[i % len(_SYLLABLES)]


def _balanced_labels(rows: int, classes: int, rng: np.random.Generator) -> npt.NDArray[Any]:
    """Class index per row, as even as ``rows`` allows, in a shuffled order."""
    labels = np.arange(rows) % classes
    rng.shuffle(labels)
    return labels


def _csv(header: list[str], rows: list[list[str]]) -> str:
    return "".join(",".join(r) + "\n" for r in [header, *rows])


# ------------------------------------------------------------------ tables


def _tabular(p: dict[str, Any], rng: np.random.Generator) -> str:
    rows, k = p["rows"], p["features"]
    names = [f"f{i}" for i in range(k)]
    if p["task"] == "classification":
        labels = _balanced_labels(rows, p["classes"], rng)
        centres = rng.normal(0.0, 2.0, size=(p["classes"], k))  # one cluster per class
        x = centres[labels] + rng.normal(0.0, 1.0, size=(rows, k))
        target, column = [f"c{c}" for c in labels], "label"
    else:
        x = rng.normal(0.0, 1.0, size=(rows, k))
        y = x @ rng.normal(0.0, 1.0, size=k) + rng.normal(0.0, 0.1, size=rows)  # linear, with a little noise
        target, column = [f"{v:.6f}" for v in y], "target"
    return _csv([*names, column], [[*(f"{v:.6f}" for v in x[i]), target[i]] for i in range(rows)])


def _timeseries(p: dict[str, Any], rng: np.random.Generator) -> str:
    t = np.arange(p["rows"])
    value = 10.0 + 0.01 * t + 2.0 * np.sin(2 * np.pi * t / p["period"]) + rng.normal(0.0, 0.1, size=len(t))
    return _csv(["t", "value"], [[str(int(i)), f"{v:.6f}"] for i, v in zip(t, value)])


def _text(p: dict[str, Any], rng: np.random.Generator) -> str:
    labels = _balanced_labels(p["rows"], p["classes"], rng)
    filler = [_word(i) for i in range(20)]
    out = []
    for c in labels:
        own = [_word(100 + int(c) * 8 + j) for j in range(8)]
        words = [own[int(rng.integers(8))] if rng.random() < 0.5 else filler[int(rng.integers(20))] for _ in range(8)]
        words[int(rng.integers(8))] = own[int(rng.integers(8))]  # at least one word of the class
        out.append([" ".join(words), f"c{c}"])
    return _csv(["text", "label"], out)


def _code_table(p: dict[str, Any], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    keys = [_word(200 + i) for i in range(p["keys"])]
    codes = [_word(1000 + int(i)) for i in rng.permutation(p["keys"])]
    return keys, codes


_PROMPTS = ("code for {key}:", "what is the code for {key}?")


def _instruct(p: dict[str, Any], rng: np.random.Generator) -> str:
    keys, codes = _code_table(p, rng)
    lines = []
    for i in range(p["rows"]):
        k = i % len(keys)  # every key appears; a holdout then asks about keys seen in training
        prompt = _PROMPTS[int(rng.integers(len(_PROMPTS)))].format(key=keys[k])
        lines.append(json.dumps({"prompt": prompt, "completion": codes[k]}, sort_keys=True))
    return "\n".join(lines) + "\n"


def _preference(p: dict[str, Any], rng: np.random.Generator) -> str:
    keys, codes = _code_table(p, rng)
    lines = []
    for i in range(p["rows"]):
        k = i % len(keys)
        wrong = (k + 1 + int(rng.integers(len(keys) - 1))) % len(keys)  # any key but k
        prompt = _PROMPTS[int(rng.integers(len(_PROMPTS)))].format(key=keys[k])
        lines.append(json.dumps({"prompt": prompt, "chosen": codes[k], "rejected": codes[wrong]}, sort_keys=True))
    return "\n".join(lines) + "\n"


def _triples(p: dict[str, Any], rng: np.random.Generator) -> str:
    entities, relations, rows = p["entities"], p["relations"], p["rows"]
    if p["rule"] == "compose":
        return _triples_compose(p, rng)
    # The chain e0 -> e1 -> ... -> e{E-1} is always there: every entity is reachable from e0, by a path of
    # up to E-1 hops. The rest are distinct random edges.
    triples = [(i, i % relations, i + 1) for i in range(entities - 1)]
    seen = set(triples)
    while len(triples) < rows:
        head, tail = int(rng.integers(entities)), int(rng.integers(entities))
        triple = (head, int(rng.integers(relations)), tail)
        if head != tail and triple not in seen:
            seen.add(triple)
            triples.append(triple)
    order = rng.permutation(len(triples))[:rows]
    return _csv(["head", "relation", "tail"],
                [[f"e{triples[i][0]}", f"r{triples[i][1]}", f"e{triples[i][2]}"] for i in order])


def _cycle(n: int, rng: np.random.Generator) -> npt.NDArray[Any]:
    """A random permutation of ``range(n)`` that is one single cycle, so no entity maps to itself."""
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):  # Sattolo's algorithm
        j = int(rng.integers(i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _triples_compose(p: dict[str, Any], rng: np.random.Generator) -> str:
    """``rule=compose``: r0 and r1 each map every entity to one other entity (a single random cycle), and
    r2 is their composition: ``(x, r2, z)`` holds exactly when ``(x, r0, y)`` and ``(y, r1, z)`` do. A
    triple of any of the three relations that is held out can therefore be reached in two hops through
    the other two (r0 = r2 then r1 backwards, r1 = r0 backwards then r2). The rows beyond those
    ``3 * entities`` are distinct random edges, of relations r3.. when there are any (they say nothing
    about the rule), else of r0..r2 (they are noise on it)."""
    entities, relations, rows = p["entities"], p["relations"], p["rows"]
    first = _cycle(entities, rng)
    while True:
        second = _cycle(entities, rng)
        both = second[first]
        if not (both == np.arange(entities)).any():  # r2 maps no entity to itself either
            break
    triples = [(i, r, int(to[i])) for r, to in enumerate((first, second, both)) for i in range(entities)]
    seen = set(triples)
    low = 3 if relations > 3 else 0
    while len(triples) < rows:
        head, tail = int(rng.integers(entities)), int(rng.integers(entities))
        triple = (head, int(rng.integers(low, relations)), tail)
        if head != tail and triple not in seen:
            seen.add(triple)
            triples.append(triple)
    order = rng.permutation(len(triples))
    return _csv(["head", "relation", "tail"],
                [[f"e{triples[i][0]}", f"r{triples[i][1]}", f"e{triples[i][2]}"] for i in order])


_DECISION_LEVELS = ("low", "medium", "high")
_DECISION_AMOUNTS = 10_000
_DECISION_ESCALATE_FROM = 5_000


def _decision(p: dict[str, Any], rng: np.random.Generator) -> str:
    """Rows of the ``decision`` task (``decision.py``): ``{"state", "questions", "answers"}``. The state is a
    made-up ticket (category, priority 1..5, amount, repeat); every row asks the same three questions, one of
    each type, and each answer is a function of the state alone:

    - ``queue`` (choice, ``options`` options): each category belongs to one queue, a table fixed by the seed;
    - ``urgency`` (score, low / medium / high): priority 1-2, 3, 4-5;
    - ``escalate`` (noul): true exactly when the ticket is a repeat and its amount is at least 5000.

    No two rows share a state, so the task's dedupe drops nothing and its split by state is a split by row."""
    n = p["options"]
    categories = [_word(300 + i) for i in range(n)]
    queues = [_word(2000 + i) for i in range(n)]
    queue_of = [int(q) for q in rng.permutation(n)]  # category index -> queue index
    questions = {
        "queue": {"type": "choice", "instructions": "Which queue handles this ticket?", "options": queues},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "levels": list(_DECISION_LEVELS)},
        "escalate": {"type": "noul", "statement": "This ticket must be escalated."},
    }
    seen: set[tuple[int, int, int, bool]] = set()
    lines = []
    for i in range(p["rows"]):
        c = i % n  # every category, so every queue, appears
        while True:
            ticket = (c, int(rng.integers(1, 6)), int(rng.integers(_DECISION_AMOUNTS)), bool(rng.integers(2)))
            if ticket not in seen:
                seen.add(ticket)
                break
        _, priority, amount, repeat = ticket
        state: Any = {"category": categories[c], "priority": priority, "amount": amount, "repeat": repeat}
        if p["state"] == "text":
            state = (f"category {categories[c]}; priority {priority}; amount {amount}; "
                     f"repeat {'yes' if repeat else 'no'}")
        answers = {"queue": queues[queue_of[c]],
                   "urgency": _DECISION_LEVELS[0 if priority <= 2 else 1 if priority == 3 else 2],
                   "escalate": repeat and amount >= _DECISION_ESCALATE_FROM}
        lines.append(json.dumps({"state": state, "questions": questions, "answers": answers}, sort_keys=True))
    return "\n".join(lines) + "\n"


_TABLES: dict[str, tuple[str, Callable[[dict[str, Any], np.random.Generator], str]]] = {
    "tabular": ("synthetic-tabular.csv", _tabular),
    "timeseries": ("synthetic-timeseries.csv", _timeseries),
    "text": ("synthetic-text.csv", _text),
    "instruct": ("synthetic-instruct.jsonl", _instruct),
    "preference": ("synthetic-preference.jsonl", _preference),
    "triples": ("synthetic-triples.csv", _triples),
    "decision": ("synthetic-decision.jsonl", _decision),
}


# ------------------------------------------------------------------ media


def _png(pixels: npt.NDArray[Any]) -> bytes:
    """An 8-bit RGB PNG of ``pixels`` (H x W x 3 uint8), written without an imaging library."""
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))  # filter type 0 on every row

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _write_wav(path: Path, samples: npt.NDArray[Any]) -> None:
    """Mono 16-bit PCM at 16 kHz from float samples in [-1, 1]."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())


def _tone(freq: float, seconds: float, rng: np.random.Generator) -> npt.NDArray[Any]:
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    wave_ = float(rng.uniform(0.3, 0.8)) * np.sin(2 * np.pi * freq * t + float(rng.uniform(0, 2 * np.pi)))
    fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.005 * SAMPLE_RATE))  # no clicks
    return np.asarray(wave_ * fade + rng.normal(0.0, 0.02, size=n))


def _colour(c: int, classes: int) -> npt.NDArray[Any]:
    """A saturated colour per class, evenly spaced around the hue circle."""
    h = 6.0 * c / classes
    x = 1.0 - abs(h % 2 - 1.0)
    rgb = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][int(h) % 6]
    return np.array(rgb) * 255.0


def _image(p: dict[str, Any], rng: np.random.Generator, root: Path) -> None:
    size, classes = p["size"], p["classes"]
    labels = _balanced_labels(p["rows"], classes, rng)
    yy, xx = np.mgrid[0:size, 0:size]
    for i, c in enumerate(labels):
        img = rng.normal(40.0, 12.0, size=(size, size, 3))  # dark, noisy background
        half = size * float(rng.uniform(0.2, 0.3))
        cy, cx = (float(v) for v in rng.uniform(half, size - half, size=2))
        if c % 2 == 0:  # even classes draw a square, odd classes a disc
            mask = (abs(yy - cy) <= half) & (abs(xx - cx) <= half)
        else:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= half ** 2
        img[mask] = _colour(int(c), classes) + rng.normal(0.0, 8.0, size=(int(mask.sum()), 3))
        d = root / f"c{c}"
        d.mkdir(exist_ok=True)
        (d / f"{i:06d}.png").write_bytes(_png(np.clip(img, 0, 255).astype(np.uint8)))


def _audio(p: dict[str, Any], rng: np.random.Generator, root: Path) -> None:
    labels = _balanced_labels(p["rows"], p["classes"], rng)
    for i, c in enumerate(labels):
        d = root / f"c{c}"
        d.mkdir(exist_ok=True)
        _write_wav(d / f"{i:06d}.wav", _tone(220.0 * 1.5 ** int(c), p["ms"] / 1000.0, rng))  # a fifth apart


def _asr(p: dict[str, Any], rng: np.random.Generator, root: Path) -> None:
    words = _ASR_WORDS[:p["words"]]
    (root / "clips").mkdir()
    gap = np.zeros(int(0.05 * SAMPLE_RATE))
    rows = []
    for i in range(p["rows"]):
        said = [int(w) for w in rng.integers(len(words), size=int(rng.integers(1, 4)))]
        parts: list[npt.NDArray[Any]] = []
        for w in said:  # one 150 ms tone per word, its pitch set by the word
            parts += [_tone(300.0 + 150.0 * w, 0.15, rng), gap]
        clip = np.concatenate(parts[:-1])
        name = f"clips/{i:06d}.wav"
        _write_wav(root / name, clip)
        rows.append([name, " ".join(words[w] for w in said), str(int(round(1000 * len(clip) / SAMPLE_RATE)))])
    (root / "manifest.csv").write_text(_csv(["audio", "transcript", "duration_ms"], rows), encoding="utf-8")


def _detection(p: dict[str, Any], rng: np.random.Generator, root: Path) -> None:
    """A YOLO dataset in the layout ``data._materialise_yolo`` and ``backends/hf_detection.py`` read:
    ``data.yaml`` (``train``, ``val``, ``names``), ``images/{train,val}/<i>.png`` and, next to them,
    ``labels/{train,val}/<i>.txt`` with one ``class cx cy w h`` line per box (normalised, box centre).

    Each image holds one to three solid rectangles on a dark noisy background; a rectangle's colour is its
    class's. Rectangles lie on whole pixels and do not overlap, so every label is exact. The first rectangle
    of image ``i`` takes the classes in turn (every class is drawn); the last ``max(1, rows // 5)`` images
    are the val split. ``data.yaml`` carries no ``path:``: the materialiser sets it to where the tree is."""
    size, classes, rows = p["size"], p["classes"], p["rows"]
    first = _balanced_labels(rows, classes, rng)
    n_val = max(1, rows // 5)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    low, high = max(4, size // 6), max(5, int(size * 0.4))
    for i in range(rows):
        img = rng.normal(40.0, 12.0, size=(size, size, 3))  # dark, noisy background
        boxes: list[tuple[int, int, int, int, int]] = []  # class, x0, y0, width, height in pixels
        wanted = int(rng.integers(1, 4))
        for _ in range(12 * wanted):  # a draw that overlaps an earlier rectangle is dropped and drawn again
            if len(boxes) == wanted:
                break
            c = int(first[i]) if not boxes else int(rng.integers(classes))
            w, h = (int(v) for v in rng.integers(low, high + 1, size=2))
            x0, y0 = int(rng.integers(0, size - w + 1)), int(rng.integers(0, size - h + 1))
            if any(x0 < bx + bw and bx < x0 + w and y0 < by + bh and by < y0 + h for _, bx, by, bw, bh in boxes):
                continue
            boxes.append((c, x0, y0, w, h))
            img[y0:y0 + h, x0:x0 + w] = _colour(c, classes) + rng.normal(0.0, 8.0, size=(h, w, 3))
        split = "val" if i >= rows - n_val else "train"
        (root / "images" / split / f"{i:06d}.png").write_bytes(_png(np.clip(img, 0, 255).astype(np.uint8)))
        (root / "labels" / split / f"{i:06d}.txt").write_text(
            "".join(f"{c} {(x0 + w / 2) / size:.6f} {(y0 + h / 2) / size:.6f} {w / size:.6f} {h / size:.6f}\n"
                    for c, x0, y0, w, h in boxes), encoding="utf-8")
    names = "".join(f"  {c}: c{c}\n" for c in range(classes))
    (root / "data.yaml").write_text(f"train: images/train\nval: images/val\nnames:\n{names}", encoding="utf-8")


def _image_regression(p: dict[str, Any], rng: np.random.Generator, root: Path) -> None:
    """The table ``data._materialise_image_table`` reads: ``table.csv`` with an ``image`` column (a path
    relative to the table) and a numeric ``target``, and ``images/<i>.png``. Each image is a bright square
    centred on a dark noisy background; the target is the square's side as a fraction of the image's side
    (0.2 to 0.8), so it can be read off the picture."""
    size = p["size"]
    (root / "images").mkdir()
    rows = []
    for i in range(p["rows"]):
        # ceil and floor, not round: at size 16 a rounded bound gave sides of 3 and 13, targets
        # of 0.1875 and 0.8125, outside the range the contract states.
        side = int(rng.integers(math.ceil(0.2 * size), math.floor(0.8 * size) + 1))
        img = rng.normal(40.0, 12.0, size=(size, size, 3))
        at = (size - side) // 2
        img[at:at + side, at:at + side] = 220.0 + rng.normal(0.0, 8.0, size=(side, side, 3))
        name = f"images/{i:06d}.png"
        (root / name).write_bytes(_png(np.clip(img, 0, 255).astype(np.uint8)))
        rows.append([name, f"{side / size:.6f}"])
    (root / "table.csv").write_text(_csv(["image", "target"], rows), encoding="utf-8")


_FOLDERS: dict[str, Callable[[dict[str, Any], np.random.Generator, Path], None]] = {
    "image": _image, "audio": _audio, "asr": _asr, "detection": _detection, "image_regression": _image_regression,
}

__all__ = ["KINDS", "SCHEME", "SyntheticSource", "check", "generate", "is_synthetic", "parse_uri"]
