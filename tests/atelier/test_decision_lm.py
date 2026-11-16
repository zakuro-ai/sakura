"""The decision readout, loss and training loop on a tiny fake causal LM: no GPU, no `transformers`."""

from __future__ import annotations

import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from sakura.atelier.backends import decision_lm as D  # noqa: E402
from sakura.atelier.decision import DecisionDataError, Instance, build_prompt, identity_order  # noqa: E402
from sakura.atelier.types import MetricsSink  # noqa: E402
from tests.atelier.decision_fakes import LETTER_BASE, CharTokenizer, FakeLM, SplitLetterTokenizer  # noqa: E402

TOK = CharTokenizer()


def inst(state="red", labels=("warm", "cold"), target=(1.0, 0.0), kind="choice", qid="q", text="Which?"):
    return Instance(state=state, qid=qid, kind=kind, text=text, labels=list(labels), target=list(target))


# ---------------------------------------------------------------- readout

def test_every_option_letter_is_a_single_distinct_token():
    r = D.make_readout(TOK)
    assert r.letters == [LETTER_BASE + i for i in range(26)] and r.start == []


def test_a_tokenizer_that_splits_the_letters_is_refused_by_name():
    with pytest.raises(RuntimeError, match="not a single token"):
        D.make_readout(SplitLetterTokenizer())


def test_the_letter_is_the_token_after_the_answer_slot_in_ordinary_text_too():
    class Merging(CharTokenizer):
        def encode(self, text, add_special_tokens=True):
            ids = super().encode(text, add_special_tokens)
            return ids[:-2] + [9] if text.endswith(":") and len(ids) > 3 else ids  # "Answer:" merges its tail

    with pytest.raises(RuntimeError, match="Answer:"):
        D.make_readout(Merging())


def test_special_tokens_a_tokenizer_starts_with_are_kept():
    class WithBos(CharTokenizer):
        def encode(self, text, add_special_tokens=True):
            return ([5] if add_special_tokens else []) + super().encode(text, add_special_tokens)

    assert D.make_readout(WithBos()).start == [5]


def test_the_sequence_ends_at_the_answer_slot_with_no_eos():
    r = D.make_readout(TOK)
    e = D.encode_instance(TOK, r, inst(), identity_order(2), max_len=512)
    ids = e.prefix + e.suffix
    assert ids[-len(TOK.encode("Answer:")):] == TOK.encode("Answer:")
    assert TOK.eos_token_id not in ids


def test_prefix_and_suffix_are_tokenised_apart_and_joined():
    r = D.make_readout(TOK)
    e = D.encode_instance(TOK, r, inst(), [1, 0], max_len=512)
    prefix, suffix = build_prompt(inst(), [1, 0])
    assert e.prefix == TOK.encode(prefix) and e.suffix == TOK.encode(suffix) and not e.truncated


def test_a_state_over_the_budget_is_cut_but_the_question_and_options_are_not():
    r = D.make_readout(TOK)
    long = inst(state="x" * 5000)
    e = D.encode_instance(TOK, r, long, identity_order(2), max_len=300)
    assert e.truncated and len(e.prefix) + len(e.suffix) <= 300
    assert e.suffix == TOK.encode(build_prompt(long, identity_order(2))[1])
    assert len(e.prefix) > 250  # as much of the state as fits, not a stub


def test_a_question_that_alone_exceeds_the_budget_is_a_data_error():
    with pytest.raises(DecisionDataError, match="max_seq_length"):
        D.encode_instance(TOK, D.make_readout(TOK), inst(text="why " * 500), identity_order(2), max_len=100)


# --------------------------------------------------- restricted readout, loss

class TableLM(torch.nn.Module):
    """Returns a fixed logits row at every position, whatever the input: lets a test choose exact logits."""

    def __init__(self, row):
        super().__init__()
        self.row = torch.tensor(row, dtype=torch.float32)
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **_):
        b, t = input_ids.shape
        return type("O", (), {"logits": (self.row + self.w).expand(b, t, -1), "past_key_values": None})()


def table(**letters):
    row = [-1000.0] * 3100
    for letter, v in letters.items():
        row[LETTER_BASE + ord(letter) - ord("A")] = v
    return row


def test_the_answer_is_read_from_the_k_option_letters_only():
    r = D.make_readout(TOK)
    # a huge logit on letter D (not an option of a 3-way question) and on a non-letter token must not matter
    row = table(A=1.0, B=2.0, C=0.5, D=1e6)
    row[100] = 1e9
    got = D.score_instances(TableLM(row), TOK, r, [inst(labels=("a", "b", "c"), target=(0, 1, 0))], 512, "cpu")
    assert got[0].logits == pytest.approx([1.0, 2.0, 0.5]) and len(got[0].logits) == 3


def test_the_loss_is_the_cross_entropy_of_the_restricted_softmax():
    r = D.make_readout(TOK)
    model = TableLM(table(A=1.0, B=2.0, C=0.5))
    batch = [(inst(labels=("a", "b", "c"), target=(0, 1, 0)), identity_order(3))]
    loss, _ = D.batch_loss(model, TOK, r, batch, 512, "cpu")
    z = [1.0, 2.0, 0.5]
    expected = -(z[1] - math.log(sum(math.exp(v) for v in z)))
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-5)


def test_a_soft_label_is_trained_against_its_distribution():
    r = D.make_readout(TOK)
    model = TableLM(table(A=1.0, B=2.0, C=0.5))
    target = (0.5, 0.3, 0.2)
    loss, _ = D.batch_loss(model, TOK, r, [(inst(labels=("a", "b", "c"), target=target), identity_order(3))], 512, "cpu")
    z = [1.0, 2.0, 0.5]
    logz = math.log(sum(math.exp(v) for v in z))
    assert float(loss.detach()) == pytest.approx(-sum(t * (v - logz) for t, v in zip(target, z)), abs=1e-5)


def test_the_target_follows_the_option_order_shown():
    r = D.make_readout(TOK)
    model = TableLM(table(A=3.0, B=0.0))
    i = inst(labels=("warm", "cold"), target=(1.0, 0.0))
    straight, _ = D.batch_loss(model, TOK, r, [(i, [0, 1])], 512, "cpu")   # warm is letter A: logit 3
    swapped, _ = D.batch_loss(model, TOK, r, [(i, [1, 0])], 512, "cpu")    # warm is letter B: logit 0
    assert float(straight.detach()) < 0.1 < float(swapped.detach())


def test_gradients_reach_only_the_option_letter_rows_of_the_head():
    r = D.make_readout(TOK)
    model = FakeLM()
    loss, _ = D.batch_loss(model, TOK, r, [(inst(), identity_order(2))], 512, "cpu")
    loss.backward()
    grad = model.head.weight.grad
    nonzero = {int(i) for i in torch.nonzero(grad.abs().sum(dim=1)).flatten()}
    assert nonzero == {r.letters[0], r.letters[1]}  # the K letters of a 2-way question, nothing else


def test_a_batch_of_different_lengths_scores_as_each_alone():
    r = D.make_readout(TOK)
    model = FakeLM(seed=3)
    items = [inst(state="red"), inst(state="a much longer state " * 5, labels=("a", "b", "c"), target=(0, 0, 1)),
             inst(state="blue", labels=("a", "b"))]
    together = D.score_instances(model, TOK, r, items, 512, "cpu", batch_size=3)
    alone = [D.score_instances(model, TOK, r, [i], 512, "cpu")[0] for i in items]
    for a, b in zip(together, alone):
        assert a.logits == pytest.approx(b.logits, abs=1e-5)


def test_scored_rows_carry_their_question_and_target():
    r = D.make_readout(TOK)
    got = D.score_instances(FakeLM(), TOK, r, [inst(qid="topic", kind="choice", target=(0.0, 1.0))], 512, "cpu")[0]
    assert (got.qid, got.kind, got.target) == ("topic", "choice", [0.0, 1.0])


# ------------------------------------------------------------- shared prefix

def test_questions_on_one_state_share_its_prefix_and_score_the_same_as_alone():
    r = D.make_readout(TOK)
    model = FakeLM(seed=1)
    state = "Hi, I was charged twice for March."
    insts = [inst(state=state, qid="a"), inst(state=state, qid="b", labels=("x", "y", "z"), target=(1, 0, 0)),
             inst(state=state, qid="c", kind="noul", labels=("true", "false"), text="Is it?")]
    shared = D.score_state(model, TOK, r, state, insts, 512, "cpu")
    alone = [D.score_state(model, TOK, r, state, [i], 512, "cpu")[0] for i in insts]
    for a, b in zip(shared, alone):
        assert a == pytest.approx(b, abs=1e-5)
    assert [len(x) for x in shared] == [2, 3, 2]


def test_the_state_prefix_is_run_once_not_once_per_question():
    r = D.make_readout(TOK)
    model = FakeLM()
    state = "s" * 40
    insts = [inst(state=state, qid=f"q{i}") for i in range(4)]
    D.score_state(model, TOK, r, state, insts, 512, "cpu")
    assert model.calls == 1 + 4  # one prefix pass, then one short pass per question
    model.calls = 0
    D.score_state(model, TOK, r, state, insts[:1], 512, "cpu")
    assert model.calls == 1  # a single question is just one forward pass


def test_one_question_does_not_leak_into_the_next_through_the_cache():
    r = D.make_readout(TOK)
    model = FakeLM(seed=2)
    state = "hello there"
    a, b = inst(state=state, qid="a", text="first?"), inst(state=state, qid="b", text="second?")
    ab = D.score_state(model, TOK, r, state, [a, b], 512, "cpu")
    ba = D.score_state(model, TOK, r, state, [b, a], 512, "cpu")
    assert ab[0] == pytest.approx(ba[1], abs=1e-5) and ab[1] == pytest.approx(ba[0], abs=1e-5)


def test_the_cached_path_matches_the_training_path_for_the_same_example():
    # score_state (cached) must see the same ids score_instances (batched, left-padded) does
    r = D.make_readout(TOK)
    model = FakeLM(seed=4)
    state = "same state for both"
    insts = [inst(state=state, qid="a"), inst(state=state, qid="b", text="other?")]
    cached = D.score_state(model, TOK, r, state, insts, 512, "cpu")
    batched = D.score_instances(model, TOK, r, insts, 512, "cpu", batch_size=2)
    for c, b in zip(cached, batched):
        assert c == pytest.approx(b.logits, abs=1e-5)


# ------------------------------------------------------------------ training

def easy_instances(n):
    out = []
    for i in range(n):
        warm = i % 2 == 0
        out.append(inst(state=("red sun fire " if warm else "blue ice snow ") + str(i), qid="q",
                        target=(1.0, 0.0) if warm else (0.0, 1.0)))
    return out


def cfg(**kw):
    base = dict(epochs=6, batch_size=4, grad_accum=1, learning_rate=0.05, max_len=256, seed=0)
    return D.TrainConfig(**{**base, **kw})


def run_loop(tmp_path, model=None, **kw):
    model = model or FakeLM(seed=7)
    r = D.make_readout(TOK)
    train, calib = easy_instances(24), easy_instances(8)
    path = tmp_path / "metrics.jsonl"
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    with MetricsSink(path) as sink:
        summary = D.train_loop(model, TOK, r, train, calib, cfg=cfg(**kw), device="cpu", optimizer=opt,
                               metrics=sink, order_fn=lambda rng, k: identity_order(k))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return model, summary, lines


def test_training_lowers_the_held_out_log_loss(tmp_path):
    model = FakeLM(seed=7)
    r = D.make_readout(TOK)
    before = D.score_instances(model, TOK, r, easy_instances(8), 256, "cpu")
    from sakura.atelier.decision import split_metrics
    base = split_metrics(before, {})["log_loss"]
    _, summary, lines = run_loop(tmp_path, model)
    assert summary["log_loss"] < base * 0.5
    assert summary["accuracy"] == 1.0


def test_metrics_lines_have_the_existing_shape_and_names(tmp_path):
    _, _, lines = run_loop(tmp_path, epochs=2)
    assert {tuple(sorted(x)) for x in lines} == {("epoch", "name", "split", "step", "t", "value")}
    train = [x for x in lines if x["split"] == "train"]
    val = [x for x in lines if x["split"] == "validation"]
    assert {x["name"] for x in train} == {"loss"} and {x["name"] for x in val} == {"log_loss", "accuracy"}
    assert [x["epoch"] for x in val if x["name"] == "accuracy"] == [1.0, 2.0]  # one validation point per epoch


def test_grad_accum_takes_one_optimizer_step_per_group_of_batches(tmp_path):
    _, _, lines = run_loop(tmp_path, epochs=1, grad_accum=2)
    steps = [x["step"] for x in lines if x["split"] == "train"]
    assert steps == [0, 0, 1, 1, 2, 2]  # 24 examples / batch 4 = 6 micro-batches -> 3 optimizer steps


def test_options_are_reshuffled_per_example_per_epoch():
    seen = []
    model, r = FakeLM(), D.make_readout(TOK)

    def spy(rng, k):
        order = random.sample(range(k), k)
        seen.append(tuple(order))
        return order

    train = easy_instances(8)
    D.train_loop(model, TOK, r, train, easy_instances(4), cfg=cfg(epochs=3, batch_size=4), device="cpu",
                 optimizer=torch.optim.SGD(model.parameters(), lr=0.01), metrics=None, order_fn=spy)
    assert len(seen) == 8 * 3  # one draw per example per epoch
    assert len(set(seen)) == 2  # (K = 2: both orders occur)


def test_scoring_uses_the_option_order_given_never_a_shuffled_one():
    class Recording(FakeLM):
        seen: list = []

        def forward(self, input_ids, *a, **kw):
            self.seen.append(input_ids[0][-60:].tolist())
            return super().forward(input_ids, *a, **kw)

    model, r = Recording(seed=5), D.make_readout(TOK)
    i = inst(state="red", labels=("warm", "cold", "hot"), target=(1.0, 0.0, 0.0))
    D.score_instances(model, TOK, r, [i], 256, "cpu")
    given = D.encode_instance(TOK, r, i, [0, 1, 2], 256)
    shuffled = D.encode_instance(TOK, r, i, [2, 0, 1], 256)
    assert given.suffix != shuffled.suffix  # the order is visible in the ids...
    assert model.seen[0][-len(given.suffix):] == given.suffix  # ...and scoring fed the given one


def test_training_is_seeded(tmp_path):
    a, _, _ = run_loop(tmp_path, FakeLM(seed=9), epochs=2)
    (tmp_path / "metrics.jsonl").unlink()
    b, _, _ = run_loop(tmp_path, FakeLM(seed=9), epochs=2)
    assert torch.equal(a.head.weight, b.head.weight)


def test_training_with_shuffled_options_still_runs_end_to_end(tmp_path):
    model, r = FakeLM(seed=1), D.make_readout(TOK)
    summary = D.train_loop(model, TOK, r, easy_instances(16), easy_instances(8), cfg=cfg(epochs=2), device="cpu",
                           optimizer=torch.optim.Adam(model.parameters(), lr=0.02), metrics=None)
    assert math.isfinite(summary["log_loss"]) and 0.0 <= summary["accuracy"] <= 1.0
