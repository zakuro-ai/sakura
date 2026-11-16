"""The peft text-generation loop on a tiny random causal language model, on a CPU, in seconds: the learning-rate
schedule, clipping, evaluations inside an epoch, early stopping and the best checkpoint (CONTRACTS §2.2), and
first of all that a spec with none of those keys trains and reports exactly as it did before them.

Needs ``transformers`` and ``peft`` (the backend's own venv, ``docs/atelier/envs/peft.txt``); skipped elsewhere."""

from __future__ import annotations

import json
import random

import pytest

from sakura.atelier.backends import peft as backend_mod
from sakura.atelier.types import RunDir
from tests.atelier.tiny_models import tiny_llama_dir

pytest.importorskip("transformers")
pytest.importorskip("peft")

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 =:,{}\"\n"
ROWS = [{"prompt": f"k{i}=", "completion": "ab"[i % 2] * (1 + i % 3)} for i in range(9)]


class _Metrics:
    def __init__(self):
        self.rows = []

    def log(self, name, value, *, step, split, epoch):
        self.rows.append((name, split, step, epoch, value))

    def series(self, name, split):
        return [(r[2], r[4]) for r in self.rows if r[0] == name and r[1] == split]


def _resolved(tmp_path, hp, limits=""):
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    spec = load_spec(f"""
atelier: 1
name: tiny
task: text_generation
data: {{uri: "file://{tmp_path}/x.jsonl", format: jsonl, split: {{validation_fraction: 0.5}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 7
{limits}
""").model_copy(update={"hyperparameters": hp})
    return resolve(spec)


def _data(train, validation):
    return type("D", (), {"train": train, "validation": validation})()


def _train(tmp_path, hp, train, validation, *, limits="", name="run"):
    import torch

    out = RunDir(tmp_path / name).ensure()
    metrics = _Metrics()
    torch.manual_seed(0)
    result = backend_mod.BACKEND.train(_resolved(tmp_path, hp, limits), _data(train, validation), out, metrics, None)
    return result, metrics, out


def _hp(base, **over):
    return {"base_model": str(base), "epochs": 2, "batch_size": 2, "learning_rate": 5e-3, "lora_r": 4,
            "lora_alpha": 8, "lora_target_modules": ["q_proj", "v_proj"], "max_seq_length": 48, **over}


# ------------------------------------------------------------------ a spec without the keys is unchanged

def _loop_before_the_keys(hp, seed, train_examples, val_examples, max_steps=None):
    """The text-generation loop as it was before the schedule and evaluation keys (peft.py at 7dc9456), kept
    as the reference a spec without them is held to: AdamW at a constant rate, one validation per epoch on
    every row, one row at a time, 8 generated tokens, first line compared. Returns the logged points and the
    final metrics."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hp["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hp["base_model"], dtype=torch.float32).to("cpu")
    model = get_peft_model(model, LoraConfig(
        r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp.get("lora_dropout", 0.05),
        target_modules=hp["lora_target_modules"], task_type="CAUSAL_LM"))
    model.train()
    max_len, batch_size, grad_accum = hp["max_seq_length"], hp["batch_size"], hp.get("grad_accum", 1)

    def encode(ex):
        text = ex["prompt"] + ex["completion"] + tokenizer.eos_token
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        prompt_len = len(tokenizer(ex["prompt"], truncation=True, max_length=max_len)["input_ids"])
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[enc["attention_mask"] == 0] = -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def collate(batch):
        width = max(int(b["input_ids"].shape[1]) for b in batch)
        ids = torch.full((len(batch), width), int(tokenizer.pad_token_id), dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for row, b in enumerate(batch):
            n = int(b["input_ids"].shape[1])
            ids[row, :n], attn[row, :n], labels[row, :n] = b["input_ids"][0], b["attention_mask"][0], b["labels"][0]
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["learning_rate"])
    rng = random.Random(seed)
    rows = []

    @torch.no_grad()
    def validate(step, epoch):
        model.eval()
        losses, correct = [], 0
        for ex in val_examples:
            losses.append(float(model(**collate([encode(ex)])).loss))
            prompt_ids = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"]
            gen = model.generate(input_ids=prompt_ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            completion = str(tokenizer.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=True))
            correct += completion.strip().split("\n")[0].strip() == ex["completion"].strip()
        model.train()
        scores = (sum(losses) / len(losses), correct / len(val_examples))
        rows.append(("loss", "validation", step, epoch, scores[0]))
        rows.append(("exact_match", "validation", step, epoch, scores[1]))
        return scores

    step, stopped, scores = 0, False, (0.0, 0.0)
    for epoch in range(hp["epochs"]):
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for bi, batch_idx in enumerate(batches):
            if max_steps is not None and step >= max_steps:
                stopped = True
                break
            loss = model(**collate([encode(train_examples[j]) for j in batch_idx])).loss
            (loss / grad_accum).backward()
            boundary = (bi + 1) % grad_accum == 0 or bi == len(batches) - 1
            if boundary:
                optim.step()
                optim.zero_grad()
            rows.append(("loss", "train", step, epoch + bi * batch_size / len(order), float(loss.detach())))
            if boundary:
                step += 1
        if stopped:
            optim.zero_grad()
            scores = validate(step, epoch + bi * batch_size / len(order))
            break
        scores = validate(step, float(epoch + 1))
    return rows, {"loss": scores[0], "exact_match": scores[1]}


@pytest.mark.parametrize("over, max_steps", [
    ({}, None),  # lora-fast@1's shape: one optimiser step per batch, dropout drawn from the torch generator
    ({"grad_accum": 2, "lora_dropout": 0.0}, None),  # 5 batches in groups of 2: the last group is one batch
    ({"epochs": 3}, 6),  # limits.max_steps ends the run inside the second epoch
])
def test_a_spec_without_the_new_keys_trains_and_reports_exactly_as_before(tmp_path, over, max_steps):
    import torch

    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, **over)
    train, validation = ROWS, ROWS[:4]
    torch.manual_seed(0)
    want_rows, want_metrics = _loop_before_the_keys(hp, 7, train, validation, max_steps)
    limits = f"limits: {{max_steps: {max_steps}}}" if max_steps else ""
    result, metrics, out = _train(tmp_path, hp, train, validation, limits=limits)

    # the same points: names, splits, steps and epochs exactly; the values to float error, since the loss is now
    # taken on the reply positions only (a different order of float operations from the model's own loss)
    assert [r[:4] for r in metrics.rows] == [r[:4] for r in want_rows]
    assert [r[4] for r in metrics.rows] == pytest.approx([r[4] for r in want_rows], rel=1e-4, abs=1e-6)
    assert result.metrics == pytest.approx(want_metrics, rel=1e-4, abs=1e-6)
    assert len({v for _, v in metrics.series("loss", "train")}) > 1  # it did train: the loss moves
    assert (result.epoch_metrics, result.early_stop) == (None, None)
    # the one addition a spec without the keys gets: what the hub sizes and prices a job from (CONTRACTS §2.2)
    assert set(result.report) == {"truncation", "training"}
    assert result.report["training"]["steps_done"] == max(step for step, _ in metrics.series("loss", "train")) + 1
    assert (result.stopped_early or {}).get("reason") == ("max_steps" if max_steps else None)
    assert set(json.loads((out.artifacts / "preprocess.json").read_text())) == {
        "prompt_template", "base_model", "base_revision", "base_commit", "max_seq_length", "eos_token"}
    # the evaluation-field keys (#238) change nothing for a spec that sets none: records, options and report
    assert all(set(row) == {"prompt", "completion"} for row in [*train, *validation])
    options = backend_mod.train_eval_options(hp)
    assert options["eval_scope"] == "all" and options["extended"] is False
    assert "generation_eval" not in result.report


def _padded_batch(tokenizer, texts, prompt_len=2):
    import torch

    rows = [tokenizer(text)["input_ids"] for text in texts]
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [tokenizer.pad_token_id] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :prompt_len] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def test_the_loss_on_the_reply_positions_equals_the_models_own_loss_and_never_hands_it_the_labels(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32).eval()
    batch = _padded_batch(tokenizer, ["k1=ab", "k22=abcabc", "k=a b c d e f"])
    with torch.no_grad():
        own = model(**batch).loss
        seen = []
        hook = model.register_forward_pre_hook(lambda _m, _a, kw: seen.append(sorted(kw)), with_kwargs=True)
        mine = backend_mod._reply_loss(model(**backend_mod._model_inputs(batch)).logits, batch["labels"])
        rows = backend_mod._row_losses(model, batch)
        hook.remove()
        alone = [float(model(**_padded_batch(tokenizer, [t])).loss) for t in ("k1=ab", "k22=abcabc", "k=a b c d e f")]
    assert torch.allclose(mine, own, rtol=1e-5, atol=1e-6)
    assert rows == pytest.approx(alone, rel=1e-5)
    assert seen and all("labels" not in kw for kw in seen)  # the model builds no loss of its own


# ------------------------------------------------------------------ schedule, clipping, weight decay, adapters

def test_the_rate_follows_the_schedule_and_every_optimiser_step_is_clipped(tmp_path, monkeypatch):
    import peft
    import torch

    from sakura.atelier.lr_schedule import lr_factor

    clips, adamw, lora = [], {}, {}
    real_clip, real_adamw, real_lora = torch.nn.utils.clip_grad_norm_, torch.optim.AdamW, peft.LoraConfig
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda params, norm: clips.append(norm) or real_clip(params, norm))
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, **kw: adamw.update(kw) or real_adamw(params, **kw))
    monkeypatch.setattr(peft, "LoraConfig", lambda **kw: lora.update(kw) or real_lora(**kw))
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, grad_accum=2, lr_schedule="cosine", warmup_ratio=0.25, min_lr_ratio=0.1, max_grad_norm=0.5,
             weight_decay=0.0, lora_use_rslora=True, lora_use_dora=True)
    result, metrics, out = _train(tmp_path, hp, ROWS, ROWS[:4])

    # 9 rows in batches of 2 are 5 batches: groups of 2, 2 and 1, so 3 optimiser steps an epoch and 6 in all;
    # the warmup is ceil(0.25 * 6) = 2 of them
    rates = metrics.series("lr", "train")
    assert [step for step, _ in rates] == [0, 1, 2, 3, 4, 5]
    assert [rate for _, rate in rates] == pytest.approx([5e-3 * lr_factor(s, 6, "cosine", 2, 0.1) for s in range(6)])
    assert rates[0][1] == 0.0 and rates[1][1] == pytest.approx(2.5e-3) and rates[2][1] == pytest.approx(5e-3)
    assert rates[5][1] < rates[4][1] < rates[3][1] < 5e-3
    assert [step for step, _ in metrics.series("loss", "train")] == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
    assert clips == [0.5] * 6  # once per optimiser step, after the accumulation
    assert adamw == {"lr": 5e-3, "weight_decay": 0.0}
    assert (lora["use_rslora"], lora["use_dora"]) == (True, True)
    assert not list((out.checkpoint / "merged").glob("adapter*"))  # DoRA merges into a plain dense model too
    assert result.early_stop["reason"] == "max_epochs" and result.early_stop["kept"] == "last"


def test_without_the_keys_the_optimiser_and_the_adapter_are_built_as_before(tmp_path, monkeypatch):
    import peft
    import torch

    adamw, lora = {}, {}
    real_adamw, real_lora = torch.optim.AdamW, peft.LoraConfig
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, **kw: adamw.update(kw) or real_adamw(params, **kw))
    monkeypatch.setattr(peft, "LoraConfig", lambda **kw: lora.update(kw) or real_lora(**kw))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **kw: pytest.fail("clipped without max_grad_norm"))
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", lambda *a, **kw: pytest.fail("a schedule without the keys"))
    _, metrics, _ = _train(tmp_path, _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1), ROWS, ROWS[:2])
    assert adamw == {"lr": 5e-3} and not {"use_rslora", "use_dora"} & set(lora)
    assert {name for name, *_ in metrics.rows} == {"loss", "exact_match"}  # no lr series


# ------------------------------------------------------------------ limits and wrap-up, with the keys

def test_a_step_cap_shortens_the_schedule_and_the_run_is_scored_where_it_stopped(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=3, lr_schedule="linear", group_by_length=True, eval_every=0.4)
    result, metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4], limits="limits: {max_steps: 4}")
    # 15 steps were planned; the cap makes the plan 4 steps, so the rate reaches the floor's last notch at step 3
    assert [rate for _, rate in metrics.series("lr", "train")] == pytest.approx([5e-3, 3.75e-3, 2.5e-3, 1.25e-3])
    assert result.stopped_early == {"reason": "max_steps", "epoch": pytest.approx(4 * 2 / 9), "step": 4}
    # evaluated every 2 steps (0.4 of the 5 steps of an epoch); step 4 was scored once, not again at the stop
    assert [step for step, _ in metrics.series("loss", "validation")] == [2, 4]
    assert result.early_stop["reason"] == "max_steps" and result.early_stop["stopped_step"] == 4
    assert result.metrics["loss"] == metrics.series("loss", "validation")[-1][1]  # what was trained is what is scored
    assert (result.checkpoint.path / "merged" / "model.safetensors").is_file()


def test_a_wrap_up_request_inside_an_epoch_stops_there_and_scores_what_was_trained(tmp_path):
    import torch

    out = RunDir(tmp_path / "run").ensure()

    class Requests(_Metrics):
        def log(self, name, value, *, step, split, epoch):
            super().log(name, value, step=step, split=split, epoch=epoch)
            if (name, split, step) == ("loss", "train", 2):
                (out.root / "wrap-up-requested").write_text("")  # as the runner does, during the third step

    metrics = Requests()
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), lr_schedule="cosine", warmup_ratio=0.1, keep_best=True)
    torch.manual_seed(0)
    result = backend_mod.BACKEND.train(_resolved(tmp_path, hp), _data(ROWS, ROWS[:4]), out, metrics, None)
    assert result.stopped_early == {"reason": "wrap_up", "epoch": pytest.approx(3 * 2 / 9), "step": 3}
    assert [step for step, _ in metrics.series("loss", "train")] == [0, 1, 2]
    assert [step for step, _ in metrics.series("loss", "validation")] == [3]
    assert result.early_stop["reason"] == "wrap_up" and result.early_stop["kept_step"] == 3
    assert set(result.metrics) == {"loss", "exact_match"}
    assert (out.checkpoint / "merged" / "model.safetensors").is_file()


# ------------------------------------------------------------------ evaluations inside an epoch, early stop, best

#: Validation rows that contradict the training rows: the more the model learns, the worse its validation loss.
OPPOSITE = [{"prompt": r["prompt"], "completion": r["completion"].translate(str.maketrans("ab", "yz"))} for r in ROWS[:4]]


def _position(step):
    """Where optimiser step ``step`` ends, in epochs: 5 steps an epoch, each a batch of 2 of the 9 rows."""
    return step / 5 if step % 5 == 0 else step // 5 + (step % 5) * 2 / 9


#: With eval_every 0.4 and 5 optimiser steps an epoch: every round(0.4 * 5) = 2 steps, and at the end of the epoch.
CADENCE = [2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20]


@pytest.mark.parametrize("keep_best", [True, False])
def test_evaluations_at_the_cadence_stop_early_and_the_best_adapter_is_the_one_exported(tmp_path, keep_best):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=4, learning_rate=2e-2, lora_target_modules="all-linear", lora_dropout=0.0, eval_every=0.4,
             early_stop_patience=3, keep_best=keep_best)
    result, metrics, out = _train(tmp_path, hp, ROWS, OPPOSITE)

    losses = metrics.series("loss", "validation")
    steps = [step for step, _ in losses]
    assert steps == CADENCE[:len(steps)] and len(steps) < len(CADENCE)  # at the cadence, and stopped before the end
    best = min(range(len(losses)), key=lambda i: losses[i][1])
    assert len(losses) == best + 1 + 3  # three evaluations in a row without improvement, then the stop
    stop = result.early_stop
    assert (stop["metric"], stop["field"], stop["mode"], stop["patience"], stop["enabled"]) == ("loss", "completion", "min", 3, True)
    assert (stop["best_step"], stop["best_value"]) == (steps[best], round(losses[best][1], 6))
    assert stop["best_epoch"] == pytest.approx(_position(steps[best]), abs=1e-6)
    assert (stop["stopped_step"], stop["max_epochs"]) == (steps[-1], 4)
    assert stop["stopped_epoch"] == pytest.approx(_position(steps[-1]), abs=1e-6)
    assert (stop["reason"], stop["evaluations"]) == ("early_stop", len(losses))
    assert (stop["kept"], stop["kept_step"]) == (("best", steps[best]) if keep_best else ("last", steps[-1]))
    assert [step for step, _ in metrics.series("loss", "train")] == list(range(steps[-1]))  # nothing after the stop
    # the final loss is measured on the weights that are merged: the best evaluation's, or the last one's
    assert result.metrics["loss"] == pytest.approx(losses[best if keep_best else -1][1], rel=1e-5)
    assert losses[-1][1] > 1.01 * losses[best][1]  # so the two cases above are told apart
    assert result.stopped_early is None  # that key is for wrap-up and limits

    rows = result.epoch_metrics
    assert [r["step"] for r in rows] == steps
    assert [r["epoch"] for r in rows] == pytest.approx([_position(step) for step in steps], abs=1e-6)
    assert all(set(r["validation"]) == {"loss", "exact_match"} and set(r["train"]) == {"loss"} for r in rows)
    assert rows[1]["validation"]["loss"] == round(losses[1][1], 6)
    train = [v for _, v in metrics.series("loss", "train")]
    assert rows[1]["train"]["loss"] == round((train[2] + train[3]) / 2, 6)  # the mean since the last evaluation
    assert not (out.checkpoint / "best.pt").exists() and not list((out.checkpoint / "merged").glob("adapter*"))


def test_a_model_trained_in_full_keeps_its_best_weights_on_disk_and_restores_them(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, init="random", epochs=4, learning_rate=5e-3, eval_every=0.4, early_stop_patience=2, keep_best=True)
    result, metrics, out = _train(tmp_path, hp, ROWS, OPPOSITE)
    losses = metrics.series("loss", "validation")
    best = min(losses, key=lambda point: point[1])
    assert result.early_stop["kept_step"] == best[0] < losses[-1][0] and result.early_stop["reason"] == "early_stop"
    assert result.metrics["loss"] == pytest.approx(best[1], rel=1e-5) and losses[-1][1] > 1.01 * best[1]
    assert result.provenance["training"] == "full"
    assert not (out.checkpoint / "best.pt").exists()  # removed once restored: it is not part of the checkpoint


def test_evaluation_subsets_are_fixed_for_the_run_and_the_final_evaluation_reads_every_row(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, eval_every=0.4, eval_max_rows=3, gen_eval_rows=0, final_gen_rows=2, eval_batch_size=2)
    seen = []
    real = backend_mod._generate
    try:
        backend_mod._generate = lambda net, tok, prompts, n, device: seen.append(list(prompts)) or real(net, tok, prompts, n, device)
        result, metrics, _ = _train(tmp_path, hp, ROWS, ROWS)
    finally:
        backend_mod._generate = real
    assert [step for step, _ in metrics.series("loss", "validation")] == [2, 4, 5]
    assert metrics.series("exact_match", "validation") == []  # gen_eval_rows: 0, a periodic evaluation is loss only
    assert all(set(r["validation"]) == {"loss"} for r in result.epoch_metrics)
    assert len(seen) == 1 and len(seen[0]) == 2  # the final evaluation generated from 2 rows, as one batch
    assert set(result.metrics) == {"loss", "exact_match"}
    # the final loss is over all 9 rows, a periodic one over 3: not the same number
    assert result.metrics["loss"] != metrics.series("loss", "validation")[-1][1]


def test_a_batch_of_rows_scores_like_the_rows_one_at_a_time(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32).eval()
    prompts = ["k1=", "a longer prompt: {", "x", "the longest prompt of the four, by far="]
    with torch.no_grad():
        together = backend_mod._generate(model, tokenizer, prompts, 12, "cpu")
        alone = [backend_mod._generate(model, tokenizer, [p], 12, "cpu")[0] for p in prompts]
    assert together == alone and all(len(text) > 0 for text in alone)
    assert tokenizer.padding_side == "right"  # put back: training pads on the right

    def batch(rows):
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [tokenizer.pad_token_id] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        labels = ids.masked_fill(mask == 0, -100)
        labels[:, :2] = -100  # the first two tokens are the prompt
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    rows = [tokenizer(text)["input_ids"] for text in ("k1=ab", "k22=abcabc", "k=a b c d e f")]
    with torch.no_grad():
        together_loss = backend_mod._row_losses(model, batch(rows))
        alone_loss = [backend_mod._row_losses(model, batch([r]))[0] for r in rows]
    assert together_loss == pytest.approx(alone_loss, rel=1e-5)


def test_a_prepared_file_with_columns_beside_prompt_and_completion_still_trains(tmp_path):
    # The hub's prepare step writes `conversation`, `turn`, `last_turn` and kept source fields on every row.
    # The loader reads the two columns the spec names; the run trains and is scored as on a two-column file.
    from sakura.atelier.data import materialise
    from sakura.atelier.spec import load_spec

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps({
        **row, "conversation": f"c{i // 3}", "turn": i % 3, "last_turn": i % 3 == 2,
        "exp_cal_state": {"events": [{"title": "sync", "start": f"{9 + i}:00"}]}}) + "\n" for i, row in enumerate(ROWS)))
    plain = tmp_path / "plain.jsonl"
    plain.write_text("".join(json.dumps(row) + "\n" for row in ROWS))

    def run(path, name):
        import torch

        hp = _hp(tiny_llama_dir(tmp_path / name / "base", ALPHABET), epochs=1, lr_schedule="cosine", warmup_ratio=0.2)
        spec = load_spec(f"""
atelier: 1
name: prepared
task: text_generation
data: {{uri: "file://{path}", format: jsonl, split: {{train: "0:6", validation: "6:"}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: hf-causal-lm
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 7
""")
        data = materialise(spec, tmp_path / name / "data")
        from sakura.atelier.registry import resolve

        metrics = _Metrics()
        torch.manual_seed(0)
        result = backend_mod.BACKEND.train(resolve(spec.model_copy(update={"hyperparameters": hp})), data,
                                           RunDir(tmp_path / name / "run").ensure(), metrics, None)
        return data, result, metrics

    data, result, metrics = run(prepared, "prepared")
    assert len(data.train) == 6 and all(set(row) == {"prompt", "completion"} for row in [*data.train, *data.validation])
    assert set(result.metrics) == {"loss", "exact_match"} and len(metrics.series("lr", "train")) == 3
    _, same, same_metrics = run(plain, "plain")
    assert metrics.rows == same_metrics.rows and result.metrics == same.metrics  # the extra columns change nothing


# ------------------------------------------------------------------ metrics, generation length, baseline

JSON_ROWS = [{"prompt": f"item {'abcd'[i % 4]}:", "completion": json.dumps({"t": "x", "k": "abcd"[i % 4]}, separators=(",", ":"))}
             for i in range(56)]  # completions of 17 characters, so 17 tokens: no 8-token answer can be right


def test_a_json_task_is_scored_on_json_at_full_length_against_the_untrained_model(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    train, validation = JSON_ROWS[:48], JSON_ROWS[48:]
    hp = _hp(base, init="random", init_config={"num_hidden_layers": 1, "hidden_size": 48, "intermediate_size": 96},
             epochs=12, batch_size=8, learning_rate=1e-2, metrics="auto", eval_max_new_tokens="auto", baseline=True,
             keep_best=True, eval_every=0.5, eval_max_rows=6, gen_eval_rows=2, final_gen_rows=6, eval_batch_size=4)
    result, metrics, out = _train(tmp_path, hp, train, validation)

    names = ["json_field_f1", "json_match", "json_valid"]  # chosen from the targets: every one is a JSON object
    assert list(result.metrics) == ["loss", *names]
    # it writes JSON, and gets at least the field every target shares right; a field is credit a whole match is not
    assert result.metrics["json_valid"] > 0.5, result.metrics
    assert result.metrics["json_field_f1"] > 0.25 and result.metrics["json_field_f1"] >= result.metrics["json_match"]
    assert result.report["generation_eval"] == {"metric": "json_field_f1", "metrics": names, "max_new_tokens": 17 + 8,
                                                "rows": 6, "loss_rows": 8, "rows_scored": 6}
    baseline = result.report["baseline"]  # the random weights, before a single step, on the same 6 rows
    assert list(baseline) == ["loss", *names, "n", "n_scored"] and baseline["n"] == baseline["n_scored"] == 6
    assert baseline["json_field_f1"] == 0.0 and baseline["json_match"] == 0.0
    assert baseline["loss"] > 2 * result.metrics["loss"]
    # every evaluation logs each metric: 3 optimiser steps between two of them (0.5 of the 6 steps of an epoch)
    assert [step for step, _ in metrics.series("json_field_f1", "validation")][:4] == [3, 6, 9, 12]
    assert all(set(r["validation"]) == {"loss", *names} for r in result.epoch_metrics)
    pre = json.loads((out.artifacts / "preprocess.json").read_text())["eval"]
    assert {k: pre[k] for k in ("metric", "metrics", "max_new_tokens", "exact_match_rule", "batch_size")} == {
        "metric": "json_field_f1", "metrics": names, "max_new_tokens": 25, "exact_match_rule": "whole", "batch_size": 4}
    assert len(pre["rows"]) == 6 and pre["rows"] == sorted(set(pre["rows"])) and max(pre["rows"]) < 8

    # the delivered model, reloaded, re-scored on the same rows at the same length with the same metric
    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    backend_mod._LOADED.clear()
    ev = backend_mod.BACKEND.runtime_eval(artifact, _data(train, validation))
    assert (ev.runtime, ev.metric, ev.n) == ("transformers", "json_field_f1", 6)
    assert ev.value == pytest.approx(result.metrics["json_field_f1"], abs=1e-6)

    # the same data without the keys: 8 generated tokens against 17-character targets, 0 by construction
    before, _, _ = _train(tmp_path, _hp(base, epochs=1, batch_size=8), train, validation, name="before")
    assert list(before.metrics) == ["loss", "exact_match"] and before.metrics["exact_match"] == 0.0


def test_the_baseline_is_the_base_model_before_the_adapter_on_the_final_rows(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "wrap-up-requested").write_text("")  # stop before the first step
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), baseline=True, metrics=["token_f1", "rouge_l", "exact_match"],
             eval_max_new_tokens=12, final_gen_rows=3, eval_batch_size=2)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS)
    baseline = result.report["baseline"]
    assert list(baseline) == ["loss", "token_f1", "rouge_l", "exact_match", "n", "n_scored"] and baseline["n"] == 3
    assert result.stopped_early["step"] == 0 and list(result.metrics) == ["loss", "token_f1", "rouge_l", "exact_match"]
    # a LoRA starts as the identity (its B matrix is zero): a run stopped before its first step scores what the
    # baseline scored, on the same rows
    assert {k: baseline[k] for k in result.metrics} == pytest.approx(result.metrics, rel=1e-5, abs=1e-7)
    assert result.report["generation_eval"]["max_new_tokens"] == 12 and result.report["generation_eval"]["rows"] == 3


def test_a_selection_metric_the_run_does_not_score_is_refused_before_training(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), metrics="auto", selection_metric="json_field_f1")
    with pytest.raises(ValueError, match=r"metrics: auto chose \['exact_match'\]"):  # short one-line targets
        _train(tmp_path, hp, ROWS, ROWS[:4])


def test_the_best_checkpoint_can_be_chosen_on_a_generation_metric(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), metrics=["token_f1"], selection_metric="token_f1",
             eval_max_new_tokens=6, eval_every=0.4, keep_best=True, early_stop_patience=2, eval_batch_size=4)
    result, metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    scores = metrics.series("token_f1", "validation")
    stop = result.early_stop
    assert (stop["metric"], stop["mode"]) == ("token_f1", "max")
    best = max(range(len(scores)), key=lambda i: scores[i][1])  # the first of equal values: a tie is no improvement
    assert (stop["best_step"], stop["best_value"], stop["kept_step"]) == (scores[best][0], round(scores[best][1], 6), scores[best][0])


# ------------------------------------------------------------------ replies cut by max_seq_length

#: With max_seq_length 12 (one token per character): "k0=" + "ab" + the end token is 6 tokens, fits.
FITS = {"prompt": "k0=", "completion": "ab"}
CUT = {"prompt": "x" * 10, "completion": "aaa"}  # 10 + 3 + 1 = 14 tokens: the reply keeps 2 of its 4 tokens
NO_TARGET = {"prompt": "y" * 20, "completion": "ab"}  # the prompt alone fills the window: no reply token left


def test_rows_cut_by_the_window_are_counted_and_those_without_a_reply_are_left_out(tmp_path):
    import math

    train = [FITS] * 6 + [CUT, NO_TARGET, NO_TARGET]
    validation = [FITS] * 3 + [CUT, NO_TARGET]
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=2, batch_size=2, max_seq_length=12)
    result, metrics, _ = _train(tmp_path, hp, train, validation)

    assert result.report["truncation"] == {
        "max_seq_length": 12,
        "train": {"rows": 9, "rows_truncated": 3, "rows_without_target": 2, "rows_used": 7},
        "validation": {"rows": 5, "rows_truncated": 2, "rows_without_target": 1, "rows_used": 4}}
    losses = [v for _, v in metrics.series("loss", "train")]
    assert len(losses) == 2 * 4  # 7 usable rows in batches of 2: 4 batches an epoch; the two bare rows never ran
    assert all(math.isfinite(v) for v in losses)
    assert all(math.isfinite(v) for _, v in metrics.series("loss", "validation"))
    assert math.isfinite(result.metrics["loss"])
    assert result.report["training"]["steps_planned"] == result.report["training"]["steps_done"] == 8


def test_a_run_whose_rows_mostly_have_no_reply_left_stops_before_training(tmp_path):
    train = [FITS] * 3 + [NO_TARGET] * 5
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), max_seq_length=12)
    with pytest.raises(ValueError) as err:
        _train(tmp_path, hp, train, [FITS])
    text = str(err.value)
    assert "5 of the 8 training rows" in text and "max_seq_length 12" in text
    assert "raise max_seq_length" in text.lower() or "Raise max_seq_length" in text
    assert not (tmp_path / "run" / "checkpoint" / "merged").exists()  # nothing was trained or saved


def test_half_of_the_rows_without_a_reply_is_still_trained(tmp_path):
    train = [FITS] * 4 + [NO_TARGET] * 4
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1, max_seq_length=12)
    result, metrics, _ = _train(tmp_path, hp, train, [FITS])
    assert result.report["truncation"]["train"] == {"rows": 8, "rows_truncated": 4, "rows_without_target": 4, "rows_used": 4}
    assert len(metrics.series("loss", "train")) == 2


def test_nothing_cut_means_zero_counts(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    assert result.report["truncation"] == {
        "max_seq_length": 48,
        "train": {"rows": 9, "rows_truncated": 0, "rows_without_target": 0, "rows_used": 9},
        "validation": {"rows": 4, "rows_truncated": 0, "rows_without_target": 0, "rows_used": 4}}


# ------------------------------------------------------------------ a loss that is not a number stops the run

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_loss_stops_the_run_before_the_optimiser_steps(tmp_path, monkeypatch, bad):
    import peft
    import torch

    real_get, steps = peft.get_peft_model, []
    real_step = torch.optim.AdamW.step

    def counting(self, *a, **kw):
        steps.append(1)
        return real_step(self, *a, **kw)

    def poisoned(*a, **kw):
        model = real_get(*a, **kw)
        forward, calls = model.forward, []

        def wrapped(**batch):
            out = forward(**batch)
            calls.append(1)
            if len(calls) == 4:  # the third batch of the run, after the early pass on the longest row
                out.logits = out.logits * bad  # the loss is taken from the logits
            return out
        model.forward = wrapped
        return model

    monkeypatch.setattr(peft, "get_peft_model", poisoned)
    monkeypatch.setattr(torch.optim.AdamW, "step", counting)
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET))
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, hp, ROWS, ROWS[:4])
    text = str(err.value)
    assert "not a finite number" in text and "optimiser step 2" in text and "0.44 epochs" in text
    assert "learning rate 0.005" in text and "ower learning_rate" in text
    assert len(steps) == 2  # the two steps before it ran; the poisoned batch never reached the optimiser


# ------------------------------------------------------------------ what the hub sizes and prices a job from

def test_the_report_carries_tokens_steps_seconds_and_the_settings_the_run_used(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=2, grad_accum=2, precision="fp32")
    result, metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    t = result.report["training"]
    assert list(t) == ["train_tokens", "train_loop_seconds", "eval_seconds", "final_eval_seconds", "baseline_seconds",
                       "steps_done", "steps_planned", "precision", "weights_dtype", "adapter_dtype", "autocast",
                       "device", "batch_size", "grad_accum", "effective_batch_size", "oom_backoffs",
                       "oom_backoff_steps", "gradient_checkpointing", "eval_batch_tokens", "peak_memory_bytes"]
    assert t["gradient_checkpointing"] is False
    # a row is its 3-character prompt, its reply and the end token, one token per character; both epochs
    assert t["train_tokens"] == 2 * sum(3 + len(r["completion"]) + 1 for r in ROWS)
    assert (t["steps_done"], t["steps_planned"]) == (6, 6)  # 5 batches in groups of 2, 2 and 1, over 2 epochs
    assert (t["precision"], t["weights_dtype"], t["adapter_dtype"], t["autocast"]) == ("fp32", "float32", "float32", False)
    assert (t["batch_size"], t["grad_accum"], t["effective_batch_size"]) == (2, 2, 4)
    assert t["train_loop_seconds"] > 0 and t["baseline_seconds"] is None
    # without the keys, the validation at each epoch's end is the periodic evaluation and the last one is reused
    assert t["eval_seconds"] > 0 and t["final_eval_seconds"] == 0.0
    assert t["train_loop_seconds"] + t["eval_seconds"] + t["final_eval_seconds"] <= result.train_seconds + 0.005  # rounded


def test_the_baseline_and_the_final_evaluation_are_timed_apart(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1, baseline=True, eval_max_rows=2)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    t = result.report["training"]
    assert t["baseline_seconds"] > 0 and t["final_eval_seconds"] > 0  # 2 of the 4 rows were scored in the loop
    assert t["baseline_seconds"] + t["train_loop_seconds"] + t["eval_seconds"] + t["final_eval_seconds"] > 0
    assert t["steps_done"] == t["steps_planned"] == 5


# ------------------------------------------------------------------ re-scoring the exports on a bounded number of rows

@pytest.mark.parametrize("value, ok", [(None, True), (1, True), (50, True), (0, False), (-3, False), (2.5, False),
                                      (True, False), ("50", False)])
def test_runtime_eval_rows_is_a_whole_number_from_one_or_null(value, ok):
    hp = {"base_model": "x", "runtime_eval_rows": value}
    if ok:
        assert backend_mod.train_eval_options(hp)["runtime_eval_rows"] == value
    else:
        with pytest.raises(ValueError, match="runtime_eval_rows"):
            backend_mod.train_eval_options(hp)
    assert backend_mod.train_eval_options({"base_model": "x"})["runtime_eval_rows"] is None


def test_the_exports_are_re_scored_on_the_first_rows_of_the_final_evaluation_and_compared_on_those_rows(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=2, metrics=["token_f1", "exact_match"], eval_max_new_tokens=6, final_gen_rows=6,
             eval_batch_size=2, runtime_eval_rows=3)
    train, validation = ROWS, [{"prompt": f"k{i}=", "completion": "ab"[i % 2] * (1 + i % 3)} for i in range(8)]
    result, _, out = _train(tmp_path, hp, train, validation)

    pre = json.loads((out.artifacts / "preprocess.json").read_text())["eval"]
    final = backend_mod._evaluation_rows(backend_mod._training_options(hp), 8, 7)[2]  # the seeded order, 6 rows
    assert len(final) == 6 and pre["rows"] == sorted(final[:3])  # the first 3 of it, so the subset nests
    assert pre["runtime_eval_rows"] == 3 and pre["train_n"] == 3
    # the training-side score on those 3 rows: what an export of the same weights must give on them
    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    backend_mod._LOADED.clear()
    ev = backend_mod.BACKEND.runtime_eval(artifact, _data(train, validation))
    assert ev.n == 3 and ev.train_n == 3 and ev.metric == "token_f1"
    assert ev.train_value == pytest.approx(pre["train_value"]) and ev.value == pytest.approx(ev.train_value, abs=1e-6)
    # a value over the 6 rows is a different number the run reports as its own metric
    assert result.report["generation_eval"]["rows"] == 6


def test_without_the_key_or_with_a_number_past_the_final_rows_the_export_is_scored_as_before(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    for name, over in (("absent", {}), ("past", {"runtime_eval_rows": 99})):
        hp = _hp(base, epochs=1, metrics=["token_f1"], eval_max_new_tokens=6, final_gen_rows=4, **over)
        result, _, out = _train(tmp_path, hp, ROWS, ROWS[:6], name=name)
        pre = json.loads((out.artifacts / "preprocess.json").read_text())["eval"]
        assert not {"runtime_eval_rows", "train_value", "train_n"} & set(pre)
        artifact = backend_mod.BACKEND.export(result, "safetensors", out)
        backend_mod._LOADED.clear()
        ev = backend_mod.BACKEND.runtime_eval(artifact, _data(ROWS, ROWS[:6]))
        assert ev.n == 4 and ev.train_value is None and ev.train_n is None


# ------------------------------------------------------------------ memory: bounded batches and out-of-memory back-off

def _oom_above(monkeypatch, rows, message="CUDA out of memory. Tried to allocate 3.18 GiB"):
    """The base model raises ``torch.OutOfMemoryError`` whenever it is called with more than ``rows`` rows."""
    import torch

    real = backend_mod._load_base_model

    def load(*a, **kw):
        model = real(*a, **kw)

        def hook(_module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None and ids.shape[0] > rows:
                raise torch.OutOfMemoryError(message)

        model.register_forward_pre_hook(hook, with_kwargs=True)
        return model

    monkeypatch.setattr(backend_mod, "_load_base_model", load)


def test_a_batch_holds_at_most_the_rows_and_the_tokens_it_is_bounded_by():
    lengths = {0: 100, 1: 90, 2: 20, 3: 10, 4: 10, 5: 5}
    batches = backend_mod._token_batches(list(lengths), lengths.__getitem__, 4, 200)
    assert sorted(j for b in batches for j in b) == sorted(lengths)  # every row once
    assert batches[0] == [0, 1]  # long rows, a few at a time (2 x 100 tokens)
    for b in batches:
        assert len(b) <= 4 and len(b) * max(lengths[j] for j in b) <= 200
    assert backend_mod._token_batches([7], lambda j: 5000, 4, 200) == [[7]]  # one row is always a batch
    assert len(backend_mod._token_batches(list(lengths), lengths.__getitem__, 4, None)[0]) == 4  # no token bound


def test_an_out_of_memory_is_told_from_any_other_error():
    import torch

    assert backend_mod._is_oom(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert backend_mod._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))  # older torch
    assert not backend_mod._is_oom(RuntimeError("shape mismatch"))
    assert not backend_mod._is_oom(ValueError("CUDA out of memory"))


def test_generation_halves_a_batch_that_does_not_fit_and_returns_the_same_texts(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32).eval()
    prompts = ["k1=", "a longer prompt: {", "x", "the longest prompt of the four, by far="]
    seen = []

    def hook(_m, args, kwargs):
        seen.append(int(kwargs["input_ids"].shape[0]))
        if seen[-1] > 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    model.register_forward_pre_hook(hook, with_kwargs=True)
    resources = backend_mod._Resources()
    with torch.no_grad():
        got = backend_mod._generate_bounded(model, tokenizer, prompts, 6, "cpu", max_rows=4, max_len=48,
                                            resources=resources)
        alone = [backend_mod._generate(model, tokenizer, [p], 6, "cpu")[0] for p in prompts]
    assert got == alone
    assert resources.backoffs == [{"phase": None, "step": 0, "batch_rows": 4, "action": "halve"}]
    assert max(seen[1:]) <= 2  # after the failure no batch is over 2 rows


def test_out_of_memory_above_two_rows_is_survived_by_the_baseline_the_evaluations_and_the_training(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, batch_size=4, grad_accum=1, eval_batch_size=8, baseline=True, lora_dropout=0.0)
    plain, plain_metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="plain")
    assert plain.report["training"]["oom_backoffs"] == 0 and plain.report["training"]["oom_backoff_steps"] == []

    _oom_above(monkeypatch, 2)
    result, metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="tight")
    t, p = result.report["training"], plain.report["training"]
    # training ran at half the batch and double the accumulation: the same rows per optimiser step and steps
    assert (t["batch_size"], t["grad_accum"], t["effective_batch_size"]) == (2, 2, 4)
    assert (p["batch_size"], p["grad_accum"]) == (4, 1)
    assert (t["steps_done"], t["steps_planned"]) == (p["steps_done"], p["steps_planned"]) == (3, 3)
    assert t["train_tokens"] == p["train_tokens"]
    # every back-off is counted with its phase and step
    phases = {b["phase"] for b in t["oom_backoff_steps"]}
    assert phases == {"baseline", "training", "evaluation"} and t["oom_backoffs"] == len(t["oom_backoff_steps"])
    assert [b for b in t["oom_backoff_steps"] if b["phase"] == "training"] == [
        {"phase": "training", "step": 0, "batch_rows": 4, "action": "halve"}]
    assert t["eval_batch_tokens"] == 4 * 48
    assert set(t["peak_memory_bytes"]) == {"baseline", "training", "evaluation"}
    # the baseline and the evaluations are the same numbers with smaller batches
    assert result.report["baseline"]["loss"] == pytest.approx(plain.report["baseline"]["loss"], rel=1e-4)
    assert result.report["baseline"]["exact_match"] == plain.report["baseline"]["exact_match"]
    assert [s for s, _ in metrics.series("loss", "validation")] == [s for s, _ in plain_metrics.series("loss", "validation")]
    assert result.metrics["loss"] == pytest.approx(plain.metrics["loss"], rel=1e-2)


@pytest.mark.parametrize("baseline", [False, True])
def test_a_row_that_does_not_fit_even_alone_stops_the_run_with_the_reason_in_words(tmp_path, monkeypatch, baseline):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    _oom_above(monkeypatch, 0)
    hp = _hp(base, epochs=1, batch_size=2, baseline=baseline)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, hp, ROWS, ROWS[:4])
    text = str(err.value)
    # the longest training row is tried first, so the run stops there, before any baseline
    assert "a training batch of one row" in text and "tokens does not fit in the memory of this card" in text
    assert "max_seq_length 48" in text and "Lower max_seq_length" in text and "card with more memory" in text


def test_validation_rows_that_do_not_fit_even_alone_stop_the_run_with_the_reason_in_words(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    _forwards(monkeypatch, lambda grad, n_grad, on: not grad)  # every forward without gradients runs out
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, _hp(base, epochs=1, batch_size=2, baseline=True), ROWS, ROWS[:4])
    assert "the shortest of the validation rows" in str(err.value)
    assert "does not fit in the memory of this card" in str(err.value)


# ------------------------------------------------------------------ gradient checkpointing, the last back-off

def _ckpt_model(monkeypatch, *, fits_with_checkpointing=True, supported=True):
    """A base model whose training forward (gradients on) runs out of memory until checkpointing is enabled
    (never, with ``fits_with_checkpointing=False``); the calls that enable it are recorded in ``state``."""
    import torch

    state = {"on": False, "enabled": 0, "input_grads": 0, "first_forward_on": None}
    real = backend_mod._load_base_model

    def load(*a, **kw):
        model = real(*a, **kw)
        enable, inputs = model.gradient_checkpointing_enable, model.enable_input_require_grads

        def turn_on(*args, **kwargs):
            if not supported:
                raise ValueError("this model does not support gradient checkpointing")
            state["enabled"] += 1
            state["on"] = True
            return enable(*args, **kwargs)

        def input_grads(*args, **kwargs):
            state["input_grads"] += 1
            return inputs(*args, **kwargs)

        def hook(_module, args, kwargs):
            if state["first_forward_on"] is None and torch.is_grad_enabled():
                state["first_forward_on"] = state["on"]
            if torch.is_grad_enabled() and not (state["on"] and fits_with_checkpointing):
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.18 GiB")

        model.gradient_checkpointing_enable = turn_on
        model.enable_input_require_grads = input_grads
        model.register_forward_pre_hook(hook, with_kwargs=True)
        return model

    monkeypatch.setattr(backend_mod, "_load_base_model", load)
    return state


def _adam_steps(monkeypatch):
    """Records, at the first optimiser step, whether every trained parameter has a gradient, and whether the
    weights moved by the step."""
    import torch

    seen = {"steps": 0, "all_grads": None, "moved": None}
    real = torch.optim.AdamW.step

    def step(self, *a, **kw):
        params = [p for g in self.param_groups for p in g["params"]]
        if seen["steps"] == 0:
            before = [p.detach().clone() for p in params]
            seen["all_grads"] = all(p.grad is not None for p in params)
        out = real(self, *a, **kw)
        if seen["steps"] == 0:
            seen["moved"] = any(not torch.equal(b, p.detach()) for b, p in zip(before, params))
        seen["steps"] += 1
        return out

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    return seen


def test_one_row_out_of_memory_turns_checkpointing_on_and_the_run_goes_on(tmp_path, monkeypatch, capsys):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, batch_size=2, grad_accum=2, lora_dropout=0.0)
    plain, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="plain")
    state = _ckpt_model(monkeypatch)
    seen = _adam_steps(monkeypatch)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="tight")
    t, p = result.report["training"], plain.report["training"]
    assert t["gradient_checkpointing"] is True and p["gradient_checkpointing"] is False
    assert state["enabled"] == 1 and state["input_grads"] >= 1
    assert (t["steps_done"], t["steps_planned"]) == (p["steps_done"], p["steps_planned"])
    assert t["train_tokens"] == p["train_tokens"] and t["effective_batch_size"] == p["effective_batch_size"]
    # the early pass on the longest row turned checkpointing on, so no batch was halved
    assert [(b["phase"], b["batch_rows"], b["action"], b["step"]) for b in t["oom_backoff_steps"]] == [
        ("training", 1, "gradient_checkpointing", 0)]
    assert t["oom_backoffs"] == 1 and (t["batch_size"], t["grad_accum"]) == (p["batch_size"], p["grad_accum"])
    assert seen["all_grads"] is True and seen["moved"] is True  # the adapter still learns under the checkpoints
    assert "gradient checkpointing" in capsys.readouterr().err
    # the same batches: only the order of float operations under the checkpoints differs
    assert result.metrics["loss"] == pytest.approx(plain.metrics["loss"], rel=1e-3)


def test_checkpointing_that_does_not_help_stops_the_run_and_says_it_was_tried(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    state = _ckpt_model(monkeypatch, fits_with_checkpointing=False)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, _hp(base, epochs=1), ROWS, ROWS[:4])
    text = str(err.value)
    assert state["enabled"] == 1
    assert "a training batch of one row" in text and "does not fit in the memory of this card" in text
    assert "even with gradient checkpointing on" in text and "Lower max_seq_length" in text


def test_a_model_that_cannot_checkpoint_stops_the_run_and_says_so(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    _ckpt_model(monkeypatch, supported=False)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, _hp(base, epochs=1), ROWS, ROWS[:4])
    assert "does not support gradient checkpointing" in str(err.value)
    assert "a training batch of one row" in str(err.value)


def test_checkpointing_false_never_turns_it_on_and_true_turns_it_on_before_the_first_step(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    state = _ckpt_model(monkeypatch)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, _hp(base, epochs=1, gradient_checkpointing=False), ROWS, ROWS[:4], name="off")
    assert state["enabled"] == 0 and "a training batch of one row" in str(err.value)
    assert "checkpointing" not in str(err.value)

    state.update(on=False, enabled=0, input_grads=0, first_forward_on=None)
    result, _, _ = _train(tmp_path, _hp(base, epochs=1, gradient_checkpointing=True), ROWS, ROWS[:4], name="on")
    t = result.report["training"]
    assert state["first_forward_on"] is True and state["enabled"] == 1 and state["input_grads"] >= 1
    assert t["gradient_checkpointing"] is True and t["oom_backoffs"] == 0 and t["oom_backoff_steps"] == []


def test_the_setting_is_one_of_auto_true_false():
    for good, want in (({}, "auto"), ({"gradient_checkpointing": "auto"}, "auto"), ({"gradient_checkpointing": True}, "true"),
                       ({"gradient_checkpointing": False}, "false")):
        assert backend_mod._training_options({"max_seq_length": 8, **good}).gradient_checkpointing == want
    with pytest.raises(ValueError, match="gradient_checkpointing must be"):
        backend_mod._training_options({"gradient_checkpointing": "sometimes"})
    from sakura.atelier.recipe import check_recipe

    with pytest.raises(ValueError, match="gradient_checkpointing must be"):  # refused before a run
        check_recipe("text_generation", {"gradient_checkpointing": "sometimes"}, "peft")


def test_full_training_follows_the_same_rule(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    real = backend_mod._random_causal_lm
    state = {"on": False}

    def random_lm(*a, **kw):
        import torch

        model = real(*a, **kw)
        enable = model.gradient_checkpointing_enable

        def turn_on(*args, **kwargs):
            state["on"] = True
            return enable(*args, **kwargs)

        def hook(_module, args, kwargs):
            if torch.is_grad_enabled() and not state["on"]:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.18 GiB")

        model.gradient_checkpointing_enable = turn_on
        model.register_forward_pre_hook(hook, with_kwargs=True)
        return model

    monkeypatch.setattr(backend_mod, "_random_causal_lm", random_lm)
    hp = _hp(base, epochs=1, init="random", init_config={"num_hidden_layers": 1, "hidden_size": 16,
                                                         "intermediate_size": 32, "num_attention_heads": 2,
                                                         "num_key_value_heads": 2})
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    assert result.report["training"]["gradient_checkpointing"] is True


def _forwards(monkeypatch, oom):
    """Records every forward of the base model as ``"grad"`` (training) or ``"eval"``; ``oom(grad, n_grad, on)``
    decides whether one runs out of memory (``n_grad``: training forwards so far, this one included; ``on``:
    gradient checkpointing). Returns the record."""
    import torch

    seen: list[str] = []
    real = backend_mod._load_base_model

    def load(*a, **kw):
        model = real(*a, **kw)

        def hook(module, _args, _kwargs):
            grad = torch.is_grad_enabled() and module.training
            seen.append("grad" if grad else "eval")
            if oom(grad, seen.count("grad"), module.is_gradient_checkpointing):
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.18 GiB")

        model.register_forward_pre_hook(hook, with_kwargs=True)
        return model

    monkeypatch.setattr(backend_mod, "_load_base_model", load)
    return seen


def test_the_longest_row_is_tried_before_the_baseline_and_checkpointing_is_on_before_it(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, batch_size=2, baseline=True, lora_dropout=0.0)
    seen = _forwards(monkeypatch, lambda grad, n_grad, on: grad and not on)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    t = result.report["training"]
    # out of memory, checkpointing on, the same row again: both before the baseline's first forward
    assert seen[:3] == ["grad", "grad", "eval"]
    assert t["gradient_checkpointing"] is True and t["oom_backoffs"] == 1
    assert result.report["baseline"]["n"] == 4


def test_a_row_that_never_fits_stops_the_run_before_the_baseline(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    seen = _forwards(monkeypatch, lambda grad, n_grad, on: grad)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, _hp(base, epochs=1, baseline=True), ROWS, ROWS[:4])
    assert seen == ["grad", "grad"]  # the row, then the row under checkpointing; no baseline forward
    assert "a training batch of one row" in str(err.value) and "even with gradient checkpointing on" in str(err.value)


def test_a_batch_of_one_row_that_runs_out_of_memory_later_still_gets_checkpointing(tmp_path, monkeypatch):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, batch_size=1, grad_accum=2, lora_dropout=0.0)
    plain, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="plain")
    _forwards(monkeypatch, lambda grad, n_grad, on: grad and n_grad > 1 and not on)  # the early pass fits
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="late")
    t, p = result.report["training"], plain.report["training"]
    assert t["gradient_checkpointing"] is True
    assert [(b["batch_rows"], b["action"]) for b in t["oom_backoff_steps"]] == [(1, "gradient_checkpointing")]
    assert (t["steps_done"], t["train_tokens"]) == (p["steps_done"], p["train_tokens"])
    assert result.metrics["loss"] == pytest.approx(plain.metrics["loss"], rel=1e-3)


def test_a_real_run_with_checkpointing_learns_as_the_same_run_without_it(tmp_path):
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=3, batch_size=2, grad_accum=2, lora_dropout=0.0)
    plain, plain_metrics, _ = _train(tmp_path, hp, ROWS, ROWS[:4], name="plain")
    checked, checked_metrics, _ = _train(tmp_path, {**hp, "gradient_checkpointing": True}, ROWS, ROWS[:4],
                                         name="checked")
    assert checked.report["training"]["gradient_checkpointing"] is True
    a, b = plain_metrics.series("loss", "validation"), checked_metrics.series("loss", "validation")
    assert [s for s, _ in a] == [s for s, _ in b]
    for (_, x), (_, y) in zip(a, b):
        assert y == pytest.approx(x, rel=1e-3, abs=1e-4)
    assert b[-1][1] < b[0][1]  # and it did learn
    assert checked.report["training"]["steps_done"] == plain.report["training"]["steps_done"]


# ------------------------------------------------------------------ an evaluation row that does not fit is skipped

LONG = {"prompt": "k9=" + "a" * 30, "completion": "ab"}


def _long_row_does_not_fit(monkeypatch, *, every_row=False):
    """Every forward without gradients over more than 20 tokens (the long row's loss and its generation) runs out
    of memory; with ``every_row`` every such forward does. ``raised`` counts them."""
    import torch

    raised = []
    real = backend_mod._load_base_model

    def load(*a, **kw):
        model = real(*a, **kw)

        def hook(_module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if not torch.is_grad_enabled() and ids is not None and (every_row or ids.shape[1] > 20):
                raised.append(int(ids.shape[1]))
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.18 GiB")

        model.register_forward_pre_hook(hook, with_kwargs=True)
        return model

    monkeypatch.setattr(backend_mod, "_load_base_model", load)
    return raised


def _skipping_run(tmp_path, monkeypatch, **over):
    raised = _long_row_does_not_fit(monkeypatch)
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=2, baseline=True, eval_batch_size=1,
             lora_dropout=0.0, eval_max_new_tokens=4, **over)
    result, _, out = _train(tmp_path, hp, ROWS, [*ROWS[:4], LONG])
    return result, out, raised


def test_a_row_too_long_to_evaluate_is_skipped_once_remembered_and_said_so(tmp_path, monkeypatch):
    result, out, raised = _skipping_run(tmp_path, monkeypatch)
    # the loss and the generation of that row ran out of memory once on each side (the starting weights, the tuned
    # ones), over a baseline, two periodic evaluations and the final one: the later passes did not pay again
    assert len(raised) == 4
    t, ge = result.report["training"], result.report["generation_eval"]
    skips = [(b["phase"], b["batch_rows"], b["action"]) for b in t["oom_backoff_steps"] if b["action"] == "skip"]
    assert sorted(skips) == [("baseline", 1, "skip")] * 2 + [("evaluation", 1, "skip")] * 2
    assert (ge["rows"], ge["rows_skipped_memory"], ge["loss_rows_skipped_memory"]) == (5, 1, 1)
    assert (ge["rows_scored"], result.report["baseline"]["n_scored"]) == (4, 4)  # the skipped row has no score
    assert result.report["baseline"]["rows_skipped_memory"] == 1
    assert result.report["baseline"]["loss_rows_skipped_memory"] == 1
    lines = [json.loads(line) for line in (out.artifacts / "predictions.jsonl").read_text().splitlines()]
    assert [r["index"] for r in lines] == [0, 1, 2, 3, 4]
    assert lines[4]["skipped"] == "memory" and lines[4]["baseline_skipped"] == "memory"
    assert not {"prediction", "scores", "baseline_prediction", "baseline_scores"} & set(lines[4])
    assert all("skipped" not in r and "prediction" in r for r in lines[:4])
    # the runtime re-scoring is over the rows both sides scored: the skipped row is not in it
    pre = json.loads((out.artifacts / "preprocess.json").read_text())
    assert pre["eval"]["rows"] == [0, 1, 2, 3]


def test_a_run_that_skips_nothing_reports_no_skip_keys(tmp_path):
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1, baseline=True, eval_batch_size=1)
    result, _, _ = _train(tmp_path, hp, ROWS, ROWS[:4])
    ge = result.report["generation_eval"]
    assert ge["rows_scored"] == ge["rows"] == 4 and result.report["baseline"]["n_scored"] == result.report["baseline"]["n"]
    assert not any("skipped" in key for key in ge | result.report["baseline"])
    assert all(b["action"] != "skip" for b in result.report["training"]["oom_backoff_steps"])


def test_when_every_evaluation_row_runs_out_of_memory_the_run_stops_with_the_reason_in_words(tmp_path, monkeypatch):
    _long_row_does_not_fit(monkeypatch, every_row=True)
    hp = _hp(tiny_llama_dir(tmp_path / "base", ALPHABET), epochs=1, baseline=True, eval_batch_size=1)
    with pytest.raises(RuntimeError) as err:
        _train(tmp_path, hp, ROWS, ROWS[:4])
    assert "does not fit in the memory of this card" in str(err.value) and "validation row" in str(err.value)


# ------------------------------------------------------------------ the default cap on the rows an export is re-scored on

def _rows(n):
    return [{"prompt": f"k{i}=", "completion": "ab"[i % 2] * (1 + i % 3)} for i in range(n)]


def _rescored(tmp_path, hp, validation, name="run"):
    result, _, out = _train(tmp_path, hp, ROWS, validation, name=name)
    pre = json.loads((out.artifacts / "preprocess.json").read_text())
    artifact = backend_mod.BACKEND.export(result, "safetensors", out)
    backend_mod._LOADED.clear()
    return result, pre, backend_mod.BACKEND.runtime_eval(artifact, _data(ROWS, validation))


def test_a_run_that_sets_no_row_count_re_scores_the_first_200_rows_of_the_seeded_order_like_for_like(tmp_path):
    assert backend_mod.DEFAULT_RUNTIME_EVAL_ROWS == 200
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, eval_max_new_tokens=4, eval_batch_size=64, lora_dropout=0.0)  # extended: a key is set
    validation = _rows(500)
    result, pre, ev = _rescored(tmp_path, hp, validation)
    order = backend_mod._eval_order(500, 7)
    assert pre["eval"]["rows"] == sorted(order[:200]) and pre["eval"]["runtime_eval_rows"] == 200
    assert pre["eval"]["runtime_eval_total"] == 500 and pre["eval"]["train_n"] == 200
    assert ev.n == 200 and ev.n_total == 500 and ev.train_n == 200
    assert ev.value == pytest.approx(ev.train_value, abs=1e-6)  # the export against the same rows, not all 500


def test_a_run_with_none_of_the_evaluation_keys_is_capped_too_and_compared_on_the_same_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod, "DEFAULT_RUNTIME_EVAL_ROWS", 4)
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    hp = _hp(base, epochs=1, lora_dropout=0.0)  # the legacy 8-token, first-line path
    result, pre, ev = _rescored(tmp_path, hp, _rows(9))
    assert pre["eval"]["rows"] == sorted(backend_mod._eval_order(9, 7)[:4]) and pre["eval"]["max_new_tokens"] == 8
    assert pre["eval"]["exact_match_rule"] == "first_line" and pre["eval"]["metric"] == "exact_match"
    assert (ev.n, ev.n_total, ev.train_n, ev.metric) == (4, 9, 4, "exact_match")
    assert ev.value == pytest.approx(ev.train_value, abs=1e-6)
    assert set(result.report) == {"truncation", "training"}  # nothing else changes for the legacy run


def test_a_validation_set_within_the_cap_is_re_scored_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod, "DEFAULT_RUNTIME_EVAL_ROWS", 9)
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    result, pre, ev = _rescored(tmp_path, _hp(base, epochs=1, lora_dropout=0.0), _rows(9))
    assert set(pre) == {"prompt_template", "base_model", "base_revision", "base_commit", "max_seq_length", "eos_token"}
    assert ev.n == 9 and ev.n_total is None and ev.train_value is None


def test_an_explicit_null_is_every_row_and_an_explicit_number_is_obeyed(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod, "DEFAULT_RUNTIME_EVAL_ROWS", 4)
    base = tiny_llama_dir(tmp_path / "base", ALPHABET)
    _, pre, ev = _rescored(tmp_path, _hp(base, epochs=1, lora_dropout=0.0, runtime_eval_rows=None), _rows(9), "null")
    assert pre["eval"]["rows"] is None and "runtime_eval_total" not in pre["eval"]
    assert ev.n == 9 and ev.n_total is None
    _, pre, ev = _rescored(tmp_path, _hp(base, epochs=1, lora_dropout=0.0, runtime_eval_rows=6), _rows(9), "six")
    assert len(pre["eval"]["rows"]) == 6 and (ev.n, ev.n_total) == (6, 9)  # above the default, still obeyed


def test_a_capped_check_without_a_training_side_score_on_its_rows_says_so_and_does_not_compare(tmp_path):
    from sakura.atelier.types import Artifact

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"x")
    artifact = Artifact(path=path, format="safetensors", sha256="x", bytes=1)
    scoring = {"metric": "exact_match", "max_new_tokens": 8, "exact_match_rule": "first_line", "rows": [0, 1],
               "runtime_eval_rows": 2, "runtime_eval_total": 5, "train_unavailable": "no per-row scores"}
    ev = backend_mod._rescore(artifact, _rows(5), "transformers", lambda p: "a", scoring=scoring)
    assert (ev.n, ev.n_total, ev.matches_training, ev.train_value) == (2, 5, None, None)
    assert "not compared" in ev.reason and "no per-row scores" in ev.reason
