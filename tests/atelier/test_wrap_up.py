"""Graceful wrap-up (zak-forge#167): the signal, the engine's report, and every backend's step boundary.

No real training stack is needed: the loops are driven with tiny fakes, and a wrap-up request is raised the
way the runner raises it, by a file appearing while training is in progress."""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakura import atelier
from sakura.atelier import wrap_up
from sakura.atelier.types import MetricsSink, RunDir
from tests.atelier.fake_wrap_up_backend import LoopBackend

# --------------------------------------------------------------------------- the signal


def test_features_advertise_wrap_up():
    assert isinstance(atelier.FEATURES, frozenset) and "wrap_up" in atelier.FEATURES


def test_the_signal_is_a_file_in_out_unless_the_env_names_another(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    assert wrap_up.signal_path(tmp_path) == tmp_path / "wrap-up-requested"
    assert wrap_up.signal_path(None) is None and not wrap_up.requested(None)
    assert not wrap_up.requested(tmp_path)
    (tmp_path / "wrap-up-requested").write_text("")
    assert wrap_up.requested(tmp_path)  # an empty body is a valid request
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(wrap_up.ENV_VAR, str(elsewhere))
    assert wrap_up.signal_path(tmp_path) == elsewhere and not wrap_up.requested(tmp_path)  # the env wins
    elsewhere.write_text("{}")
    assert wrap_up.requested(tmp_path)


@pytest.mark.parametrize("body, reason", [
    ('{"reason": "budget"}', "budget"), ("", None), ("not json", None), ('["x"]', None), ('{"reason": 3}', None),
])
def test_the_reason_is_read_from_the_json_body(tmp_path, monkeypatch, body, reason):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    assert wrap_up.read_reason(tmp_path) is None  # no file
    (tmp_path / "wrap-up-requested").write_text(body)
    assert wrap_up.read_reason(tmp_path) == reason


def test_the_watch_is_sticky_and_records_where_it_first_saw_the_signal(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path)
    assert not watch.stop(0.5, 3) and watch.stopped_early is None and not watch.triggered
    (tmp_path / "wrap-up-requested").write_text('{"reason": "budget"}')
    assert watch.stop(1.25, 7) and watch.triggered
    assert watch.stopped_early == {"reason": "wrap_up", "epoch": 1.25, "step": 7}
    (tmp_path / "wrap-up-requested").unlink()
    assert watch.stop(9.0, 99) and watch.stopped_early["step"] == 7  # still stopped; first sighting kept
    assert watch.reason() is None  # the file is gone now


def test_the_watch_accepts_an_unknown_position(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    (tmp_path / "wrap-up-requested").write_text("")
    watch = wrap_up.WrapUpWatch(tmp_path)
    assert watch.stop() and watch.stopped_early == {"reason": "wrap_up", "epoch": None, "step": None}


# --------------------------------------------------------------------------- the engine's report


def _text_generation_spec(tmp_path: Path) -> Path:
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) + "\n" for i in range(20)))
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"""
atelier: 1
name: wrap
task: text_generation
data: {{uri: "file://{data}", format: jsonl, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: prompt, type: text}}]
  output: {{name: completion, type: text}}
model: lfm2-350m
hp: text_generation/lora-fast@1
export: [safetensors]
seed: 1
""")
    return spec


def _run(tmp_path, monkeypatch, **env):
    import sakura.atelier.__main__ as cli

    monkeypatch.setenv("SAKURA_ATELIER_BACKEND_MODULE_PEFT", "tests.atelier.fake_wrap_up_backend")
    monkeypatch.setenv("SAKURA_ATELIER_NOTEBOOK", "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "out"
    rc = cli.main(["run", str(_text_generation_spec(tmp_path)), "--out", str(out)])
    return rc, json.loads((out / "report.json").read_text()), out


def test_a_wrap_up_mid_run_still_exports_evaluates_and_reports_done(tmp_path, monkeypatch):
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    rc, report, out = _run(tmp_path, monkeypatch)
    assert rc == 0 and report["status"] == "done" and report["error"] is None
    assert report["stopped_early"] == {"reason": "wrap_up", "epoch": 0.75, "step": 3}
    assert LoopBackend.steps_run == 3  # stopped at the next step boundary, not run to the end
    assert report["artifacts"][0]["format"] == "safetensors" and (out / "artifacts" / "model.safetensors").is_file()
    assert report["runtime_eval"][0]["value"] == 0.5  # the evaluation stage ran on what training left


def test_a_run_nobody_wrapped_up_has_no_stopped_early(tmp_path, monkeypatch):
    class Quiet(LoopBackend):
        def train(self, spec, data, out, metrics, resume):
            result = super().train(spec, data, out, metrics, resume)
            (out.root / "wrap-up-requested").unlink(missing_ok=True)
            result.stopped_early = None
            return result

    import tests.atelier.fake_wrap_up_backend as holder

    monkeypatch.setattr(holder, "BACKEND", Quiet())
    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    rc, report, _ = _run(tmp_path, monkeypatch)
    assert rc == 0 and "stopped_early" not in report


def test_the_env_path_is_honoured_end_to_end(tmp_path, monkeypatch):
    signal = tmp_path / "signal-elsewhere"
    signal.write_text("{}")  # raised before the run even starts: no step is trained, the run still delivers
    rc, report, out = _run(tmp_path, monkeypatch, ATELIER_WRAP_UP_FILE=str(signal))
    assert rc == 0 and report["status"] == "done"
    assert report["stopped_early"] == {"reason": "wrap_up", "epoch": 0.0, "step": 0}
    assert LoopBackend.steps_run == 0 and (out / "artifacts" / "model.safetensors").is_file()


# --------------------------------------------------------------------------- ludwig

Metric = namedtuple("Metric", ["epoch", "step", "value"])


def _series(values):
    return [Metric(i, (i + 1) * 7, v) for i, v in enumerate(values)]


def test_ludwig_reports_wrap_up_as_the_reason_a_run_stopped_short():
    from sakura.atelier.backends.ludwig import early_stop_summary

    stats = {"validation": {"label": {"loss": _series([2.0, 1.5, 1.2])}}}
    kw = {"field": "label", "metric": "loss", "patience": 5, "max_epochs": 30}
    assert early_stop_summary(stats, **kw)["reason"] == "other"  # stopped short of both, unexplained
    summary = early_stop_summary(stats, wrap_up=True, **kw)
    assert summary["reason"] == "wrap_up" and summary["stopped_epoch"] == 3 and summary["best_epoch"] == 3


def test_ludwig_adapter_ends_training_at_the_next_batch_once_a_checkpoint_exists(tmp_path, monkeypatch):
    from sakura.adapters.ludwig import LudwigAdapter

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path)
    adapter = LudwigAdapter(MetricsSink(tmp_path / "m.jsonl"), output_feature="label", wrap_up=watch)
    trainer = SimpleNamespace(total_steps=100)
    tracker = SimpleNamespace(epoch=2, steps=40)
    adapter.on_save_best_checkpoint(trainer, tracker, str(tmp_path))  # an evaluation round saved one
    adapter.on_batch_end(trainer, tracker, str(tmp_path))
    assert trainer.total_steps == 100 and not adapter.should_early_stop(trainer, tracker, True)
    (tmp_path / "wrap-up-requested").write_text("{}")
    adapter.on_batch_end(trainer, SimpleNamespace(epoch=2, steps=41), str(tmp_path))
    assert trainer.total_steps == 41  # Ludwig's batch and epoch loops both stop at total_steps
    assert watch.stopped_early == {"reason": "wrap_up", "epoch": 2.0, "step": 41}
    assert adapter.should_early_stop(trainer, tracker, True)  # and an evaluation round ends the run too


class _FakeLudwigTrainer:
    """A Ludwig trainer in miniature, with the behaviour that matters (0.17.9 ``trainer.py``): 3 evaluation
    rounds of 4 steps that save the best checkpoint, a loop that ends when ``total_steps`` is capped, and a
    teardown that saves the best checkpoint at the last step when ``skip_all_evaluation`` is on."""

    def __init__(self, adapter, tmp_path):
        self.adapter, self.tmp_path = adapter, tmp_path
        self.total_steps = 12
        self.steps = 0
        self.skip_all_evaluation = False
        self.config = SimpleNamespace(skip_all_evaluation=False)
        self.best_written_at: list[int] = []
        self.evaluated_at: list[int] = []

    def run(self, request_at_step: int | None):
        a = self.adapter
        for step in range(1, 100):
            if step > self.total_steps:
                break
            if step == request_at_step:  # the runner writes the file while this batch trains
                (self.tmp_path / "wrap-up-requested").write_text('{"reason": "budget"}')
            self.steps = step
            tracker = SimpleNamespace(epoch=(step - 1) // 4 + (step - 1) % 4 / 4, steps=step)
            a.on_batch_end(self, tracker, str(self.tmp_path))
            if step % 4 == 0 and not self.skip_all_evaluation:  # evaluation boundary
                self.evaluated_at.append(step)
                self.best_written_at.append(step)
                a.on_save_best_checkpoint(self, tracker, str(self.tmp_path))
                if a.should_early_stop(self, tracker, True):
                    break
        if self.skip_all_evaluation:  # teardown: "evaluation was skipped, so save the current step as the best"
            self.best_written_at.append(self.steps)
        a.on_train_end(str(self.tmp_path))


def _fake_trainer_run(tmp_path, monkeypatch, request_at_step):
    from sakura.adapters.ludwig import LudwigAdapter

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    watch = wrap_up.WrapUpWatch(tmp_path)
    adapter = LudwigAdapter(MetricsSink(tmp_path / "m.jsonl"), output_feature="label", wrap_up=watch)
    trainer = _FakeLudwigTrainer(adapter, tmp_path)
    trainer.run(request_at_step)
    return trainer, watch


def test_ludwig_wrap_up_before_any_checkpoint_saves_one_at_that_batch_and_stops(tmp_path, monkeypatch):
    """Regression (staging job 61307af1): stopping with no checkpoint on disk made Ludwig raise "No checkpoint
    was saved". The adapter now arranges for the best checkpoint to be saved at the batch that saw the
    request, and stops there: no waiting for an evaluation round."""
    trainer, watch = _fake_trainer_run(tmp_path, monkeypatch, request_at_step=2)
    assert trainer.best_written_at == [2] and trainer.evaluated_at == []  # step 2, before the round at step 4
    assert trainer.total_steps == 2 and trainer.steps == 2  # stopped within the batch that saw it
    assert watch.stopped_early == {"reason": "wrap_up", "epoch": 0.25, "step": 2}  # a fractional epoch
    assert trainer.config.skip_all_evaluation is False  # the switch is put back before the model is saved


def test_ludwig_wrap_up_after_a_checkpoint_keeps_it_and_stops_at_the_next_step(tmp_path, monkeypatch):
    trainer, watch = _fake_trainer_run(tmp_path, monkeypatch, request_at_step=6)
    assert trainer.best_written_at == [4]  # only the evaluation round's checkpoint: the best is kept
    assert trainer.steps == 6 and watch.stopped_early["step"] == 6
    assert trainer.skip_all_evaluation is False  # nothing was flipped


def test_ludwig_trained_for_real_stops_within_a_batch_with_a_usable_checkpoint(tmp_path, monkeypatch):
    """The real Ludwig trainer (skipped where Ludwig is not installed): the request arrives mid-epoch, before
    any evaluation; the adapter's checkpoint is what ``model.train`` hands back."""
    pytest.importorskip("ludwig")
    import numpy as np
    import pandas as pd
    from ludwig.api import LudwigModel

    from sakura.adapters.ludwig import LudwigAdapter

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    df["y"] = (df.a + df.b > 0).astype(int).astype(str)
    watch = wrap_up.WrapUpWatch(tmp_path)

    class Requesting(LudwigAdapter):
        def on_batch_end(self, trainer, progress_tracker, save_path, sync_step=True):
            if progress_tracker.steps == 3:
                (tmp_path / "wrap-up-requested").write_text("{}")
            super().on_batch_end(trainer, progress_tracker, save_path, sync_step)

    adapter = Requesting(MetricsSink(tmp_path / "m.jsonl"), output_feature="y", wrap_up=watch)
    cfg = {"input_features": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}],
           "output_features": [{"name": "y", "type": "category"}],
           "trainer": {"epochs": 5, "batch_size": 48, "early_stop": -1}}  # 10 steps an epoch, 50 in all
    model = LudwigModel(cfg, callbacks=[adapter], logging_level=40)
    model.train(training_set=df[:480], validation_set=df[480:], output_directory=str(tmp_path / "out"),
                skip_save_processed_input=True)
    assert watch.stopped_early is not None and watch.stopped_early["step"] == 3
    assert list((tmp_path / "out").rglob("best.ckpt*"))  # the adapter's checkpoint, in Ludwig's own place
    pred, _ = model.predict(df[480:], skip_save_predictions=True, skip_save_eval_stats=True)
    assert len(pred) == 120  # the returned model works


def test_ludwig_wrap_up_with_evaluation_skipped_stops_at_once(tmp_path, monkeypatch):
    from sakura.adapters.ludwig import LudwigAdapter

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    (tmp_path / "wrap-up-requested").write_text("{}")
    adapter = LudwigAdapter(MetricsSink(tmp_path / "m.jsonl"), output_feature="label",
                            wrap_up=wrap_up.WrapUpWatch(tmp_path))
    trainer = SimpleNamespace(total_steps=100, skip_all_evaluation=True)  # Ludwig saves best when training ends
    adapter.on_batch_end(trainer, SimpleNamespace(epoch=0, steps=3), str(tmp_path))
    assert trainer.total_steps == 3


def test_ludwig_adapter_without_a_watch_never_stops_training(tmp_path):
    from sakura.adapters.ludwig import LudwigAdapter

    (tmp_path / "wrap-up-requested").write_text("{}")
    adapter = LudwigAdapter(MetricsSink(tmp_path / "m.jsonl"), output_feature="label")
    trainer = SimpleNamespace(total_steps=100)
    adapter.on_batch_end(trainer, SimpleNamespace(epoch=0, steps=1), str(tmp_path))
    assert trainer.total_steps == 100 and not adapter.should_early_stop(trainer, None, True)


# --------------------------------------------------------------------------- deepspeech


def test_deepspeech_loader_ends_the_epoch_and_flags_the_trainer(tmp_path, monkeypatch):
    from sakura.atelier.backends.deepspeech import _WrapUpLoader

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)

    class Loader:
        batch_size = 4

        def __len__(self):
            return 6

        def __iter__(self):
            return iter(range(6))

    trainer = SimpleNamespace(_stop=False)
    watch = wrap_up.WrapUpWatch(tmp_path)
    loader = _WrapUpLoader(Loader(), watch, [trainer])
    assert len(loader) == 6 and loader.batch_size == 4  # looks like the loader it wraps
    assert list(loader) == [0, 1, 2, 3, 4, 5] and not trainer._stop  # epoch 0: nobody asked

    seen = []
    for batch in loader:  # epoch 1: the request arrives after batch 2
        seen.append(batch)
        if batch == 2:
            (tmp_path / "wrap-up-requested").write_text("{}")
    assert seen == [0, 1, 2]  # batch 3 is the boundary that sees the request and is not trained
    assert trainer._stop  # the trainer's own stop flag: no further epoch
    assert watch.stopped_early == {"reason": "wrap_up", "epoch": 1.5, "step": 9}
    assert list(loader) == []  # and a later epoch trains nothing


def test_deepspeech_loader_trains_one_batch_even_when_wrap_up_came_first(tmp_path, monkeypatch):
    """The trainer raises on an epoch with no valid batch, so an immediate wrap-up still trains one step and
    ends the run with a scored, saved model rather than a crash."""
    from sakura.atelier.backends.deepspeech import _WrapUpLoader

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    (tmp_path / "wrap-up-requested").write_text("{}")
    trainer = SimpleNamespace(_stop=False)
    loader = _WrapUpLoader([0, 1, 2, 3], wrap_up.WrapUpWatch(tmp_path), [trainer])
    assert list(loader) == [0] and trainer._stop


# --------------------------------------------------------------------------- torch loops

torch = pytest.importorskip("torch")


def test_open_clip_stops_at_the_next_step_and_still_scores(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from sakura.atelier.backends import open_clip as oc
    from sakura.atelier.registry import resolve
    from sakura.atelier.spec import load_spec

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    out = RunDir(tmp_path / "run").ensure()
    spec = resolve(load_spec("""
atelier: 1
name: age
task: image_regression
data: {uri: "file:///x", format: csv, split: {validation_fraction: 0.2}}
features:
  inputs: [{name: img, type: image}]
  output: {name: age, type: number}
model: siglip2_b16_256
hp: image_regression/siglip2@1
export: [onnx]
overrides: {epochs: 3, batch_size: 4}
"""))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.head(self.visual(x))

    batches_seen: list[int] = []

    def loader(*_a, **_k):
        class DL:
            def __len__(self):
                return 5

            def __iter__(self):
                for i in range(5):
                    batches_seen.append(i)
                    if len(batches_seen) == 3:  # the request arrives while batch 2 is being fetched
                        (out.root / "wrap-up-requested").write_text('{"reason": "budget"}')
                    yield torch.zeros(4, 2), torch.zeros(4, 1)

        return DL()

    monkeypatch.setattr(oc, "_build", lambda model: (Net(), {"size": 2}))
    monkeypatch.setattr(oc, "_loader", loader)
    monkeypatch.setattr(oc, "_predict_torch", lambda *a, **k: np.zeros(4))
    data = SimpleNamespace(
        train=pd.DataFrame({"img": ["a"] * 4, "age": [1.0, 2.0, 3.0, 4.0]}),
        validation=pd.DataFrame({"img": ["a"] * 4, "age": [1.0, 2.0, 3.0, 4.0]}))
    sink = MetricsSink(out.metrics_path)
    result = oc.BACKEND.train(spec, data, out, sink, None)
    sink.close()

    assert result.stopped_early == {"reason": "wrap_up", "epoch": pytest.approx(0.0 + 2 / 5), "step": 2}
    assert len(batches_seen) == 3  # no batch after the one that saw the request was trained
    assert (out.checkpoint / "model.pt").is_file() and "mae" in result.metrics  # still delivers a model
    points = [json.loads(line) for line in out.metrics_path.read_text().splitlines()]
    assert sum(p["name"] == "loss" for p in points) == 2  # two steps trained, then the wrap-up


def test_decision_loop_drops_the_half_accumulated_step_and_stops(tmp_path, monkeypatch):
    from sakura.atelier.backends import decision_lm as D
    from sakura.atelier.decision import identity_order
    from tests.atelier.decision_fakes import CharTokenizer, FakeLM
    from tests.atelier.test_decision_lm import cfg, easy_instances

    monkeypatch.delenv(wrap_up.ENV_VAR, raising=False)
    tok = CharTokenizer()
    model = FakeLM(seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    watch = wrap_up.WrapUpWatch(tmp_path)
    steps = []
    real_step = opt.step

    def counting_step(*a, **k):
        steps.append(1)
        if len(steps) == 2:
            (tmp_path / "wrap-up-requested").write_text("{}")
        return real_step(*a, **k)

    monkeypatch.setattr(opt, "step", counting_step)
    with MetricsSink(tmp_path / "m.jsonl") as sink:
        summary = D.train_loop(model, tok, D.make_readout(tok), easy_instances(24), easy_instances(8),
                               cfg=cfg(epochs=6, batch_size=4), device="cpu", optimizer=opt, metrics=sink,
                               order_fn=lambda rng, k: identity_order(k), watch=watch)
    assert len(steps) == 2  # 24 examples / 4 = 6 steps an epoch, 6 epochs: it stopped at the step boundary
    assert summary["stopped_early"] == {"reason": "wrap_up", "epoch": pytest.approx(2 * 4 / 24), "step": 2}
    assert summary["log_loss"] is None  # stopped inside the first epoch: nothing was scored in the loop


def test_decision_loop_without_a_watch_is_unchanged(tmp_path):
    from sakura.atelier.backends import decision_lm as D
    from sakura.atelier.decision import identity_order
    from tests.atelier.decision_fakes import CharTokenizer, FakeLM
    from tests.atelier.test_decision_lm import cfg, easy_instances

    tok = CharTokenizer()
    model = FakeLM(seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    summary = D.train_loop(model, tok, D.make_readout(tok), easy_instances(8), easy_instances(4),
                           cfg=cfg(epochs=1, batch_size=4), device="cpu", optimizer=opt, metrics=None,
                           order_fn=lambda rng, k: identity_order(k))
    assert summary["stopped_early"] is None and summary["log_loss"] is not None
