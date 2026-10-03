"""P0 ablation driver: config matrix, command construction, sanity gate, dry-run."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_p0_ablation.py"
_spec = importlib.util.spec_from_file_location("bench_p0_ablation", _SCRIPT)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)


def _rep(**kw):
    base = dict(tokens_per_sec=0.0, peak_gpu_mem_mb=0.0, gpu_util_mean_pct=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _good_reports():
    return {
        "baseline": _rep(tokens_per_sec=41000, peak_gpu_mem_mb=8200, gpu_util_mean_pct=72),
        "amp": _rep(tokens_per_sec=77000, peak_gpu_mem_mb=5200, gpu_util_mean_pct=85),
        "compile": _rep(tokens_per_sec=97000, peak_gpu_mem_mb=5300, gpu_util_mean_pct=91),
        "activation_ckpt": _rep(tokens_per_sec=90000, peak_gpu_mem_mb=3600, gpu_util_mean_pct=89),
    }


def test_config_matrix_is_the_four_p0_rows():
    assert [label for label, _ in drv.P0_CONFIGS] == [
        "baseline", "amp", "compile", "activation_ckpt"]
    ac_args = dict(drv.P0_CONFIGS)["activation_ckpt"]
    # AC row composes amp + compile + activation_checkpoint
    assert ac_args.count("--service") == 3
    assert "mixed_precision:fp16" in ac_args and "compile" in ac_args \
        and "activation_checkpoint" in ac_args


def test_build_run_cmd_targets_gpt2_cli():
    cmd, out = drv.build_run_cmd("/v/py", "amp", dict(drv.P0_CONFIGS)["amp"], "/o")
    assert cmd[:5] == ["/v/py", "-m", "sakura.bench", "run", "--workload"]
    assert "gpt2-124m" in cmd and "mixed_precision:fp16" in cmd
    assert out == "/o/gpt2-p0-amp.json" and cmd[-1] == out


def test_sizing_env_sets_gpt2_vars():
    e = drv.sizing_env(512, 8, 2, 128, 8, base={})
    assert e["SAKURA_GPT2_SEQ_LEN"] == "512"
    assert e["SAKURA_GPT2_BATCH_SIZE"] == "8"
    assert e["SAKURA_GPT2_EPOCHS"] == "2"
    assert e["SAKURA_GPT2_N_TRAIN_BATCHES"] == "128"
    assert e["SAKURA_GPT2_N_VAL_BATCHES"] == "8"


def test_check_sanity_passes_on_good_matrix():
    assert drv.check_sanity(_good_reports(), 0) == []


def test_check_sanity_flags_amp_not_faster():
    r = _good_reports()
    r["amp"] = _rep(tokens_per_sec=30000, peak_gpu_mem_mb=5200, gpu_util_mean_pct=85)
    assert any("[amp]" in m for m in drv.check_sanity(r, 0))


def test_check_sanity_flags_ac_not_smaller_than_compile():
    r = _good_reports()
    r["activation_ckpt"] = _rep(tokens_per_sec=90000, peak_gpu_mem_mb=9999, gpu_util_mean_pct=89)
    assert any("[activation_ckpt]" in m for m in drv.check_sanity(r, 0))


def test_check_sanity_flags_zero_util_and_orphans():
    r = _good_reports()
    r["compile"] = _rep(tokens_per_sec=97000, peak_gpu_mem_mb=5300, gpu_util_mean_pct=0.0)
    msgs = drv.check_sanity(r, 2)
    assert any("gpu_util_mean_pct" in m for m in msgs)
    assert any("[orphans]" in m for m in msgs)


def test_count_orphan_workers_no_workers_returns_zero():
    # argv-list pgrep (no shell) -> no self-match; no workers in the test env.
    assert drv.count_orphan_workers() == 0


def test_dry_run_prints_all_commands_without_executing(capsys, tmp_path):
    rc = drv.main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # driver-level pre-warm was removed; only the four P0 config labels appear
    for label in ("baseline", "amp", "compile", "activation_ckpt"):
        assert f"[{label}]" in out
    assert "[warmup]" not in out, "cross-process pre-warm was removed; should not appear"
    assert "gpt2-124m" in out
    assert "gpt2-p0-baseline.json" in out
    assert "activation_checkpoint" in out
    # dry-run writes no report files
    assert list(tmp_path.glob("*.json")) == []
