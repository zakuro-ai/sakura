"""Test tiers. `pytest --tier N` (or SAKURA_TEST_TIER=N) runs tiers 1..N; the default is 3 (everything).

- tier 1 -- fast unit tests: every commit, every local change. Anything not listed below.
- tier 2 -- integration: real subprocesses, timers, tiny model training. Every PR and push to master.
- tier 3 -- release: the slowest end-to-end runs (multi-rank, full training, import hygiene in a fresh
  interpreter). Release gate (publish.yml), nightly, workflow_dispatch, PRs into release/* or labelled ci:full.

Lists are by measured duration (6 xdist workers, OMP_NUM_THREADS=2, 2026-11-09): a test that takes >= 20 s is
tier 3; any other test in a file whose slow tests total >= 5 s is tier 2. Re-measure with
`pytest --durations=0 --durations-min=3` and move entries when a test's cost changes. GPU tests keep their own
`gpu` marker (hardware, not cost)."""

#: Exact node ids (a parametrised test's id without the [param] covers every param).
TIER3 = (
    "tests/atelier/notebook/test_build.py::test_every_fault_actually_fails_when_the_cell_runs",
    "tests/atelier/notebook/test_bundle_cli.py::test_a_failed_check_still_delivers_the_bundle_with_the_recorded_outputs",
    "tests/atelier/notebook/test_bundle_cli.py::test_run_report_carries_the_evaluation_block_and_the_onnx_provider",
    "tests/atelier/notebook/test_bundle_cli.py::test_run_writes_the_bundle_and_the_check_and_lists_the_artifact",
    "tests/atelier/test_backend_whisper.py::test_run_trains_exports_and_rescores",
    "tests/atelier/test_risk_sim.py::test_the_notebook_replays_the_run",
    "tests/atelier/test_smoke_runs.py::test_the_notebook_of_a_capped_synthetic_run_replays_it",
    "tests/atelier/test_backend_rl.py::test_the_notebook_of_a_reinforcement_learning_run_replays_it",
    "tests/atelier/test_backends_registry.py::test_importing_the_engine_imports_no_training_stack",
    "tests/atelier/test_backends_registry.py::test_loading_the_peft_backend_does_not_import_the_decision_torch_half_or_a_stack",
    "tests/bench/test_time_to_target.py::test_cli_time_to_target_reachable",
    "tests/bench/test_time_to_target.py::test_cli_time_to_target_unreachable",
    "tests/bench/test_workloads_smoke.py::test_mnist_workload_runs_one_epoch",
    "tests/dispatch/test_process_dispatcher.py::test_construction_does_not_block_on_a_large_initializer_payload",
    "tests/dispatch/test_process_dispatcher.py::test_dead_worker_fails_pending_futures",
    "tests/dispatch/test_process_dispatcher.py::test_initializer_failure_surfaces",
    "tests/dispatch/test_process_dispatcher.py::test_shutdown_is_idempotent_and_blocks_submit",
    "tests/dispatch/test_process_dispatcher.py::test_wait_ready_raises_if_worker_dies_during_startup",
    "tests/services/test_leader_restart.py::test_leader_killed_mid_round_restarts_without_double_apply[bcast-dn]",
    "tests/services/test_leader_restart.py::test_leader_killed_mid_round_restarts_without_double_apply[persist-dn]",
    "tests/services/test_leader_restart.py::test_leader_killed_mid_round_restarts_without_double_apply[persist-plain]",
    "tests/test_pkg_info.py::TestNoGnutools::test_import_sakura_with_gnutools_blocked",
    "tests/zero/test_sharded_optimizer.py::test_sharded_optimizer_2_ranks",
    "tests/zero/test_sharded_optimizer_multi_rank.py::TestZeRO1MultiRankCorrectness::test_multi_step_convergence_matches_reference",
    "tests/zero/test_sharded_optimizer_multi_rank.py::TestZeRO1MultiRankCorrectness::test_one_step_agrees_across_ranks_and_matches_reference",
)

#: Whole files (or directories, ending in /).
TIER2 = (
    "tests/atelier/notebook/test_agent.py",
    "tests/atelier/notebook/test_build.py",
    "tests/atelier/notebook/test_bundle_cli.py",
    "tests/atelier/test_backend_tabpfn.py",
    "tests/atelier/test_backend_whisper.py",
    "tests/atelier/test_backends_registry.py",
    "tests/atelier/test_decision_notebook.py",
    "tests/atelier/test_decision_peft.py",
    "tests/atelier/test_decision_run.py",
    "tests/atelier/test_classic_ml.py",  # 5 estimators x 2 tasks + ONNX merges, ~20 s on 4 workers (2026-11-09)
    "tests/atelier/test_hf_detection_any.py",  # tiny YOLOS: download + 1 training (2026-11-09)
    "tests/atelier/test_risk_sim.py",  # real numpy/scipy runs, a few seconds each (2026-11-11)
    "tests/atelier/test_hf_classification.py",  # tiny HF repos, downloads + 3 trainings (2026-11-09)
    "tests/atelier/test_smoke_runs.py",  # real capped runs on synthetic data, one process each (2026-11-14)
    "tests/atelier/test_init_random.py",  # two trainings from random weights, one process each (2026-11-14)
    "tests/atelier/test_distillation.py",  # a teacher and a student run for each of two backends (2026-11-14)
    "tests/atelier/test_dpo.py",  # two DPO runs on a tiny random Llama, one process each (2026-11-14)
    "tests/atelier/test_peft_training_eval.py",  # LoRA trainings on a tiny random Llama, a few seconds each (2026-11-15)
    "tests/atelier/test_onnx_int8.py",  # one training, two ONNX exports re-scored (2026-11-14)
    "tests/atelier/test_curation.py",  # two real curated runs, one process each (2026-11-14)
    "tests/atelier/test_backend_rl.py",  # small REINFORCE trainings on a CPU, one whole run in a process (2026-11-14)
    "tests/bench/test_cli.py",
    "tests/bench/test_time_to_target.py",
    "tests/bench/test_workloads_smoke.py",
    "tests/dispatch/test_process_dispatcher.py",
    "tests/services/test_compile.py",
    "tests/services/test_dynamic_membership.py",
    "tests/services/test_fed_report.py",
    "tests/services/test_federation.py",
    "tests/services/test_federation_region.py",
    "tests/services/test_fleet_join.py",
    "tests/services/test_leader_restart.py",
    "tests/services/test_link_probe.py",
    "tests/services/test_link_reprobe.py",
    "tests/services/test_sim_fed_halos.py",
    "tests/services/test_sparseloco_parity.py",
    "tests/test_pkg_info.py",
    "tests/zero/test_sharded_optimizer.py",
    "tests/zero/test_sharded_optimizer_multi_rank.py",
)


def tier_of(nodeid: str) -> int:
    base = nodeid.split("[", 1)[0]
    if base in TIER3 or nodeid in TIER3:
        return 3
    path = nodeid.split("::", 1)[0]
    if any(path == p or (p.endswith("/") and path.startswith(p)) for p in TIER2):
        return 2
    return 1
