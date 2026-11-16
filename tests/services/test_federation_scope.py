"""SAK-0017 scope advisory: the rule, its two measured fixtures, and the
first-round log on leader and worker (CPU-only loopback e2e)."""

from __future__ import annotations

import logging
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from sakura.services.federation import Leader  # noqa: E402
from sakura.services.federation.scope import (  # noqa: E402
    MIN_FLEET_SPEEDUP,
    MIN_SOLO_EPOCH_S,
    format_advisory,
    scope_advisory,
)
from tests.services.test_federation import (  # noqa: E402
    TinyNet,
    _start_leader,
    _worker_thread,
)

# E5 (docs/internal/REPORT.md): ResNet-50@224, per-worker in-fleet img/s = concurrent tuned
# solo x E5 retention (4090 1647.94 x 98.4%, 2080 Ti 744.34 x 95.6%,
# P620 29.75 x 98.9%). E5 ran on synthetic data; the dataset size is
# ImageNet-1k's, the regime "ResNet-50@224" stands for. Federation won (+43%).
E5_RESNET50_RATES = {"4090": 1621.6, "2080ti": 711.6, "p620": 29.4}
IMAGENET_1K = 1_281_167
# E6 (results/e6/*.log): CIFAR-10 / ResNet-18, img/s = sum(n_samples) over
# the WORKER_ROUND timestamps. The solo 4090 control beat the fleet by ~2 pp.
E6_RESNET18_RATES = {"2080ti": 4463.7, "4090": 4434.0, "p620": 355.9}
CIFAR10_TRAIN = 50_000


class TestRule:
    def test_e5_resnet50_is_ok_with_margin(self):
        adv = scope_advisory(E5_RESNET50_RATES, IMAGENET_1K)
        assert adv["verdict"] == "ok", adv
        assert adv["reasons"] == []
        assert adv["fastest"] == "4090"
        # measured fleet gain over its fastest member (+46%)
        assert adv["fleet_speedup"] == pytest.approx(1.457, abs=0.01)
        assert adv["fleet_speedup"] > MIN_FLEET_SPEEDUP
        # ~790 s per solo epoch: > 10x above the threshold
        assert adv["solo_epoch_s"] > 10 * MIN_SOLO_EPOCH_S
        assert adv["skew"] == pytest.approx(55.2, abs=0.1)

    def test_e6_resnet18_warns_with_margin(self):
        adv = scope_advisory(E6_RESNET18_RATES, CIFAR10_TRAIN)
        assert adv["verdict"] == "warn", adv
        assert adv["fastest"] == "2080ti"
        # the fleet DID add ~2x throughput: speedup alone does not predict E6
        assert adv["fleet_speedup"] > MIN_FLEET_SPEEDUP
        # ~11 s per solo epoch: > 5x below the threshold
        assert adv["solo_epoch_s"] < MIN_SOLO_EPOCH_S / 5
        assert len(adv["reasons"]) == 1 and "sweep all 50000" in adv["reasons"][0]

    def test_e6_with_the_solo_control_rate_also_warns(self):
        # the solo control ran at 6 054 img/s (no co-location tax)
        adv = scope_advisory({"4090": 6054.0, "2080ti": 4463.7, "p620": 355.9}, CIFAR10_TRAIN)
        assert adv["verdict"] == "warn"

    def test_data_scale_drives_the_verdict(self):
        # the E5 fleet on a CIFAR-sized set: one 4090 sweeps it in ~31 s
        assert scope_advisory(E5_RESNET50_RATES, CIFAR10_TRAIN)["verdict"] == "warn"

    def test_fleet_dominated_by_one_gpu_warns_without_dataset_size(self):
        adv = scope_advisory({"4090": 1648.0, "p620": 29.75})
        assert adv["verdict"] == "warn"
        assert adv["fleet_speedup"] < MIN_FLEET_SPEEDUP
        assert adv["solo_epoch_s"] is None

    def test_unknown_dataset_size_is_unchecked_not_ok(self):
        adv = scope_advisory(E6_RESNET18_RATES, None)
        assert adv["verdict"] == "unchecked"
        assert any("dataset size unknown" in r for r in adv["reasons"])

    def test_no_rates_is_unchecked(self):
        adv = scope_advisory({"a": 0.0}, 1000)
        assert adv["verdict"] == "unchecked"
        assert adv["rates"] == {}

    def test_format_mentions_verdict_and_assumptions(self):
        line = format_advisory(scope_advisory(E6_RESNET18_RATES, CIFAR10_TRAIN))
        assert line.startswith("federation scope advisory: WARN")
        assert "IID shards" in line
        # tolerant of a malformed advisory off the wire
        assert "NONE" in format_advisory({})


class TestFirstRoundAdvisory:
    def test_leader_and_workers_log_advisory_once(self, caplog):
        caplog.set_level(logging.INFO, logger="sakura.federation")
        # TinyNet trains thousands of samples/s on CPU: a 100-sample "dataset"
        # is swept in well under a second, so the advisory must warn.
        leader = Leader(
            model_fn=TinyNet,
            port=0,
            expected=["a", "b", "c"],
            codec="int8",
            gap_s=1.2,
            seed=42,
            max_rounds=2,
            dataset_size=100,
        )
        port = _start_leader(leader, warmup_s=1.0)
        stop = threading.Event()
        out: dict = {}
        ths = []
        for wid in ["a", "b", "c"]:
            t = threading.Thread(
                target=_worker_thread, args=(wid, port, stop, out), daemon=True
            )
            t.start()
            ths.append(t)
            time.sleep(0.1)
        t0 = time.monotonic()
        while leader.round < 2 and time.monotonic() - t0 < 60:
            time.sleep(0.25)
        # let the workers finish logging round 2 before stopping
        time.sleep(0.5)
        leader.stop.set()
        stop.set()
        for t in ths:
            t.join(15)
        try:
            leader._srv.close()
        except Exception:
            pass

        assert leader.round >= 2
        first, second = leader.stats[0], leader.stats[1]
        adv = first["scope_advisory"]
        assert adv["verdict"] == "warn", adv
        assert set(adv["rates"]) == {"a", "b", "c"}
        assert all(r > 0 for r in adv["rates"].values())
        assert "scope_advisory" not in second  # once, at the first round
        assert leader.scope_advisory == adv

        for wid in ("a", "b", "c"):
            fed = out[wid]["fed"]
            assert fed.scope_advisory == adv
            assert fed.records[0]["window_s"] > 0
            assert fed.records[0]["scope_advisory"] == "warn"
            assert all("scope_advisory" not in r for r in fed.records[1:])

        warns = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "scope advisory" in r.getMessage()
        ]
        assert len(warns) == 4  # leader + one per worker
        assert sum(m.startswith(("a ", "b ", "c ")) for m in warns) == 3
