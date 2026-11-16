"""tests/tiers.py: the lists name real files, and tier_of reads them as documented."""

from pathlib import Path

from tests.tiers import TIER2, TIER3, tier_of

ROOT = Path(__file__).resolve().parent.parent


def test_every_listed_path_exists():
    stale = [p for p in TIER2 if not (ROOT / p).exists()]
    stale += [n for n in TIER3 if not (ROOT / n.split("::", 1)[0]).is_file()]
    assert not stale, f"tests/tiers.py lists files that no longer exist: {stale}"


def test_tier_of():
    t3 = TIER3[0]
    assert tier_of(t3) == 3
    assert tier_of(t3.split("::", 1)[0] + "::test_some_other_test") == (2 if t3.split("::", 1)[0] in TIER2 else 1)
    assert tier_of("tests/atelier/test_decision_run.py::test_x") == 2
    assert tier_of("tests/test_tiers.py::test_tier_of") == 1
    p = "tests/services/test_leader_restart.py::test_leader_killed_mid_round_restarts_without_double_apply"
    assert tier_of(p + "[bcast-dn]") == 3
