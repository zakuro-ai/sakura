import pytest
import sys
import os

# Ensure the project root is on sys.path so `sakura` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# No module stubs are injected here: `import sakura` must work with only the
# declared runtime dependencies installed.
#
# mpi4py / bson are no longer used (Zakuro dispatch replaces the MPI+Redis
# transport); they're not pulled in at import time anymore so no mocks needed.
# gnutools-python was dropped as a dependency (#23); nothing in sakura/ uses it.


# ------------------------------------------------------------------ test tiers (tests/tiers.py)

def pytest_addoption(parser):
    parser.addoption("--tier", type=int, choices=(1, 2, 3), default=None,
                     help="run tiers 1..N (tests/tiers.py); default $SAKURA_TEST_TIER or 3 = everything")


def pytest_collection_modifyitems(config, items):
    from tests.tiers import tier_of

    tier = config.getoption("--tier") or int(os.environ.get("SAKURA_TEST_TIER") or 3)
    if tier >= 3:
        return
    keep, drop = [], []
    for item in items:
        (keep if tier_of(item.nodeid) <= tier else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@pytest.fixture(autouse=True)
def _no_licence_fetch(monkeypatch):
    """No test reaches Hugging Face for a base model's licence (sakura.atelier.licence);
    tests of the fetch itself inject their own `fetch`."""
    monkeypatch.setenv("SAKURA_ATELIER_LICENCE_FETCH", "0")
