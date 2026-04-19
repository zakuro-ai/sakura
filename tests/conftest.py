import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure the project root is on sys.path so `sakura` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock heavy dependencies that are unavailable in the test environment.
# These must be injected before any sakura module is imported.
#
# mpi4py / bson are no longer used (Zakuro dispatch replaces the MPI+Redis
# transport); they're not pulled in at import time anymore so no mocks needed.

# gnutools — provide a working RecNamespace so sakura.functional is testable
class _RecNamespace(SimpleNamespace):
    """Minimal stand-in for gnutools.utils.RecNamespace."""
    def __init__(self, d=None, **kwargs):
        if d is not None:
            kwargs.update(d)
        super().__init__(**kwargs)

_gnutools = MagicMock()
_gnutools_utils = MagicMock()
_gnutools_utils.RecNamespace = _RecNamespace
_gnutools_utils.RecDict = dict
sys.modules.setdefault("gnutools", _gnutools)
sys.modules.setdefault("gnutools.utils", _gnutools_utils)
sys.modules.setdefault("gnutools.fs", MagicMock())
