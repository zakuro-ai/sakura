"""Optional-dependency behavior (issue #71).

Sakura's core (dispatch / runtime / events / wire) must import without the
heavy ML stack. Training features are gated behind extras and raise a friendly
install hint when their dependency is missing.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


# --------------------------------------------------------------------------- #
# sakura._optional.load
# --------------------------------------------------------------------------- #


def test_load_returns_the_imported_module():
    from sakura._optional import load

    mod = load("json", extra="training")
    import json

    assert mod is json


def test_load_missing_dependency_raises_with_install_hint():
    from sakura._optional import load

    with pytest.raises(ModuleNotFoundError) as excinfo:
        load("a_module_that_does_not_exist_xyz", extra="training")

    msg = str(excinfo.value)
    assert "a_module_that_does_not_exist_xyz" in msg
    assert "sakura-ml[training]" in msg


# --------------------------------------------------------------------------- #
# Core import-safety without torch (run in a subprocess with torch blocked)
# --------------------------------------------------------------------------- #

_BLOCK_TORCH = (
    "import sys\n"
    "sys.modules['torch'] = None\n"
    "sys.modules['torch.distributed'] = None\n"
)


def _run_without_torch(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + body],
        capture_output=True,
        text=True,
    )


def test_import_sakura_without_torch():
    proc = _run_without_torch(
        "import sakura\n"
        "from sakura import SakuraRuntime, Telemetry\n"
        "from sakura.dispatch import RemoteDispatcher, LocalDispatcher\n"
        "rt = SakuraRuntime()\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_service_torch_path_without_torch_raises_install_hint():
    proc = _run_without_torch(
        "from sakura.services import Compile\n"
        "svc = Compile()\n"
        "try:\n"
        "    svc.on_train_begin(object())\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'sakura-ml[training]' in str(e), str(e)\n"
        "    print('RAISED_HINT')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RAISED_HINT" in proc.stdout


def test_sharded_optimizer_without_torch_raises_install_hint():
    proc = _run_without_torch(
        "from sakura import ShardedOptimizer\n"
        "try:\n"
        "    ShardedOptimizer(object())\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'sakura-ml[training]' in str(e), str(e)\n"
        "    print('RAISED_HINT')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RAISED_HINT" in proc.stdout


def test_activation_checkpoint_without_torch_raises_install_hint():
    proc = _run_without_torch(
        "from sakura.services import ActivationCheckpoint\n"
        "svc = ActivationCheckpoint(target_types=(object,))\n"
        "try:\n"
        "    svc.on_train_begin(object())\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'sakura-ml[training]' in str(e), str(e)\n"
        "    print('RAISED_HINT')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RAISED_HINT" in proc.stdout


def test_async_checkpoint_writer_without_torch_raises_install_hint():
    proc = _run_without_torch(
        "from sakura.services.async_checkpoint import _torch_save_writer\n"
        "try:\n"
        "    _torch_save_writer({}, '/tmp/x')\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'sakura-ml[training]' in str(e), str(e)\n"
        "    print('RAISED_HINT')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RAISED_HINT" in proc.stdout


def test_mixed_precision_without_torch_raises_install_hint():
    proc = _run_without_torch(
        "from sakura.services import MixedPrecision\n"
        "svc = MixedPrecision()\n"
        "try:\n"
        "    svc.on_train_begin(object())\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'sakura-ml[training]' in str(e), str(e)\n"
        "    print('RAISED_HINT')\n"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RAISED_HINT" in proc.stdout
