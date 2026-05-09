"""Service install order, priority sorting, dependency resolution, dispatch order."""
from __future__ import annotations

import pytest

from sakura.events import OnEpochEnd
from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _Recorder(BaseService):
    log: list[tuple[str, int]] = []

    def __init__(self, name: str, priority: int, requires: tuple = ()):
        self.name = name
        self.priority = priority
        self.requires = requires
        super().__init__()

    def on_epoch_end(self, event):
        type(self).log.append((self.name, event.epoch))


@pytest.fixture(autouse=True)
def _reset_recorder_log():
    _Recorder.log = []
    yield


class TestInstallOrder:
    def test_services_dispatch_in_priority_order(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="late", priority=80))
        rt.install(_Recorder(name="early", priority=10))
        rt.install(_Recorder(name="middle", priority=50))
        rt.dispatch(OnEpochEnd(epoch=1, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("early", 1), ("middle", 1), ("late", 1)]

    def test_install_order_breaks_priority_ties(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="first", priority=50))
        rt.install(_Recorder(name="second", priority=50))
        rt.install(_Recorder(name="third", priority=50))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("first", 0), ("second", 0), ("third", 0)]

    def test_uninstall_removes_service_from_dispatch(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20))
        rt.uninstall("a")
        rt.dispatch(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("b", 2)]

    def test_install_duplicate_name_raises(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="dup", priority=10))
        with pytest.raises(ValueError, match="already installed"):
            rt.install(_Recorder(name="dup", priority=20))

    def test_install_with_unmet_requires_raises(self):
        rt = SakuraRuntime()
        with pytest.raises(ValueError, match="requires.*missing"):
            rt.install(_Recorder(name="needs_a", priority=50, requires=("a",)))

    def test_install_with_satisfied_requires_succeeds(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20, requires=("a",)))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("a", 0), ("b", 0)]

    def test_uninstall_blocked_by_dependent(self):
        rt = SakuraRuntime()
        rt.install(_Recorder(name="a", priority=10))
        rt.install(_Recorder(name="b", priority=20, requires=("a",)))
        with pytest.raises(ValueError, match="depended on by"):
            rt.uninstall("a")

    def test_install_calls_on_install(self):
        installed = []

        class Installer(BaseService):
            name = "installer"
            priority = 50

            def on_install(self, runtime):
                installed.append(runtime)

            def on_event(self, event):
                pass

        rt = SakuraRuntime()
        s = Installer()
        rt.install(s)
        assert installed == [rt]

    def test_dispatch_before_install_is_safe_noop(self):
        rt = SakuraRuntime()
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))

    def test_service_exception_does_not_block_others(self):
        class Boom(BaseService):
            name = "boom"
            priority = 30

            def on_epoch_end(self, event):
                raise RuntimeError("intentional")

        rt = SakuraRuntime()
        rt.install(_Recorder(name="early", priority=10))
        rt.install(Boom())
        rt.install(_Recorder(name="late", priority=80))
        rt.dispatch(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                                rank=0, world_size=1))
        assert _Recorder.log == [("early", 0), ("late", 0)]
