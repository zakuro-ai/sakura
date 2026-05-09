"""SakuraRuntime start/shutdown/__enter__/__exit__ semantics."""
from __future__ import annotations

from sakura.runtime import SakuraRuntime
from sakura.service import BaseService


class _Hook(BaseService):
    name = "hook"
    priority = 50

    def __init__(self):
        super().__init__()
        self.started = False
        self.shutdown_called = False

    def on_runtime_start(self, runtime):
        self.started = True

    def on_runtime_shutdown(self, runtime):
        self.shutdown_called = True

    def on_event(self, event):
        pass


class TestLifecycle:
    def test_start_calls_on_runtime_start_for_each_service(self):
        rt = SakuraRuntime()
        h = _Hook()
        rt.install(h)
        rt.start()
        assert h.started is True

    def test_start_is_idempotent(self):
        rt = SakuraRuntime()
        rt.start()
        rt.start()

    def test_shutdown_calls_on_runtime_shutdown(self):
        rt = SakuraRuntime()
        h = _Hook()
        rt.install(h)
        rt.start()
        rt.shutdown()
        assert h.shutdown_called is True

    def test_shutdown_without_start_is_safe_noop(self):
        rt = SakuraRuntime()
        rt.shutdown()

    def test_context_manager_starts_and_shuts_down(self):
        h = _Hook()
        with SakuraRuntime() as rt:
            rt.install(h)
            assert h.started is True
        assert h.shutdown_called is True

    def test_context_manager_shuts_down_on_exception(self):
        h = _Hook()
        try:
            with SakuraRuntime() as rt:
                rt.install(h)
                raise RuntimeError("user code crashed")
        except RuntimeError:
            pass
        assert h.shutdown_called is True

    def test_history_accumulates_dispatched_event_records(self):
        from sakura.events import OnEpochEnd

        rt = SakuraRuntime()
        rt.dispatch(OnEpochEnd(epoch=1, model=None, optimizer=None, metrics={"a": 1.0},
                                rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={"a": 0.9},
                                rank=0, world_size=1))
        h = rt.history()
        assert len(h) >= 2
        assert all(isinstance(record, dict) for record in h)
        assert any(record.get("event") == "OnEpochEnd" for record in h)
