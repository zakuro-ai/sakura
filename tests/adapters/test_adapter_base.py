"""Adapter ABC: holds the runtime + provides an emit() helper."""
from __future__ import annotations

import pytest

from sakura.adapters.base import Adapter
from sakura.events import OnEpochEnd
from sakura.runtime import SakuraRuntime


class TestAdapterBase:
    def test_emit_dispatches_event_to_runtime(self):
        rt = SakuraRuntime()
        seen = []
        from sakura.service import BaseService

        class Recorder(BaseService):
            name = "rec"
            priority = 10

            def on_epoch_end(self, event):
                seen.append(event.epoch)

        rt.install(Recorder())
        adapter = Adapter(rt)
        adapter.emit(OnEpochEnd(epoch=7, model="m", optimizer="o", metrics={},
                                 rank=0, world_size=1))
        assert seen == [7]

    def test_runtime_property(self):
        rt = SakuraRuntime()
        adapter = Adapter(rt)
        assert adapter.runtime is rt
