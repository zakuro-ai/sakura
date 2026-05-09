"""Tests for the Service ABC and BaseService helper."""
from __future__ import annotations

import pytest

from sakura.events import OnEpochEnd, OnTrainBegin
from sakura.service import BaseService, Service


class TestServiceABC:
    def test_service_is_a_protocol(self):
        class _Duck:
            name = "duck"
            priority = 50
            requires: tuple[str, ...] = ()

            def on_install(self, runtime):
                pass

            def on_event(self, event):
                pass

        assert isinstance(_Duck(), Service)

    def test_base_service_provides_defaults(self):
        class MyService(BaseService):
            name = "my_service"
            priority = 50

            def on_event(self, event):
                pass

        s = MyService()
        assert s.name == "my_service"
        assert s.priority == 50
        assert s.requires == ()
        s.on_install(runtime=None)

    def test_base_service_requires_name_and_priority(self):
        class _MissingName(BaseService):
            priority = 10

            def on_event(self, event):
                pass

        with pytest.raises(TypeError, match="name"):
            _MissingName()

        class _MissingPriority(BaseService):
            name = "x"

            def on_event(self, event):
                pass

        with pytest.raises(TypeError, match="priority"):
            _MissingPriority()

    def test_base_service_does_not_require_on_event(self):
        """A BaseService subclass with no on_event and no dispatch methods is valid;
        it just no-ops on every event. (Useful as a marker / installation hook.)"""

        class _Empty(BaseService):
            name = "empty"
            priority = 10

        s = _Empty()
        # default on_event still works (no-op)
        from sakura.events import OnEpochEnd
        s.on_event(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                              rank=0, world_size=1))

    def test_base_service_dispatches_by_event_type(self):
        seen: list[str] = []

        class Routing(BaseService):
            name = "routing"
            priority = 50

            def on_train_begin(self, event):
                seen.append(f"train_begin:{event.world_size}")

            def on_epoch_end(self, event):
                seen.append(f"epoch_end:{event.epoch}")

        s = Routing()
        s.on_event(OnTrainBegin(model=None, optimizer=None, train_loader=None,
                                rank=0, world_size=4))
        s.on_event(OnEpochEnd(epoch=2, model=None, optimizer=None, metrics={},
                              rank=0, world_size=4))
        assert seen == ["train_begin:4", "epoch_end:2"]

    def test_base_service_unknown_events_silently_ignored(self):
        class Quiet(BaseService):
            name = "quiet"
            priority = 50

        s = Quiet()
        s.on_event(OnEpochEnd(epoch=0, model=None, optimizer=None, metrics={},
                              rank=0, world_size=1))
