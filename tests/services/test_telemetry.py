"""Telemetry: per-event JSON record sink."""
from __future__ import annotations

import io
import json

from sakura.events import OnEpochEnd, OnTrainBegin
from sakura.runtime import SakuraRuntime
from sakura.services.telemetry import Telemetry


class TestTelemetry:
    def test_records_event_to_sink(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        rt.dispatch(OnTrainBegin(model="m", optimizer="o", train_loader="loader",
                                  val_loader=None, rank=0, world_size=1))
        rt.dispatch(OnEpochEnd(epoch=3, model="m", optimizer="o", metrics={"loss": 0.21},
                                rank=0, world_size=1))
        assert len(sink) == 2
        assert sink[0]["event"] == "OnTrainBegin"
        assert sink[1]["event"] == "OnEpochEnd"
        assert sink[1]["payload"]["epoch"] == 3
        assert sink[1]["payload"]["metrics"] == {"loss": 0.21}
        assert "ts" in sink[1]
        assert sink[1]["service"] == "telemetry"

    def test_writes_jsonl_to_file_path(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        rt = SakuraRuntime()
        rt.install(Telemetry(output=str(path)))
        rt.dispatch(OnEpochEnd(epoch=0, model="m", optimizer="o", metrics={},
                                rank=0, world_size=1))
        rt.shutdown()  # flush
        with rt:
            pass  # restart and shut down again to confirm idempotent flush
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["event"] == "OnEpochEnd"

    def test_writes_jsonl_to_stream(self):
        buf = io.StringIO()
        rt = SakuraRuntime()
        rt.install(Telemetry(output=buf))
        rt.dispatch(OnEpochEnd(epoch=1, model="m", optimizer="o", metrics={"a": 1.0},
                                rank=0, world_size=1))
        rt.shutdown()
        rec = json.loads(buf.getvalue().strip())
        assert rec["event"] == "OnEpochEnd"
        assert rec["payload"]["metrics"] == {"a": 1.0}

    def test_priority_is_zero(self):
        t = Telemetry(output=lambda r: None)
        assert t.priority == 0
        assert t.name == "telemetry"

    def test_skips_non_serializable_payload_gracefully(self):
        sink: list[dict] = []
        rt = SakuraRuntime()
        rt.install(Telemetry(output=sink.append))
        # Non-serializable: an actual model object
        class _Mdl:
            pass
        rt.dispatch(OnEpochEnd(epoch=0, model=_Mdl(), optimizer=_Mdl(), metrics={},
                                rank=0, world_size=1))
        # Should record without raising; payload omits non-serializable fields.
        assert len(sink) == 1
        # epoch must still be there (it's an int)
        assert sink[0]["payload"]["epoch"] == 0
