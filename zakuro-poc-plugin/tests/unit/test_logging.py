import logging

from zakuro_poc.observability.logging import configure_logging


def test_configure_logging_sets_debug_level(monkeypatch):
    recorded = {}

    def fake_basic_config(**kwargs):  # noqa: ANN003
        recorded.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    configure_logging(verbose=True)

    assert recorded["level"] == logging.DEBUG
    assert recorded["datefmt"] == "%Y-%m-%d %H:%M:%S"


def test_configure_logging_sets_info_level(monkeypatch):
    recorded = {}

    def fake_basic_config(**kwargs):  # noqa: ANN003
        recorded.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    configure_logging(verbose=False)

    assert recorded["level"] == logging.INFO
