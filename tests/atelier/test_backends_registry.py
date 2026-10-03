"""Routing a spec to a backend must never import a training stack."""

import sys

import pytest

from sakura.atelier.backends import backend_name


def test_tasks_route_to_their_phase1_backend():
    assert backend_name("image_classification", "resnet18") == "ludwig"
    assert backend_name("object_detection", "yolo11n") == "ultralytics"
    assert backend_name("text_generation", "lfm2-350m") == "peft"


def test_a_model_can_pick_a_backend_other_than_its_tasks_default():
    assert backend_name("tabular_classification", "tabpfn_v2") == "tabpfn"
    assert backend_name("tabular_classification", "ludwig_ecd") == "ludwig"


def test_an_unknown_task_says_so():
    with pytest.raises(ValueError, match="no backend"):
        backend_name("speech_synthesis", "x")


def test_importing_the_engine_imports_no_training_stack():
    import sakura.atelier  # noqa: F401
    for stack in ("ludwig", "ultralytics", "tabpfn", "peft"):
        assert stack not in sys.modules, f"importing sakura.atelier pulled in {stack}"
