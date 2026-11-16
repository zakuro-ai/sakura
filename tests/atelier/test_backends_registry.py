"""Routing a spec to a backend must never import a training stack."""

import sys

import pytest

from sakura.atelier.backends import backend_name


def test_tasks_route_to_their_phase1_backend():
    assert backend_name("image_classification", "resnet18") == "ludwig"
    assert backend_name("object_detection", "rtdetr_r18") == "hf_detection"
    assert backend_name("image_regression", "siglip2_b16_256") == "open_clip"
    assert backend_name("text_generation", "lfm2-350m") == "peft"


def test_a_model_can_pick_a_backend_other_than_its_tasks_default():
    assert backend_name("tabular_classification", "tabpfn_v2") == "tabpfn"
    assert backend_name("tabular_classification", "ludwig_ecd") == "ludwig"


def test_an_unknown_task_says_so():
    with pytest.raises(ValueError, match="no backend"):
        backend_name("speech_synthesis", "x")


def test_importing_the_engine_imports_no_training_stack():
    # A fresh interpreter, measured against `import sakura` itself: the core package may import
    # transformers/peft when they are installed (its HF adapter), but sakura.atelier must add no backend stack.
    import subprocess

    code = (
        "import sys, sakura\n"
        "before = set(sys.modules)\n"
        "import sakura.atelier\n"
        "stacks = ('ludwig', 'ultralytics', 'tabpfn', 'peft', 'transformers', 'asr_deepspeech', 'torchmetrics', 'onnxruntime', 'open_clip', 'timm')\n"
        "bad = [m for m in stacks if m in sys.modules and m not in before]\n"
        "assert not bad, f'importing sakura.atelier pulled in {bad}'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_loading_the_peft_backend_does_not_import_the_decision_torch_half_or_a_stack():
    # The decision task adds a torch-only module next to peft.py. Resolving the backend by name (which the runner's
    # validate path does) must not import it, nor `peft` / `transformers`: those exist only in the backend's venv.
    import subprocess

    code = (
        "import sys, sakura\n"
        "before = set(sys.modules)\n"
        "import sakura.atelier.decision\n"
        "from sakura.atelier.backends import load_backend\n"
        "b = load_backend('peft')\n"
        "assert 'decision' in b.tasks\n"
        "assert 'sakura.atelier.backends.decision_lm' not in sys.modules, 'the torch half was imported'\n"
        "bad = [m for m in ('peft', 'transformers', 'bitsandbytes', 'ludwig') if m in sys.modules and m not in before]\n"
        "assert not bad, f'loading the peft backend pulled in {bad}'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
