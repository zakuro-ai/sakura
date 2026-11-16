"""models.yaml is the only hand-kept routing table (plan Task 1)."""

import json
import pkgutil
import subprocess
import sys

import sakura.atelier.backends as backends_pkg
from sakura.atelier.backends import MODEL_BACKENDS, TASK_BACKENDS, backend_name
from sakura.atelier.registry.routes import families, routes

#: The tables as they were hand-kept before this change: routing must not move.
PREVIOUS_TASKS = {
    "image_classification": "ludwig", "image_regression": "open_clip",
    "audio_classification": "ludwig", "tabular_classification": "ludwig",
    "tabular_regression": "ludwig", "object_detection": "hf_detection",
    "text_generation": "peft", "speech_recognition": "deepspeech",
    "time_series_forecasting": "ludwig", "text_classification": "ludwig", "decision": "peft",
    "risk_simulation": "risk_sim",
    "reinforcement_learning": "rl",
}
PREVIOUS_MODELS = {
    "resnet18": "ludwig", "stacked_cnn_audio": "ludwig", "ludwig_ecd": "ludwig",
    "tabpfn_v2": "tabpfn", "rtdetr_r18": "hf_detection", "lfm2-350m": "peft",
    "hf-causal-lm": "peft", "hf-decision-lm": "peft", "deepspeech2": "deepspeech",
    "whisper-small": "whisper", "whisper-large-v3-turbo": "whisper",
    "ludwig_forecaster": "ludwig", "ludwig_text": "ludwig", "siglip2_b16_256": "open_clip",
    "hf-image-classifier": "hf_image_classification", "hf-audio-classifier": "hf_audio_classification",
    "hf-text-classifier": "hf_text_classification",
    "scvar-copula": "risk_sim",
    "kg-path-policy": "rl",
}


def test_routes_match_previous_tables():
    r = routes()
    assert r["contract"] == 2
    assert {t: r["tasks"][t] for t in PREVIOUS_TASKS} == PREVIOUS_TASKS
    assert {m: r["models"][m] for m in PREVIOUS_MODELS} == PREVIOUS_MODELS
    assert TASK_BACKENDS == r["tasks"]
    from tests.atelier.test_decision_spec import MIRRORED_MODEL_BACKENDS  # the one hub-mirrored copy

    assert MODEL_BACKENDS == MIRRORED_MODEL_BACKENDS
    for model, backend in PREVIOUS_MODELS.items():
        assert backend_name("x_unknown_task", model) == backend


def _backend_modules():
    out = set()
    for info in pkgutil.iter_modules(backends_pkg.__path__):
        src = (backends_pkg.__path__[0] + f"/{info.name}.py")
        try:
            text = open(src).read()
        except OSError:
            continue
        if "\nBACKEND" in text:
            out.add(info.name)
    return out


def test_every_backend_module_is_in_a_family():
    fams = families()
    assert set(fams["all"]) == set().union(*(set(v) for k, v in fams.items() if k != "all"))
    seen = [b for k, v in fams.items() if k != "all" for b in v]
    assert len(seen) == len(set(seen)), "a backend is in two families"
    assert _backend_modules() <= set(seen)
    assert set(seen) <= _backend_modules(), "a family lists a backend that does not exist"


def test_hf_ref_routes_by_task():
    sha = "a" * 40
    assert backend_name("object_detection", f"hf:PekingU/rtdetr_r50vd@{sha}") == "hf_detection"
    assert backend_name("text_generation", f"hf:HuggingFaceTB/SmolLM2-135M@{sha}") == "peft"
    assert backend_name("speech_recognition", f"hf:openai/whisper-base@{sha}") == "whisper"


def test_routes_cli_filters_by_family():
    out = subprocess.run([sys.executable, "-m", "sakura.atelier", "routes", "--family", "audio"],
                         capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    assert set(r["models"].values()) <= set(families()["audio"])
    assert "whisper-small" in r["models"] and "resnet18" not in r["models"]
    assert r["tasks"].get("speech_recognition") == "deepspeech"
    assert "image_classification" not in r["tasks"]


def test_routes_name_each_models_tasks():
    """The hub builds TASK_MODELS from this (zak-marketplace forge/atelier_routes.json)."""
    r = routes()
    assert set(r["model_tasks"]) == set(r["models"])
    assert r["model_tasks"]["lightgbm"] == ["tabular_classification", "tabular_regression"]
    assert r["model_tasks"]["hf-decision-lm"] == ["decision"]
    only_peft = routes(["peft"])
    assert set(only_peft["model_tasks"]) == set(only_peft["models"])


def test_routes_name_the_hf_templates():
    """An hf:<repo>@<sha> model inherits its template's exports; the hub prices it by the template too."""
    r = routes()
    assert r["hf_templates"]["image_classification"] == "hf-image-classifier"
    assert all(r["models"][m] == r["hf_tasks"][t] for t, m in r["hf_templates"].items())
    assert routes(["peft"])["hf_templates"] == {"text_generation": "hf-causal-lm"}


CPU_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest", "hist_gradient_boosting",
              "ludwig_ecd", "ludwig_forecaster", "scvar-copula", "kg-path-policy"}


def test_routes_name_each_models_device():
    """The hub drafts `placement.gpus: 0` for a cpu model and prices a gpu model on a CPU as slow."""
    r = routes()
    assert set(r["devices"]) == set(r["models"])
    assert {m for m, d in r["devices"].items() if d == "cpu"} == CPU_MODELS
    assert set(r["devices"].values()) == {"cpu", "gpu"}
    assert r["devices"]["resnet18"] == "gpu" and r["devices"]["tabpfn_v2"] == "gpu"
    only_classic = routes(["classic_ml"])
    assert set(only_classic["devices"]) == set(only_classic["models"])
    assert set(only_classic["devices"].values()) == {"cpu"}


def test_models_yaml_device_is_validated():
    import pytest

    from sakura.atelier.registry.routes import check_models

    assert check_models({"m": {"device": "cpu"}}) == {"m": {"device": "cpu"}}
    with pytest.raises(ValueError, match="device"):
        check_models({"m": {"backend": "ludwig"}})
    with pytest.raises(ValueError, match="device"):
        check_models({"m": {"device": "tpu"}})
