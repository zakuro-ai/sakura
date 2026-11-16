"""Unit tests for sakura/atelier/backends/open_clip.py: its pure logic (the one
preprocessing path, metrics, LR schedule, checkpoint key mapping) and identity,
runnable without open_clip/onnxruntime. The real train -> ONNX -> onnxruntime
path runs in the backend's own env (docs/atelier/envs/open_clip.txt)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from sakura.atelier.backends import open_clip as backend_mod
from sakura.atelier.types import RunDir

CFG = {"size": 4, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "interpolation": "bicubic"}


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "open_clip"
    assert b.tasks == frozenset({"image_regression"})
    assert b.licence == "Apache-2.0"


def test_export_refuses_an_unsupported_format_before_touching_the_result(tmp_path):
    result = MagicMock()
    with pytest.raises(ValueError, match="only exports 'onnx'"):
        backend_mod.BACKEND.export(result=result, fmt="torchscript", out=RunDir(tmp_path / "run").ensure())
    result.model.__getitem__.assert_not_called()


def test_to_array_squash_is_chw_and_normalised():
    x = backend_mod.to_array(Image.new("RGB", (10, 6), (255, 0, 0)), {**CFG, "resize_mode": "squash"})
    assert x.shape == (3, 4, 4) and x.dtype == np.float32
    assert np.allclose(x[0], 1.0) and np.allclose(x[1:], -1.0)  # (1 - .5) / .5, (0 - .5) / .5


def test_to_array_shortest_centre_crops_a_wide_image():
    img = Image.new("RGB", (12, 4), (0, 0, 0))
    img.paste((255, 255, 255), (4, 0, 8, 4))  # white middle third
    x = backend_mod.to_array(img, {**CFG, "resize_mode": "shortest"})
    assert x.shape == (3, 4, 4)
    assert np.allclose(x, 1.0, atol=0.1)  # only the white centre survives the crop


def test_regression_metrics():
    m = backend_mod.regression_metrics([1.0, 3.0], [2.0, 2.0])
    assert m == {"mae": 1.0, "rmse": 1.0}


def test_warmup_then_cosine_to_zero():
    f = backend_mod.warmup_cosine
    assert f(0, 10, 100) == 0.0 and f(5, 10, 100) == 0.5 and f(10, 10, 100) == 1.0
    assert f(100, 10, 100) == pytest.approx(0.0)


def test_bulkorcut_checkpoint_keys_map_onto_this_module():
    state = {"backbone.visual.trunk.w": 1, "head.weight": 2, "visual.x": 3}
    assert backend_mod.remap_state(state) == {"visual.trunk.w": 1, "head.weight": 2, "visual.x": 3}


def test_every_model_id_has_an_arch():
    from sakura.atelier.registry import load_models

    ours = [m for m, e in load_models().items() if e["backend"] == "open_clip"]
    assert ours and all(m in backend_mod.ARCHS for m in ours)
