"""Unit tests for sakura/atelier/backends/hf_detection.py.

Runnable in CI without torch/transformers/torchmetrics/onnxruntime installed
(same CPU-only, no-download discipline as
tests/atelier/test_backend_ultralytics.py had): what is exercised here is
this module's own PURE logic -- split-path resolution against a YOLO
``data.yaml``, image listing (list file vs. directory), the
``images/`` -> ``labels/`` sibling-path mapping, and the YOLO-normalised ->
COCO-absolute box conversion that feeds ``RTDetrImageProcessor`` -- plus
backend identity/licence and that an unsupported export format is refused
before anything heavy is imported.

A real, end-to-end GPU run (train -> export -> onnxruntime runtime_eval) is
exercised by scripts/atelier/smoke_e2.py on the GPU host, not here; see this track's
handoff report for those numbers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sakura.atelier.backends import hf_detection as backend_mod
from sakura.atelier.types import RunDir


def test_backend_identity():
    b = backend_mod.BACKEND
    assert b.name == "hf_detection"
    assert b.tasks == frozenset({"object_detection"})
    assert b.licence == "Apache-2.0"


def test_export_refuses_an_unsupported_format_before_touching_the_result(tmp_path):
    # The format check runs before `import torch` in export() (so a bad
    # format never pays for a heavy import), but torch may already be in
    # sys.modules from an earlier test in this session (it IS installed in
    # the fast CI job, unlike ultralytics never was) -- assert the refusal
    # itself, and that it never touches `result` (a MagicMock would not
    # raise AttributeError the same way a real TrainResult might).
    out = RunDir(tmp_path / "run").ensure()
    result = MagicMock()
    with pytest.raises(ValueError, match="only exports 'onnx'"):
        backend_mod.BACKEND.export(result=result, fmt="torchscript", out=out)
    result.model.eval.assert_not_called()


def test_resolve_split_path_relative_to_the_yaml_path():
    doc = {"path": "/data/det", "train": "images/train", "val": "images/val"}
    assert backend_mod._resolve_split_path(doc, "train") == Path("/data/det/images/train")
    assert backend_mod._resolve_split_path(doc, "val") == Path("/data/det/images/val")


def test_resolve_split_path_absolute_is_used_as_is():
    doc = {"path": "/ignored", "val": "/elsewhere/val"}
    assert backend_mod._resolve_split_path(doc, "val") == Path("/elsewhere/val")


def test_resolve_split_path_without_a_path_key_is_relative_to_cwd():
    doc = {"train": "train/"}
    assert backend_mod._resolve_split_path(doc, "train") == Path("train/")


def test_list_split_images_reads_a_list_file(tmp_path):
    val_txt = tmp_path / "val_holdout.txt"
    val_txt.write_text("./images/a.jpg\n./images/b.jpg\n./images/c.jpg\n")
    assert backend_mod.list_split_images(val_txt) == [
        Path("./images/a.jpg"), Path("./images/b.jpg"), Path("./images/c.jpg"),
    ]


def test_list_split_images_reads_a_directory_and_ignores_non_images(tmp_path):
    val_dir = tmp_path / "images" / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "b.png").write_bytes(b"x")
    (val_dir / "a.jpg").write_bytes(b"x")
    (val_dir / "readme.txt").write_bytes(b"x")
    assert backend_mod.list_split_images(val_dir) == [val_dir / "a.jpg", val_dir / "b.png"]


def test_list_split_images_missing_path_is_empty(tmp_path):
    assert backend_mod.list_split_images(tmp_path / "nope") == []


def test_label_path_for_swaps_the_images_component():
    img = Path("/data/det/images/val/000000000149.jpg")
    assert backend_mod.label_path_for(img) == Path("/data/det/labels/val/000000000149.txt")


def test_label_path_for_handles_an_image_with_no_images_component():
    img = Path("/flat/0.jpg")
    assert backend_mod.label_path_for(img) == Path("/flat/0.txt")


def test_yolo_label_lines_to_coco_boxes_converts_center_to_top_left():
    # class=0, center (0.5, 0.5), size (0.2, 0.4) on a 100x200 image
    lines = ["0 0.5 0.5 0.2 0.4"]
    boxes = backend_mod.yolo_label_lines_to_coco_boxes(lines, width=100, height=200)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["category_id"] == 0
    assert b["bbox"] == pytest.approx([40.0, 60.0, 20.0, 80.0])  # x, y, w, h
    assert b["area"] == pytest.approx(20.0 * 80.0)
    assert b["iscrowd"] == 0


def test_yolo_label_lines_to_coco_boxes_skips_blank_lines():
    lines = ["", "  ", "1 0.5 0.5 0.1 0.1"]
    boxes = backend_mod.yolo_label_lines_to_coco_boxes(lines, width=10, height=10)
    assert len(boxes) == 1
    assert boxes[0]["category_id"] == 1


def test_coco_boxes_to_xyxy_converts_xywh_to_corners():
    boxes = [{"bbox": [10.0, 20.0, 5.0, 8.0], "category_id": 0, "area": 40.0, "iscrowd": 0}]
    assert backend_mod.coco_boxes_to_xyxy(boxes) == [[10.0, 20.0, 15.0, 28.0]]


def test_read_yolo_item_with_no_label_file_is_background_only(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8")
    item = backend_mod.read_yolo_item(img, tmp_path / "x.txt")
    assert item["image_path"] == img
    assert item["label_lines"] == []


def test_read_yolo_item_reads_existing_label_lines(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8")
    lbl = tmp_path / "x.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    item = backend_mod.read_yolo_item(img, lbl)
    assert len(item["label_lines"]) == 2


def test_load_id2label_is_ordered_by_index():
    id2label, label2id = backend_mod.load_id2label(["cat", "dog", "bird"])
    assert id2label == {0: "cat", 1: "dog", 2: "bird"}
    assert label2id == {"cat": 0, "dog": 1, "bird": 2}


def test_load_split_wires_data_yaml_path_resolution_to_the_label_sibling(tmp_path):
    det = tmp_path / "det"
    (det / "images" / "val").mkdir(parents=True)
    (det / "labels" / "val").mkdir(parents=True)
    (det / "images" / "val" / "0.jpg").write_bytes(b"\xff\xd8")
    (det / "labels" / "val" / "0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    data_yaml = det / "data.yaml"
    data_yaml.write_text(f"path: {det}\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n")

    items = backend_mod._load_split(str(data_yaml), "val")
    assert len(items) == 1
    assert items[0]["image_path"] == det / "images" / "val" / "0.jpg"
    assert items[0]["label_lines"] == ["0 0.5 0.5 0.2 0.2"]
