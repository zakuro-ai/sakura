"""A ``validation_fraction`` holdout on an hf:// dataset is drawn from ``train``, else from its only split.

A dataset repo that publishes one ``preview`` split (a sample of a larger release) failed with
``Unknown split "train"``: the engine always loaded ``train``, and a spec cannot name another split
alongside ``validation_fraction``. Several splits with no ``train`` stay an error, one that lists them."""
from __future__ import annotations

import sys
import types

import pytest

from sakura.atelier.data import _hf_holdout_source, materialise
from sakura.atelier.spec import load_spec


def _fake(names):
    fake = types.ModuleType("datasets")

    def get_dataset_split_names(repo, **kwargs):
        if isinstance(names, Exception):
            raise names
        return names

    fake.get_dataset_split_names = get_dataset_split_names
    return fake


@pytest.mark.parametrize("names, expected", [
    (["train", "test"], "train"),
    (["train"], "train"),
    (["preview"], "preview"),
    (["test"], "test"),
    ([], "train"),
])
def test_the_holdout_comes_from_train_else_the_only_split(names, expected):
    assert _hf_holdout_source(_fake(names), "o/n", {}) == expected


def test_split_names_that_cannot_be_read_keep_train():
    assert _hf_holdout_source(_fake(ConnectionError("offline")), "o/n", {}) == "train"
    assert _hf_holdout_source(types.ModuleType("datasets"), "o/n", {}) == "train"


def test_several_splits_without_train_are_refused_by_name():
    with pytest.raises(ValueError) as e:
        _hf_holdout_source(_fake(["preview", "extra"]), "o/n", {})
    assert "['preview', 'extra']" in str(e.value)
    assert 'train: "preview[:80%]"' in str(e.value)


def test_the_revision_is_passed_through():
    seen = {}
    fake = types.ModuleType("datasets")
    fake.get_dataset_split_names = lambda repo, **kw: seen.update(kw) or ["preview"]
    assert _hf_holdout_source(fake, "o/n", {"revision": "v1"}) == "preview"
    assert seen == {"revision": "v1"}


SPEC = """
atelier: 1
name: preview-only
task: tabular_classification
data: {{uri: "hf://{repo}", format: hf, split: {{validation_fraction: 0.25}}}}
features:
  inputs: [{{name: x, type: number}}]
  output: {{name: y, type: category}}
model: ludwig_ecd
hp: tabular/ecd@1
export: [onnx]
"""

#: The dataset card of a repo that publishes only a preview split (the layout of a
#: large release's sample repo): a named config whose one split is not `train`.
CARD = """---
configs:
  - config_name: previews
    default: true
    data_files:
      - split: preview
        path: preview/**
---
"""


def test_a_real_dataset_with_only_a_preview_split(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    repo = tmp_path / "ds"
    (repo / "preview").mkdir(parents=True)
    (repo / "README.md").write_text(CARD)
    datasets.Dataset.from_dict({"x": [float(i) for i in range(8)], "y": ["ab"[i % 2] for i in range(8)]}
                               ).to_parquet(str(repo / "preview" / "rows.parquet"))
    data = materialise(load_spec(SPEC.format(repo=repo)), tmp_path / "d")
    assert len(data.train) + len(data.validation) == 8
    assert len(data.validation) == 2
