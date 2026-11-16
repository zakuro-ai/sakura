"""`model: hf:<owner>/<repo>@<40-hex commit>`: fine-tune any Hugging Face repo,
pinned to one commit so a run is reproducible after the repo moves."""

from __future__ import annotations

import re
from dataclasses import dataclass

_REF = re.compile(r"^hf:(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)@(?P<rev>[0-9a-f]{40})$")


@dataclass(frozen=True)
class HfRef:
    repo: str
    revision: str

    @property
    def uri(self) -> str:
        return f"hf://{self.repo}@{self.revision}"


def parse_hf_model(model: str) -> HfRef | None:
    """The pinned ref of an `hf:` model id, None for a registry model id.
    Raises ValueError for an `hf:` id that is not `owner/repo@<40-hex sha>`."""
    if not model.startswith("hf:"):
        return None
    m = _REF.match(model)
    if not m or ".." in m.group("repo"):
        raise ValueError(f"model {model!r}: expected hf:<owner>/<repo>@<40-hex commit sha>")
    return HfRef(m.group("repo"), m.group("rev"))
