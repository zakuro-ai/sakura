"""Ship the model's licence with every release (`report.json` `license` +
files under `artifacts/`). An `hf:` model brings its repo's LICENSE/NOTICE
files and card licence at the pinned commit; a registry model its
`models.yaml` `license` and `source`. A fetch failure is recorded, not fatal:
the hub shows the licence as unknown rather than losing a finished run."""

from __future__ import annotations

import os
from typing import Any, Callable

from sakura.atelier.hf_ref import HfRef, parse_hf_model
from sakura.atelier.types import RunDir

_LICENCE_FILE_PREFIXES = ("LICENSE", "LICENCE", "NOTICE", "COPYING")


def _hub_fetch(ref: HfRef) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download  # backend venvs carry it (transformers)
    api = HfApi()
    info = api.model_info(ref.repo, revision=ref.revision)
    card = getattr(info, "card_data", None) or {}
    spdx = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    files: dict[str, bytes] = {}
    for name in api.list_repo_files(ref.repo, revision=ref.revision):
        if "/" not in name and name.upper().startswith(_LICENCE_FILE_PREFIXES):
            with open(hf_hub_download(ref.repo, name, revision=ref.revision), "rb") as f:
                files[name] = f.read()
    return {"license": spdx, "files": files}


def _unfetched(ref: HfRef, out: RunDir, error: str) -> dict[str, Any]:
    """The release still carries a licence file: which repo's licence applies, and where to read it."""
    (out.artifacts / "LICENSE.txt").write_text(
        f"The licence of {ref.uri} could not be fetched ({error}).\n"
        f"It is the licence published at https://huggingface.co/{ref.repo} (revision {ref.revision}).\n",
        encoding="utf-8")
    return {"spdx": None, "files": ["artifacts/LICENSE.txt"], "source": ref.uri, "error": error}


def collect_licence(model: str, base_model: str | None, out: RunDir,
                    fetch: Callable[[HfRef], dict[str, Any]] = _hub_fetch,
                    base_revision: str | None = None) -> dict[str, Any]:
    """Write the licence files into `out.artifacts`; the report's `license` block.
    A registry model whose licence is `per-base-model` ships its base repo's."""
    out.artifacts.mkdir(parents=True, exist_ok=True)
    ref = parse_hf_model(model)
    if ref is None and base_model:
        from sakura.atelier.registry import load_models
        if (load_models().get(model) or {}).get("license") == "per-base-model":
            ref = HfRef(base_model, base_revision or "main")
    if ref is not None:
        if fetch is _hub_fetch and os.environ.get("SAKURA_ATELIER_LICENCE_FETCH") == "0":
            return _unfetched(ref, out, "licence fetch disabled")
        try:
            got = fetch(ref)
        except Exception as e:  # noqa: BLE001 - recorded on the report, never fatal
            return _unfetched(ref, out, f"{type(e).__name__}: {e}")
        # ONE file whatever the repo ships (LICENSE, NOTICE, ...): the hub fetches each artifact by its
        # format, so a release declares a single `license` artifact.
        files = sorted((got.get("files") or {}).items())
        spdx = got.get("license")
        parts = [f"Licence of {ref.uri}: {spdx or 'not declared on the model card'}\n"]
        if not files:
            parts.append(f"The repository ships no licence file; see https://huggingface.co/{ref.repo} "
                         f"(revision {ref.revision}).\n")
        for name, data in files:
            parts.append(f"\n== {name} ==\n\n{data.decode('utf-8', errors='replace')}\n")
        (out.artifacts / "LICENSE.txt").write_text("".join(parts), encoding="utf-8")
        return {"spdx": spdx, "files": ["artifacts/LICENSE.txt"], "source": ref.uri,
                "included": [name for name, _ in files]}
    from sakura.atelier.registry import load_models
    entry = load_models().get(model) or {}
    spdx, source = entry.get("license"), entry.get("source")
    path = out.artifacts / "LICENSE.txt"
    path.write_text(f"{spdx}\nsource: {source}\n" + (f"base model: {base_model}\n" if base_model else ""),
                    encoding="utf-8")
    return {"spdx": spdx, "files": ["artifacts/LICENSE.txt"], "source": source}
