"""What the code ran on, written so a reader can rebuild it three ways: the
runner image by digest (when the run was containerised), ``init.sh`` with uv
on a bare machine, or the Dockerfile that combines the two. Pins come from the
venv that is executing this very run, not from a declared dependency list.

Reads exactly two variables from the environment, ``SAKURA_ATELIER_IMAGE_REF``
and ``SAKURA_ATELIER_IMAGE_DIGEST``; nothing else from ``os.environ`` (in
particular never ``SAKURA_ATELIER_LLM_KEY``) reaches the bundle.

Contract: ``docs/atelier/CONTRACTS.md`` section 2.5.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

_IMAGE_REF = "SAKURA_ATELIER_IMAGE_REF"
_IMAGE_DIGEST = "SAKURA_ATELIER_IMAGE_DIGEST"
_DEFAULT_CUDA = "12.6.3"

_TORCH_LOCAL = re.compile(r"\+(cu\d+|rocm[\d.]+)$")


def _torch_index(version: str) -> str | None:
    """The PyTorch wheel index implied by a local version tag (``+cu124``,
    ``+rocm6.2``); ``None`` for PyPI builds and ``+cpu``."""
    m = _TORCH_LOCAL.search(version)
    return f"https://download.pytorch.org/whl/{m.group(1)}" if m else None


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def capture_environment(backend: str, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The facts a reader needs to rebuild the run's machine. ``image`` is
    ``None`` when the run was not containerised. Hostname omitted on purpose."""
    e: Mapping[str, str] = os.environ if environ is None else environ
    ref, digest = e.get(_IMAGE_REF), e.get(_IMAGE_DIGEST)
    torch_version: str | None = None
    cuda: str | None = None
    gpu: str | None = None
    try:
        import torch  # the venv has it when the backend trained with it

        torch_version = str(torch.__version__)
        cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - absent, broken, or a driver that refuses to init
        pass
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "image": {"ref": ref, "digest": digest} if ref or digest else None,
        "python": platform.python_version(),
        "backend": backend,
        "cuda": cuda,
        "driver": driver.splitlines()[0].strip() if driver else None,
        "gpu": gpu,
        "torch": torch_version,
        "platform": platform.system(),
    }


def requirements_lock() -> str:
    """Every distribution visible to this interpreter, ``name==version``, sorted
    case-insensitively; git installs keep their ``@ git+url@sha`` form; a single
    ``--extra-index-url`` first line when torch carries a CUDA/ROCm tag."""
    lines: set[str] = set()
    index: str | None = None
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        direct = dist.read_text("direct_url.json")
        if direct:
            try:
                d = json.loads(direct)
                vcs = d["vcs_info"]
                lines.add(f"{name} @ {vcs['vcs']}+{d['url']}@{vcs.get('commit_id', '')}")
                continue
            except (ValueError, KeyError, TypeError):
                pass
        lines.add(f"{name}=={dist.version}")
        if name.lower() == "torch":
            index = _torch_index(dist.version)
    body = "\n".join(sorted(lines, key=str.lower))
    return (f"--extra-index-url {index}\n" if index else "") + body + "\n"


def apt_packages() -> str:
    out = _run(["dpkg-query", "-W", "-f", "${binary:Package}=${Version}\\n"])
    return (out + "\n") if out else "# dpkg not available on this host; system packages were not captured\n"


_INIT = """#!/usr/bin/env bash
# Recreate the venv this notebook ran in. Idempotent; stops at the first failing line.
set -euo pipefail
cd "$(dirname "$0")"
PY=@@PYTHON@@
if command -v uv >/dev/null 2>&1; then
  [ -d ../.venv ] || uv venv --python "$PY" ../.venv
  uv pip install --python ../.venv/bin/python -r requirements.lock.txt
else
  echo "uv not found; falling back to python -m venv (install uv: https://docs.astral.sh/uv/)" >&2
  [ -d ../.venv ] || "python$PY" -m venv ../.venv
  ../.venv/bin/pip install -r requirements.lock.txt
fi
../.venv/bin/python -m ipykernel install --user --name atelier-@@BACKEND@@ >/dev/null
echo "ready: ../.venv (kernel atelier-@@BACKEND@@). Run: ../.venv/bin/jupyter lab ../notebook.ipynb"
"""

_DOCKERFILE_IMAGE = """# The exact runner image this job executed in; nothing to install.
FROM @@IMAGE@@
WORKDIR /work
COPY . /work
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "notebook.ipynb"]
"""

_DOCKERFILE_FALLBACK = """# This run was not containerised (image unknown); rebuild from the CUDA base it reported.
FROM nvidia/cuda:@@CUDA@@-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python@@PYTHON@@ python@@PYTHON@@-venv curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/* && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
WORKDIR /work
COPY . /work
RUN bash environment/init.sh
CMD ["/work/.venv/bin/python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "notebook.ipynb"]
"""


def write_environment(env_dir: Path, env: dict[str, Any], lock: str, apt: str) -> list[Path]:
    """Write ``image.json``, ``requirements.lock.txt``, ``apt-packages.txt``,
    ``init.sh`` (0755) and ``Dockerfile`` into ``env_dir``; returns the paths."""
    from sakura.atelier.notebook.cells import fill

    env_dir.mkdir(parents=True, exist_ok=True)
    py = ".".join(str(env["python"]).split(".")[:2])
    image: dict[str, Any] = env.get("image") or {}
    if image.get("digest"):
        ref = str(image.get("ref") or "runner").split("@")[0]
        if ":" in ref.rsplit("/", 1)[-1]:
            ref = ref.rsplit(":", 1)[0]
        docker = fill(_DOCKERFILE_IMAGE, IMAGE=f"{ref}@{image['digest']}")
    else:
        docker = fill(_DOCKERFILE_FALLBACK, CUDA=str(env.get("cuda") or _DEFAULT_CUDA), PYTHON=py)
    files = {
        "image.json": json.dumps(env, indent=2, sort_keys=True) + "\n",
        "requirements.lock.txt": lock,
        "apt-packages.txt": apt,
        "init.sh": fill(_INIT, PYTHON=py, BACKEND=str(env["backend"])),
        "Dockerfile": docker,
    }
    paths: list[Path] = []
    for name, text in files.items():
        p = env_dir / name
        p.write_text(text)
        if name == "init.sh":
            p.chmod(0o755)
        paths.append(p)
    return paths
