import json
import re
import stat
import subprocess
import sys
from importlib import metadata

import pytest

from sakura.atelier.notebook import environment as env

IMAGE_ENV = {
    "SAKURA_ATELIER_IMAGE_REF": "r/x:tag",
    "SAKURA_ATELIER_IMAGE_DIGEST": "sha256:" + "0" * 64,
}
BARE = {
    "image": None,
    "python": "3.11.9",
    "backend": "ludwig",
    "cuda": "12.4",
    "driver": None,
    "gpu": None,
    "torch": "2.6.0+cu124",
    "platform": "Linux",
}


def test_capture_reads_image_from_env_and_is_null_without_it():
    e = env.capture_environment("ludwig", IMAGE_ENV)
    assert e["image"] == {"ref": "r/x:tag", "digest": "sha256:" + "0" * 64}
    assert e["backend"] == "ludwig"
    assert env.capture_environment("ludwig", {})["image"] is None
    assert e["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert set(e) == {"image", "python", "backend", "cuda", "driver", "gpu", "torch", "platform"}


def test_capture_keeps_a_partial_image_and_never_copies_the_llm_key():
    e = env.capture_environment(
        "peft", {"SAKURA_ATELIER_IMAGE_REF": "r/x:tag", "SAKURA_ATELIER_LLM_KEY": "sk-secret"}
    )
    assert e["image"] == {"ref": "r/x:tag", "digest": None}
    assert "sk-secret" not in json.dumps(e)


def test_requirements_lock_pins_this_interpreter_sorted_with_an_index_line_only_for_cuda_torch():
    lock = env.requirements_lock()
    assert lock.endswith("\n")
    lines = lock.splitlines()
    assert "pandas==" + metadata.version("pandas") in lines
    pins = [line for line in lines if not line.startswith("--")]
    assert pins == sorted(pins, key=str.lower)
    assert len(set(pins)) == len(pins)
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = ""
    index_lines = [line for line in lines if line.startswith("--extra-index-url")]
    if re.search(r"\+(cu\d+|rocm[\d.]+)$", torch_version):
        assert index_lines == [f"--extra-index-url {env._torch_index(torch_version)}"]
        assert lines[0] == index_lines[0]
    else:
        assert index_lines == []
    assert env._torch_index("2.6.0+cu124") == "https://download.pytorch.org/whl/cu124"
    assert env._torch_index("2.6.0+rocm6.2") == "https://download.pytorch.org/whl/rocm6.2"
    assert env._torch_index("2.6.0") is None
    assert env._torch_index("2.6.0+cpu") is None


def test_apt_packages_notes_when_dpkg_is_absent(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _cmd: None)
    assert env.apt_packages().startswith("# dpkg not available")


def test_write_environment_produces_every_file_and_an_executable_init(tmp_path):
    paths = env.write_environment(tmp_path, BARE, "numpy==1.26.4\n", "libgomp1=14.2\n")
    names = sorted(p.name for p in paths)
    assert names == ["Dockerfile", "apt-packages.txt", "image.json", "init.sh", "requirements.lock.txt"]
    assert all(p.parent == tmp_path for p in paths)
    assert stat.S_IMODE((tmp_path / "init.sh").stat().st_mode) & 0o111
    assert "nvidia/cuda:12.4" in (tmp_path / "Dockerfile").read_text()
    assert json.loads((tmp_path / "image.json").read_text())["backend"] == "ludwig"
    assert (tmp_path / "requirements.lock.txt").read_text() == "numpy==1.26.4\n"
    assert (tmp_path / "apt-packages.txt").read_text() == "libgomp1=14.2\n"
    assert subprocess.run(["bash", "-n", str(tmp_path / "init.sh")]).returncode == 0


def test_init_sh_is_idempotent_and_fully_filled(tmp_path):
    env.write_environment(tmp_path, BARE, "", "")
    init = (tmp_path / "init.sh").read_text()
    assert init.startswith("#!/usr/bin/env bash\n")
    assert "@@" not in init
    assert "set -euo pipefail" in init
    assert "PY=3.11" in init
    assert "atelier-ludwig" in init
    # re-running on a machine that already has the venv must not fail:
    # `uv venv` only runs when ../.venv is absent, and the fallback likewise.
    guards = re.findall(r'\[\s*(!?)\s*-d\s+"?\.\./\.venv"?\s*\]\s*(\|\||&&)', init)
    assert guards == [("", "||"), ("", "||")] or guards == [("!", "&&"), ("!", "&&")]
    assert init.count("uv venv") == 1 and init.count('"python$PY" -m venv') == 1
    assert "pip install" in init and "requirements.lock.txt" in init


def test_dockerfile_uses_the_image_digest_when_known(tmp_path):
    e = dict(BARE, image={"ref": "reg/zak-forge-runner:stg-1", "digest": "sha256:" + "a" * 64}, python="3.12", backend="peft")
    env.write_environment(tmp_path, e, "", "")
    docker = (tmp_path / "Dockerfile").read_text()
    assert ("FROM reg/zak-forge-runner@sha256:" + "a" * 64) in docker
    assert "@@" not in docker and "nvidia/cuda" not in docker


def test_dockerfile_falls_back_to_a_default_cuda_base_without_cuda(tmp_path):
    e = dict(BARE, cuda=None, torch=None)
    env.write_environment(tmp_path, e, "", "")
    docker = (tmp_path / "Dockerfile").read_text()
    assert "FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04" in docker
    assert "python3.11" in docker
    assert "@@" not in docker


def test_bundle_never_contains_the_llm_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKURA_ATELIER_LLM_KEY", "sk-very-secret")
    monkeypatch.setenv("SAKURA_ATELIER_IMAGE_REF", "r/x:tag")
    e = env.capture_environment("ludwig")
    assert e["image"] == {"ref": "r/x:tag", "digest": None}
    paths = env.write_environment(tmp_path, e, env.requirements_lock(), env.apt_packages())
    for p in paths:
        assert "sk-very-secret" not in p.read_text()


def test_templates_go_through_cells_fill_so_a_stray_placeholder_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "_INIT", env._INIT + "echo @@NOPE@@\n")
    with pytest.raises(ValueError, match="@@NOPE@@"):
        env.write_environment(tmp_path, BARE, "", "")
