import pytest

from zakuro_poc.execution.artifacts import create_artifact_dir, write_text_artifact
from zakuro_poc.execution.ids import new_job_id


def test_creates_job_directory(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    assert artifact_dir.exists()
    assert artifact_dir.is_dir()
    assert artifact_dir.name == job_id


def test_writes_stdout_artifact(tmp_path):
    job_id = new_job_id()
    artifact_dir = create_artifact_dir(tmp_path, job_id)
    file_path = write_text_artifact(artifact_dir, "stdout.txt", "hello world")
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "hello world"


def test_rejects_unsafe_job_id_containing_slash(tmp_path):
    with pytest.raises(ValueError, match="Unsafe job ID"):
        create_artifact_dir(tmp_path, "job/123")


def test_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Invalid artifact name"):
        write_text_artifact(tmp_path, "../outside.txt", "data")


def test_does_not_overwrite_outside_root(tmp_path):
    # Testing that write_text_artifact safely rejects malicious paths
    with pytest.raises(ValueError, match="Invalid artifact name"):
        write_text_artifact(tmp_path, "../../etc/passwd", "data")
