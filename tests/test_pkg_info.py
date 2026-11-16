import pathlib
import re
import subprocess
import sys

import pytest
import sakura

_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPkgInfo:
    def test_version_format(self):
        assert re.match(r"^\d+\.\d+\.\d+", sakura.__version__), (
            f"Version '{sakura.__version__}' does not match semver pattern"
        )

    def test_build_is_string(self):
        assert isinstance(sakura.__build__, str)


class TestNoGnutools:
    """gnutools-python is not a dependency (#23): nothing in sakura/ may need it."""

    def test_not_a_runtime_dependency(self):
        if sys.version_info < (3, 11):
            pytest.skip("tomllib is stdlib only from 3.11")
        import tomllib

        deps = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        assert not any(d.lower().startswith("gnutools") for d in deps), deps

    def test_sakura_sources_do_not_reference_gnutools(self):
        offenders = [
            str(p.relative_to(_ROOT))
            for p in (_ROOT / "sakura").rglob("*.py")
            if "gnutools" in p.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_import_sakura_with_gnutools_blocked(self):
        # A None entry in sys.modules makes `import gnutools` raise ImportError.
        code = "import sys; sys.modules['gnutools'] = None; import sakura"
        subprocess.run([sys.executable, "-c", code], cwd=_ROOT, check=True)
