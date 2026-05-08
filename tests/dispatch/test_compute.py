"""Compute is a URI-tag config class that resolves to a Dispatcher at runtime.start()."""
from __future__ import annotations

import pytest

from sakura.dispatch.compute import Compute


class TestCompute:
    def test_local_default(self):
        c = Compute.local()
        assert c.kind == "local"
        assert c.n_workers == 1
        assert c.gpus is None

    def test_local_with_gpus_and_pool(self):
        c = Compute.local(n_workers=2, gpus=[0, 1])
        assert c.kind == "local"
        assert c.n_workers == 2
        assert c.gpus == (0, 1)

    def test_at_quic_uri(self):
        c = Compute.at("quic://eval-1.lan:4433")
        assert c.kind == "remote"
        assert c.uris == ("quic://eval-1.lan:4433",)
        assert c.strategy == "round-robin"

    def test_at_rejects_non_quic(self):
        with pytest.raises(ValueError, match="quic://"):
            Compute.at("http://eval-1:8080")

    def test_pool_uris_and_strategy(self):
        c = Compute.pool(
            ["quic://e1:4433", "quic://e2:4433"],
            strategy="least-loaded",
        )
        assert c.kind == "remote"
        assert c.uris == ("quic://e1:4433", "quic://e2:4433")
        assert c.strategy == "least-loaded"

    def test_pool_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            Compute.pool(["quic://e1:4433"], strategy="random")

    def test_in_thread(self):
        c = Compute.in_thread()
        assert c.kind == "in_thread"

    def test_repr_is_compact(self):
        c = Compute.at("quic://localhost:4433")
        s = repr(c)
        assert "Compute" in s
        assert "quic://localhost:4433" in s
