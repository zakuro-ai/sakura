"""Compute — URI-tag config that resolves to a concrete Dispatcher.

Compute is purely a config object. Resolution to LocalDispatcher /
RemoteDispatcher / InThreadDispatcher happens at runtime.start() (or when
a service first asks for the dispatcher). This decoupling lets services
pre-declare their compute target without forcing transport setup at
import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


_VALID_STRATEGIES = ("round-robin", "least-loaded")


@dataclass(frozen=True)
class Compute:
    """A tag describing where work should run."""
    kind: Literal["local", "remote", "in_thread"]
    n_workers: int = 1
    gpus: Optional[tuple[int, ...]] = None
    uris: tuple[str, ...] = ()
    strategy: Literal["round-robin", "least-loaded"] = "round-robin"

    @classmethod
    def local(cls, *, n_workers: int = 1, gpus: Optional[list[int]] = None) -> "Compute":
        return cls(
            kind="local",
            n_workers=int(n_workers),
            gpus=tuple(gpus) if gpus is not None else None,
        )

    @classmethod
    def at(cls, uri: str) -> "Compute":
        if not uri.startswith("quic://"):
            raise ValueError(f"Compute.at requires a quic:// URI, got: {uri}")
        return cls(kind="remote", uris=(uri,), strategy="round-robin")

    @classmethod
    def pool(
        cls,
        uris: list[str],
        *,
        strategy: Literal["round-robin", "least-loaded"] = "round-robin",
    ) -> "Compute":
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got: {strategy!r}"
            )
        for u in uris:
            if not u.startswith("quic://"):
                raise ValueError(f"all pool URIs must be quic://, got: {u}")
        return cls(kind="remote", uris=tuple(uris), strategy=strategy)

    @classmethod
    def in_thread(cls) -> "Compute":
        return cls(kind="in_thread")

    def resolve(self) -> "Dispatcher":
        """Resolve to a concrete Dispatcher instance.

        Lazy import inside the method so users with the `local`/`remote`
        kinds don't pay the sakura_wire import cost when only `in_thread`
        is used (e.g., test fixtures).
        """
        if self.kind == "in_thread":
            from sakura.dispatch.in_thread import InThreadDispatcher
            return InThreadDispatcher()
        if self.kind == "local":
            from sakura.dispatch.local import LocalDispatcher
            return LocalDispatcher(
                n_workers=self.n_workers,
                gpus=list(self.gpus) if self.gpus is not None else None,
            )
        if self.kind == "remote":
            raise NotImplementedError(
                "Compute.at resolution requires a TLS cert; use RemoteDispatcher "
                "directly with cert_der= for Plan 2 cross-host. Plan 4 wires this up."
            )
        raise ValueError(f"unknown Compute.kind: {self.kind!r}")

    def __repr__(self) -> str:
        if self.kind == "local":
            return f"Compute.local(n_workers={self.n_workers}, gpus={self.gpus})"
        if self.kind == "remote":
            return f"Compute(kind=remote, uris={list(self.uris)!r}, strategy={self.strategy!r})"
        return f"Compute.{self.kind}()"


__all__ = ["Compute"]
