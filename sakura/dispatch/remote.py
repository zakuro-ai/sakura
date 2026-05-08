"""RemoteDispatcher — Python wrapper over sakura_wire.Dispatcher.

Talks to an *already-running* sakura-worker at a known quic:// URI with a
known self-signed cert.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import cloudpickle
import numpy as np
import sakura_wire as _native

from sakura.dispatch.base import Dispatcher, Future, Result

HANDLER_EXEC_CLOUDPICKLED = 0x0001


class _WireFuture(Future):
    """Wraps a sakura_wire.Future and decodes the cloudpickled return value."""

    def __init__(self, native_future: _native.Future):
        self._fut = native_future

    def result(self, timeout: Optional[float] = None) -> Result:
        wire_result = self._fut.result(timeout=timeout)
        value = cloudpickle.loads(wire_result.aux)
        if isinstance(value, BaseException):
            raise value
        return Result(value=value, elapsed_us=wire_result.elapsed_us)

    def done(self) -> bool:
        return self._fut.done()

    def cancel(self) -> bool:
        return self._fut.cancel()


def _array_to_tensor_dict(arr: np.ndarray) -> dict:
    """Pack a numpy array into the tensor-dict shape sakura-wire expects."""
    if arr.dtype == np.float32:
        dtype_id = 0
    elif arr.dtype == np.float16:
        dtype_id = 1
    elif arr.dtype == np.int64:
        dtype_id = 10
    elif arr.dtype == np.int32:
        dtype_id = 11
    elif arr.dtype == np.bool_:
        dtype_id = 13
    else:
        dtype_id = 12  # U8 fallback
    return {
        "shape": [int(d) for d in arr.shape],
        "dtype_id": dtype_id,
        "device_id": 0,  # CPU
        "data": bytes(arr.tobytes()),
    }


class RemoteDispatcher(Dispatcher):
    """Connect to a running sakura-worker at `uri` (quic://host:port)."""

    def __init__(self, *, uri: str, cert_der: bytes, server_name: str = "localhost"):
        if not uri.startswith("quic://"):
            raise ValueError(f"RemoteDispatcher requires quic:// URI, got: {uri}")
        self._uri = uri
        tls = _native.TlsConfig(cert_der, server_name)
        self._dispatcher = _native.Dispatcher(uri, tls)

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        """Cloudpickle the callable + non-tensor args; ship tensor args as raw bytes."""
        tensor_args: list[np.ndarray] = []
        non_tensor_args: list[Any] = []
        for a in args:
            if isinstance(a, np.ndarray):
                tensor_args.append(a)
            else:
                non_tensor_args.append(a)
        spec = {"fn": callable, "args": tuple(non_tensor_args), "kwargs": kwargs}
        aux = cloudpickle.dumps(spec)
        tensor_dicts = [_array_to_tensor_dict(a) for a in tensor_args]
        native_fut = self._dispatcher.submit(
            HANDLER_EXEC_CLOUDPICKLED, tensor_dicts, aux, timeout_ms
        )
        return _WireFuture(native_fut)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        # The native Dispatcher manages its own connection lifecycle.
        pass

    def stats(self) -> dict:
        return {"kind": "remote", "uri": self._uri}


__all__ = ["RemoteDispatcher"]
