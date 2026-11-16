"""RemoteDispatcher — Python wrapper over sakura_wire.Dispatcher.

Talks to an *already-running* sakura-worker at a known quic:// URI with a
known self-signed cert.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import cloudpickle  # type: ignore[import-untyped]
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sakura_wire as _native

from sakura.dispatch import result_codec, signing
from sakura.dispatch.base import Dispatcher, Future, Result

HANDLER_EXEC_CLOUDPICKLED = 0x0001


class _WireFuture(Future):
    """Wraps a sakura_wire.Future and decodes the worker's result payload.

    `wire_result.aux` came back over the wire from the worker, i.e. from a
    network peer. It is an HMAC-signed envelope (see `sakura.dispatch.signing`)
    whose payload is, by default, in the typed result codec
    (`sakura.dispatch.result_codec`), which never unpickles. Only a payload
    tagged `PICKLE_MAGIC` (a return value the typed codec cannot represent) is
    unpickled, through `signing.safe_loads` (signature verified first), and
    only when `allow_pickle_results` is true; otherwise it is refused. The
    envelope must be a result bound to this call's request nonce, fresh and
    not seen before (replay protection, sakura#139).
    """

    def __init__(
        self,
        native_future: "_native.Future",
        *,
        request_nonce: bytes,
        hmac_key: Optional[bytes] = None,
        allow_pickle_results: bool = True,
    ):
        self._fut = native_future
        self._request_nonce = request_nonce
        self._hmac_key = hmac_key
        self._allow_pickle_results = allow_pickle_results
        self._outcome: Optional[Result] = None
        self._error: Optional[Exception] = None

    def result(self, timeout: Optional[float] = None) -> Result:
        # The result envelope is single-use (replay cache), so decode it once
        # and hand the same outcome (value or error) to later calls.
        if self._outcome is None and self._error is None:
            wire_result = self._fut.result(timeout=timeout)
            try:
                self._outcome = self._decode(wire_result)
            except Exception as exc:
                self._error = exc
        if self._error is not None:
            raise self._error
        assert self._outcome is not None
        return self._outcome

    def _decode(self, wire_result: Any) -> Result:
        # Bound to this call's request nonce: a result captured from another
        # call (or replayed) is refused (sakura#139).
        verified = signing.verify_envelope(
            wire_result.aux,
            key=self._hmac_key,
            kind=signing.KIND_RESULT,
            bound_to=self._request_nonce,
        )
        payload = verified.payload
        magic = payload[: len(result_codec.TYPED_MAGIC)]
        if magic == result_codec.TYPED_MAGIC:
            # Raises RemoteExecutionError if the callable raised on the worker.
            value = result_codec.decode(payload)
        elif magic == result_codec.PICKLE_MAGIC:
            if not self._allow_pickle_results:
                raise result_codec.ResultCodecError(
                    "worker returned a cloudpickled result but this dispatcher was "
                    "built with allow_pickle_results=False; return a value the typed "
                    "result codec supports (see sakura.dispatch.result_codec)"
                )
            value = signing.safe_loads(verified, prefix=result_codec.PICKLE_MAGIC)
        else:
            raise result_codec.ResultCodecError(
                "unknown result format from worker (worker and dispatcher must run "
                "the same sakura version)"
            )
        return Result(value=value, elapsed_us=wire_result.elapsed_us)

    def done(self) -> bool:
        return bool(self._fut.done())

    def cancel(self) -> bool:
        return bool(self._fut.cancel())


def _array_to_tensor_dict(arr: np.ndarray[Any, np.dtype[Any]]) -> dict[str, Any]:
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

    def __init__(
        self,
        *,
        uri: str,
        cert_der: bytes,
        server_name: str = "localhost",
        hmac_key: Optional[bytes] = None,
        allow_pickle_results: bool = True,
        flow_control: Optional["_native.FlowControl"] = None,
    ):
        """`flow_control` is an optional `sakura.wire.FlowControl` sizing this
        client's QUIC windows (SAK-0010); None keeps quinn's defaults. Windows
        are per endpoint: configure the worker too (`sakura-worker --rtt-ms
        --bandwidth-bytes-per-s`)."""
        if not uri.startswith("quic://"):
            raise ValueError(f"RemoteDispatcher requires quic:// URI, got: {uri}")
        self._uri = uri
        # HMAC key shared with the worker at the other end of `uri`, used to
        # sign/verify aux payloads before cloudpickle ever touches them. If
        # None, falls back to the SAKURA_WIRE_HMAC_KEY env var at call time
        # (see sakura.dispatch.signing).
        self._hmac_key = hmac_key
        # Results arrive in the typed (non-pickle) codec. A value it cannot
        # represent comes back as signed cloudpickle; False refuses those, so
        # this process never unpickles anything a worker sends (SAK-0016).
        self._allow_pickle_results = allow_pickle_results
        # Lazy import: native wire loads on first dispatcher construction so
        # `import sakura` works without the Rust extension installed.
        import sakura_wire as _native
        tls = _native.TlsConfig(cert_der, server_name)
        self._dispatcher = _native.Dispatcher(uri, tls, flow_control=flow_control)

    def submit(
        self,
        callable: Callable[..., Any],
        *args: Any,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Future:
        """Cloudpickle the callable + non-tensor args; ship tensor args as raw bytes."""
        tensor_args: list[np.ndarray[Any, np.dtype[Any]]] = []
        non_tensor_args: list[Any] = []
        for a in args:
            if isinstance(a, np.ndarray):
                tensor_args.append(a)
            else:
                non_tensor_args.append(a)
        spec = {"fn": callable, "args": tuple(non_tensor_args), "kwargs": kwargs}
        aux = signing.sign(cloudpickle.dumps(spec), key=self._hmac_key)
        tensor_dicts = [_array_to_tensor_dict(a) for a in tensor_args]
        native_fut = self._dispatcher.submit(
            HANDLER_EXEC_CLOUDPICKLED, tensor_dicts, aux, timeout_ms
        )
        return _WireFuture(
            native_fut,
            request_nonce=signing.nonce_of(aux),
            hmac_key=self._hmac_key,
            allow_pickle_results=self._allow_pickle_results,
        )

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        # The native Dispatcher manages its own connection lifecycle.
        pass

    def stats(self) -> dict[str, Any]:
        return {"kind": "remote", "uri": self._uri}


__all__ = ["RemoteDispatcher"]
