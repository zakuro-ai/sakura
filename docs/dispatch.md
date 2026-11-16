# Dispatching work to a sakura-worker

How `LocalDispatcher` and `RemoteDispatcher` talk to a `sakura-worker`: what is trusted, how keys work, and how to size QUIC windows on long links. For everything else, see the [Sakura documentation](https://docs.zakuro-ai.com/sakura/overview).

## QUIC flow-control windows on long-RTT links

quinn's default per-stream receive window (1.25 MB) caps one QUIC stream at roughly `1.25 MB / RTT` (about 5 MB/s at 248 ms), whatever the link bandwidth. The `sakura-wire` QUIC endpoints accept an optional `FlowControl` that sizes the windows from the bandwidth-delay product:

```python
from sakura.wire import Dispatcher, FlowControl, TlsConfig

fc = FlowControl(rtt_ms=248, bandwidth_bytes_per_s=12_500_000)  # bytes/s, not bits
fc.windows()  # {'stream_receive_window': 6200000, 'receive_window': 49600000, 'send_window': 49600000}
d = Dispatcher("quic://host:port", TlsConfig(cert_der, "localhost"), flow_control=fc)
```

- `stream_receive_window = clamp(2 x rtt x bandwidth, min_window, max_window)`; the connection receive and send windows are 8x that. Clamps default to 1.25 MB (quinn's default, so a window is never shrunk) and 128 MiB; override with `min_window=` / `max_window=`.
- `FlowControl()` or omitting `flow_control` keeps quinn's defaults, so existing callers are unchanged.
- Windows are per endpoint: set them on both the client (`Dispatcher(..., flow_control=)`) and the worker (`sakura-worker --rtt-ms 248 --bandwidth-bytes-per-s 12500000`, or `run_server(..., flow_control=)`), since each side only controls what it advertises.
- The Python dispatchers take the same object: `RemoteDispatcher(uri=..., cert_der=..., flow_control=fc)` configures the client end (configure the worker yourself), and `LocalDispatcher(flow_control=fc)` configures both ends, forwarding `--rtt-ms --bandwidth-bytes-per-s --min-window --max-window` to the worker it spawns. `sakura-worker --min-window/--max-window` (bytes) override the window clamps.
- In Rust: `FlowControlConfig::from_rtt_bandwidth(rtt, bw)` with `bind_server_with` / `connect_with`.

## Dispatch trust model

`LocalDispatcher` and `RemoteDispatcher` talk to a `sakura-worker` over QUIC (`sakura_wire`). Every payload is HMAC-SHA256-signed, and the receiver verifies the signature before it decodes anything. A missing, short (< 16 bytes) or wrong key, an unsigned payload or a tampered one is rejected.

- **Requests** carry your callable as signed cloudpickle. Anyone holding the key can run code on the worker, so treat the key like a deploy credential.
- **Results** come back in a typed, non-pickle codec: builtins, containers and numpy arrays and scalars. An exception in the callable is raised as `RemoteExecutionError`. Any other return type falls back to signed cloudpickle. Pass `allow_pickle_results=False` to refuse that fallback, so the dispatcher never unpickles anything a worker sends.
- **Replays:** each payload carries a random nonce and its issue time under the signature. A receiver refuses a payload it has already accepted, one issued more than 120 s ago (`SAKURA_WIRE_REPLAY_WINDOW_S`) or more than 30 s in the future (`SAKURA_WIRE_CLOCK_SKEW_S`), and a result that answers a different request. Keep dispatcher and worker clocks in sync (NTP). Dispatchers and workers must run the same sakura version.
- **Keys:** `LocalDispatcher` generates a per-instance key and passes it only to the worker it spawns. For a remote worker, set the same `SAKURA_WIRE_HMAC_KEY` (hex, e.g. `openssl rand -hex 32`) on the worker and on the dispatcher, or pass `hmac_key=`.

```python
dispatcher = RemoteDispatcher(uri="quic://eval-host:4433", cert_der=cert_bytes,
                              hmac_key=bytes.fromhex(key_hex), allow_pickle_results=False)
```
