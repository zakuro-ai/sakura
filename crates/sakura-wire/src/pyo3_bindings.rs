//! PyO3 bindings: Dispatcher, Future, Result, TlsConfig, WorkerSupervisor.
//
// pyo3 0.21 generates `unsafe fn` wrappers that internally call unsafe fns;
// the crate-level `#![deny(unsafe_op_in_unsafe_fn)]` rejects this on Rust 2021
// even though the callers never use `unsafe {}` themselves.  Allow the pattern
// only for this module.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

use crate::codec::{pack_request, OwnedTensor, RpcRequestHeader, TensorView, WireVersion};
use crate::protocol::{WireError, HANDLER_ECHO};
use crate::runtime::WireRuntime;
use crate::supervisor::encode_hex;
use crate::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
    TransportError,
};

#[derive(Debug, thiserror::Error)]
enum PyWireError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
    #[error("wire error: {0}")]
    Wire(#[from] WireError),
    #[error("URI parse error: {0}")]
    UriParse(String),
    #[error("future already consumed")]
    AlreadyConsumed,
    #[error("future cancelled")]
    Cancelled,
    #[error("timeout")]
    Timeout,
    #[error("buffer protocol error: {0}")]
    #[allow(dead_code)]
    Buffer(String),
}

impl From<PyWireError> for PyErr {
    fn from(e: PyWireError) -> Self {
        match &e {
            PyWireError::Timeout => PyTimeoutError::new_err(e.to_string()),
            PyWireError::UriParse(_) | PyWireError::Buffer(_) => {
                PyValueError::new_err(e.to_string())
            }
            _ => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

#[pyclass(name = "TlsConfig")]
#[derive(Clone)]
pub struct PyTlsConfig {
    cert_der: Vec<u8>,
    server_name: String,
}

#[pymethods]
impl PyTlsConfig {
    #[new]
    fn new(cert_der: Vec<u8>, server_name: String) -> Self {
        Self {
            cert_der,
            server_name,
        }
    }
}

#[pyclass(name = "Result")]
pub struct PyRpcResult {
    #[pyo3(get)]
    pub elapsed_us: u64,
    aux: Vec<u8>,
    tensors: Vec<OwnedTensor>,
}

#[pymethods]
impl PyRpcResult {
    #[getter]
    fn aux<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.aux)
    }

    /// Returns a list of `bytes` objects (one per tensor). Plan 1: bytes-for-bytes
    /// fidelity is what callers need; Plan 2 layers numpy/torch unpacking on top.
    ///
    /// pyo3 0.24: `PyList::new` is fallible (allocation can fail under memory
    /// pressure). We propagate via PyResult so the binding raises Python-side
    /// instead of panicking.
    fn tensors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<PyBytes>> = self
            .tensors
            .iter()
            .map(|t| PyBytes::new(py, &t.bytes))
            .collect();
        PyList::new(py, items)
    }
}

#[pyclass(name = "Future")]
pub struct PyFuture {
    rx: Mutex<Option<oneshot::Receiver<Result<PyRpcResult, PyWireError>>>>,
    cancelled: Arc<AtomicBool>,
    sender_finished: Arc<AtomicBool>,
}

#[pymethods]
impl PyFuture {
    #[pyo3(signature = (timeout = None))]
    fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<PyRpcResult> {
        let rx = self
            .rx
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyErr::from(PyWireError::AlreadyConsumed))?;
        py.allow_threads(move || -> PyResult<PyRpcResult> {
            let outcome = match timeout {
                None => WireRuntime::shared().block_on(rx),
                Some(secs) => {
                    let dur = Duration::from_secs_f64(secs);
                    WireRuntime::shared().block_on(async move {
                        match tokio::time::timeout(dur, rx).await {
                            Ok(v) => v,
                            Err(_) => Ok(Err(PyWireError::Timeout)),
                        }
                    })
                }
            };
            match outcome {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(e.into()),
                Err(_) => Err(PyErr::from(PyWireError::Cancelled)),
            }
        })
    }

    fn cancel(&self) -> bool {
        let was = self.cancelled.swap(true, Ordering::AcqRel);
        !was
    }

    fn done(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.sender_finished.load(Ordering::Acquire)
    }
}

#[pyclass(name = "Dispatcher")]
pub struct PyDispatcher {
    target_uri: String,
    tls: PyTlsConfig,
}

#[pymethods]
impl PyDispatcher {
    /// Construct a dispatcher pointing at `quic://host:port`.
    #[new]
    #[pyo3(signature = (target_uri, tls))]
    fn new(target_uri: String, tls: PyTlsConfig) -> PyResult<Self> {
        if !target_uri.starts_with("quic://") {
            return Err(PyErr::from(PyWireError::UriParse(format!(
                "only quic:// is supported in v1, got: {target_uri}"
            ))));
        }
        Ok(Self { target_uri, tls })
    }

    /// Submit an RPC.
    /// `tensors` is a list of dicts `{"shape": [...], "dtype_id": int, "device_id": int, "data": bytes}`
    /// (or 4-tuples in the same order); `aux_payload` is opaque bytes.
    #[pyo3(signature = (handler_id, tensors, aux_payload, timeout_ms = None))]
    fn submit(
        &self,
        _py: Python<'_>,
        handler_id: u32,
        tensors: Vec<TensorTuple>,
        aux_payload: Vec<u8>,
        timeout_ms: Option<u32>,
    ) -> PyResult<PyFuture> {
        let target = parse_quic_uri(&self.target_uri)?;
        let cert = self.tls.cert_der.clone();
        let server_name = self.tls.server_name.clone();

        let header = RpcRequestHeader {
            version: WireVersion::V1,
            request_id: next_request_id(),
            handler_id,
            n_tensors: tensors.len() as u32,
            aux_payload_bytes: aux_payload.len() as u32,
            deadline_ms: timeout_ms,
            trace_id: 0,
        };
        let request_bytes = {
            let tensor_views: Vec<TensorView<'_>> =
                tensors.iter().map(TensorTuple::as_view).collect();
            pack_request(&header, &tensor_views, &aux_payload)
                .map_err(|e| PyRuntimeError::new_err(format!("pack: {e}")))?
        };

        let (tx, rx) = oneshot::channel::<Result<PyRpcResult, PyWireError>>();
        let cancelled = Arc::new(AtomicBool::new(false));
        let sender_finished = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);
        let sf_clone = Arc::clone(&sender_finished);

        WireRuntime::shared().spawn(async move {
            let result = run_rpc(&target, &server_name, &cert, request_bytes).await;
            sf_clone.store(true, Ordering::Release);
            if !cancelled_clone.load(Ordering::Acquire) {
                let _ = tx.send(result);
            }
        });

        Ok(PyFuture {
            rx: Mutex::new(Some(rx)),
            cancelled,
            sender_finished,
        })
    }
}

/// Minimal Python representation of one tensor.
/// Accepts either a dict `{"shape", "dtype_id", "device_id", "data"}` or a tuple `(shape, dtype_id, device_id, data)`.
struct TensorTuple {
    shape: Vec<u32>,
    dtype_id: u8,
    device_id: u8,
    data: Vec<u8>,
}

impl<'py> FromPyObject<'py> for TensorTuple {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = ob.downcast::<PyDict>() {
            let shape: Vec<u32> = dict
                .get_item("shape")?
                .ok_or_else(|| PyValueError::new_err("missing 'shape'"))?
                .extract()?;
            let dtype_id: u8 = dict
                .get_item("dtype_id")?
                .ok_or_else(|| PyValueError::new_err("missing 'dtype_id'"))?
                .extract()?;
            let device_id: u8 = dict
                .get_item("device_id")?
                .ok_or_else(|| PyValueError::new_err("missing 'device_id'"))?
                .extract()?;
            let data: Vec<u8> = dict
                .get_item("data")?
                .ok_or_else(|| PyValueError::new_err("missing 'data'"))?
                .extract()?;
            return Ok(Self {
                shape,
                dtype_id,
                device_id,
                data,
            });
        }
        let tup: (Vec<u32>, u8, u8, Vec<u8>) = ob.extract()?;
        Ok(Self {
            shape: tup.0,
            dtype_id: tup.1,
            device_id: tup.2,
            data: tup.3,
        })
    }
}

impl TensorTuple {
    fn as_view(&self) -> TensorView<'_> {
        use crate::codec::{Device, Dtype};
        let dtype = match self.dtype_id {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::BF16,
            3 => Dtype::F8E4M3,
            10 => Dtype::I64,
            11 => Dtype::I32,
            12 => Dtype::U8,
            13 => Dtype::Bool,
            _ => Dtype::U8,
        };
        let device = match self.device_id {
            0 => Device::Cpu,
            other => Device::Cuda(other - 1),
        };
        TensorView::new(self.shape.clone(), dtype, device, &self.data)
    }
}

fn parse_quic_uri(uri: &str) -> PyResult<SocketAddr> {
    let stripped = uri
        .strip_prefix("quic://")
        .ok_or_else(|| PyErr::from(PyWireError::UriParse(format!("not a quic:// uri: {uri}"))))?;
    stripped
        .parse::<SocketAddr>()
        .map_err(|e| PyErr::from(PyWireError::UriParse(format!("bad quic uri {uri}: {e}"))))
}

fn next_request_id() -> u64 {
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

async fn run_rpc(
    target: &SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
    request_bytes: Vec<u8>,
) -> Result<PyRpcResult, PyWireError> {
    let conn = connect(*target, server_name, trusted_cert).await?;
    let resp = rpc_call(&conn, request_bytes).await?;
    parse_response(&resp)
}

fn parse_response(buf: &[u8]) -> Result<PyRpcResult, PyWireError> {
    use crate::codec::{RpcResponseHeader, TensorDesc};
    let mut cursor = 0usize;
    let read_u32 = |cur: &mut usize, buf: &[u8]| -> Result<u32, PyWireError> {
        if buf.len() < *cur + 4 {
            return Err(PyWireError::Wire(WireError::DecodeFailed {
                what: "response prefix".into(),
                detail: "truncated".into(),
            }));
        }
        let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };
    let h_len = read_u32(&mut cursor, buf)? as usize;
    let header: RpcResponseHeader =
        postcard::from_bytes(&buf[cursor..cursor + h_len]).map_err(|e| {
            PyWireError::Wire(WireError::DecodeFailed {
                what: "response header".into(),
                detail: e.to_string(),
            })
        })?;
    cursor += h_len;
    let descs_len = read_u32(&mut cursor, buf)? as usize;
    let descs: Vec<TensorDesc> =
        postcard::from_bytes(&buf[cursor..cursor + descs_len]).map_err(|e| {
            PyWireError::Wire(WireError::DecodeFailed {
                what: "response descriptors".into(),
                detail: e.to_string(),
            })
        })?;
    cursor += descs_len;
    let mut tensors = Vec::with_capacity(descs.len());
    for d in descs {
        let n = d.n_bytes as usize;
        if buf.len() < cursor + n {
            return Err(PyWireError::Wire(WireError::DecodeFailed {
                what: "response tensor bytes".into(),
                detail: format!("expected {n}, got {}", buf.len() - cursor),
            }));
        }
        let bytes = buf[cursor..cursor + n].to_vec();
        cursor += n;
        tensors.push(OwnedTensor::new(d, bytes));
    }
    let aux_len = header.aux_payload_bytes as usize;
    if buf.len() < cursor + aux_len {
        return Err(PyWireError::Wire(WireError::DecodeFailed {
            what: "response aux".into(),
            detail: format!("expected {aux_len}, got {}", buf.len() - cursor),
        }));
    }
    let aux = buf[cursor..cursor + aux_len].to_vec();
    Ok(PyRpcResult {
        elapsed_us: header.elapsed_us,
        aux,
        tensors,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Task 11: WorkerSupervisor
// ────────────────────────────────────────────────────────────────────────────

use crate::supervisor::{spawn_worker, WorkerHandle};
use std::collections::HashMap;

#[pyclass(name = "WorkerSupervisor")]
pub struct PyWorkerSupervisor {
    handles: Mutex<Vec<Arc<WorkerHandle>>>,
    shutdown_timeout: Duration,
}

#[pymethods]
impl PyWorkerSupervisor {
    #[new]
    #[pyo3(signature = (shutdown_timeout_s = 30.0))]
    fn new(shutdown_timeout_s: f64) -> Self {
        Self {
            handles: Mutex::new(Vec::new()),
            shutdown_timeout: Duration::from_secs_f64(shutdown_timeout_s.max(0.1)),
        }
    }

    /// Spawn one worker.
    /// `cmd` is the argv list (e.g., `[sys.executable, "-m", "sakura.worker"]`).
    /// `env` is extra env vars (e.g., `CUDA_VISIBLE_DEVICES=1`).
    /// Returns a tuple `(uri, cert_der_bytes)` once the worker prints
    /// `SAKURA_WORKER_LISTENING <uri> <cert_hex>` on stdout.
    #[pyo3(signature = (cmd, env = None, startup_timeout_s = 10.0))]
    fn spawn(
        &self,
        py: Python<'_>,
        cmd: Vec<String>,
        env: Option<&Bound<'_, PyDict>>,
        startup_timeout_s: f64,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let mut env_map: HashMap<String, String> = HashMap::new();
        if let Some(d) = env {
            for (k, v) in d.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                env_map.insert(key, val);
            }
        }
        let timeout = Duration::from_secs_f64(startup_timeout_s.max(0.1));
        let handle = spawn_worker(&cmd, env_map, timeout)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let uri = handle.uri.clone();
        let cert = handle.cert_der.clone();
        self.handles.lock().unwrap().push(Arc::new(handle));
        let cert_py = PyBytes::new(py, &cert).unbind();
        Ok((uri, cert_py))
    }

    /// Shut down every spawned worker.
    fn shutdown(&self) {
        let handles = std::mem::take(&mut *self.handles.lock().unwrap());
        let timeout = self.shutdown_timeout;
        for h in handles {
            h.shutdown(timeout);
        }
    }

    fn __len__(&self) -> usize {
        self.handles.lock().unwrap().len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Task 12: run_echo_server — binds QUIC, prints handshake line, serves forever
// ────────────────────────────────────────────────────────────────────────────

/// Bind a QUIC server, print one handshake line on stdout, then serve the
/// echo handler forever (`HANDLER_ECHO`). Used by the `sakura-worker` daemon.
#[pyfunction]
#[pyo3(signature = (addr, print_handshake = true))]
pub fn run_echo_server(py: Python<'_>, addr: String, print_handshake: bool) -> PyResult<()> {
    py.allow_threads(|| -> PyResult<()> {
        WireRuntime::shared().block_on(async move {
            let pair = generate_self_signed("localhost")
                .map_err(|e| PyRuntimeError::new_err(format!("cert: {e}")))?;
            let bind_addr: SocketAddr = addr
                .parse()
                .map_err(|e| PyRuntimeError::new_err(format!("addr {addr}: {e}")))?;
            let endpoint = bind_server(bind_addr, &pair)
                .map_err(|e| PyRuntimeError::new_err(format!("bind: {e}")))?;
            let local = endpoint
                .local_addr()
                .map_err(|e| PyRuntimeError::new_err(format!("local_addr: {e}")))?;
            if print_handshake {
                let cert_hex = encode_hex(&pair.cert_der);
                println!("SAKURA_WORKER_LISTENING quic://{local} {cert_hex}");
                let _ = std::io::Write::flush(&mut std::io::stdout().lock());
            }

            // Accept connections forever.
            while let Some(incoming) = endpoint.accept().await {
                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => loop {
                            match accept_request(&conn).await {
                                Ok((send, req)) => {
                                    let resp = match echo_handler(&req) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            tracing::error!("echo: {e:?}");
                                            return;
                                        }
                                    };
                                    if let Err(e) = send_response(send, resp).await {
                                        tracing::error!("send_response: {e:?}");
                                        return;
                                    }
                                }
                                Err(_) => return,
                            }
                        },
                        Err(e) => tracing::error!("connection: {e:?}"),
                    }
                });
            }
            Ok(())
        })
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Task 9: run_server — dispatches every RPC to a Python callback
// ────────────────────────────────────────────────────────────────────────────

/// Bind a QUIC server, print one handshake line on stdout (if requested),
/// then dispatch every RPC to the Python `callback` until shutdown.
///
/// `callback` is invoked with three positional args:
///   - handler_id: int
///   - tensors: list[dict]   (each dict = {"shape": list[int], "dtype_id": int,
///                            "device_id": int, "data": bytes})
///   - aux_payload: bytes
/// and must return a 2-tuple `(result_tensors, result_aux_bytes)` of the same shape.
///
/// Plan 2's HandlerRegistry on the Python side is the canonical implementation;
/// run_server is the thin Rust shim.
#[pyfunction]
#[pyo3(signature = (addr, callback, print_handshake = true))]
pub fn run_server(
    py: Python<'_>,
    addr: String,
    callback: PyObject,
    print_handshake: bool,
) -> PyResult<()> {
    let cb = std::sync::Arc::new(callback);
    py.allow_threads(|| -> PyResult<()> {
        WireRuntime::shared().block_on(async move {
            let pair = generate_self_signed("localhost")
                .map_err(|e| PyRuntimeError::new_err(format!("cert: {e}")))?;
            let bind_addr: SocketAddr = addr
                .parse()
                .map_err(|e| PyRuntimeError::new_err(format!("addr {addr}: {e}")))?;
            let endpoint = bind_server(bind_addr, &pair)
                .map_err(|e| PyRuntimeError::new_err(format!("bind: {e}")))?;
            let local = endpoint
                .local_addr()
                .map_err(|e| PyRuntimeError::new_err(format!("local_addr: {e}")))?;
            if print_handshake {
                let cert_hex = encode_hex(&pair.cert_der);
                println!("SAKURA_WORKER_LISTENING quic://{local} {cert_hex}");
                let _ = std::io::Write::flush(&mut std::io::stdout().lock());
            }

            while let Some(incoming) = endpoint.accept().await {
                let cb = std::sync::Arc::clone(&cb);
                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => loop {
                            let cb = std::sync::Arc::clone(&cb);
                            match accept_request(&conn).await {
                                Ok((send, req)) => {
                                    let resp = match dispatch_via_callback(&cb, &req) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            tracing::error!("dispatch: {e:?}");
                                            return;
                                        }
                                    };
                                    if let Err(e) = send_response(send, resp).await {
                                        tracing::error!("send_response: {e:?}");
                                        return;
                                    }
                                }
                                Err(_) => return,
                            }
                        },
                        Err(e) => tracing::error!("connection: {e:?}"),
                    }
                });
            }
            Ok(())
        })
    })
}

/// Decode an RPC request, invoke the Python callback under the GIL, and
/// re-encode its response.
fn dispatch_via_callback(callback: &PyObject, req_bytes: &[u8]) -> Result<Vec<u8>, PyWireError> {
    use crate::codec::{unpack_request, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion};
    let (header, descs, tensors, aux) = unpack_request(req_bytes).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "server unpack".into(),
            detail: e.to_string(),
        })
    })?;

    let (out_tensors_descs, out_tensors_bytes, out_aux): (Vec<TensorDesc>, Vec<Vec<u8>>, Vec<u8>) =
        Python::with_gil(|py| -> PyResult<_> {
            // Build the tensors list as Python list[dict].
            let py_tensors = PyList::empty(py);
            for (desc, bytes) in descs.iter().zip(tensors.iter()) {
                let d = PyDict::new(py);
                d.set_item("shape", desc.shape.iter().copied().collect::<Vec<u32>>())?;
                d.set_item("dtype_id", desc.dtype as u8)?;
                d.set_item(
                    "device_id",
                    match desc.device_hint {
                        crate::codec::Device::Cpu => 0u8,
                        crate::codec::Device::Cuda(i) => i + 1,
                    },
                )?;
                d.set_item("data", PyBytes::new(py, bytes))?;
                py_tensors.append(d)?;
            }
            let py_aux = PyBytes::new(py, &aux);
            let result = callback.call1(py, (header.handler_id, py_tensors, py_aux))?;
            // Result is expected to be (list[dict], bytes).
            let tup: (Vec<TensorTuple>, Vec<u8>) = result.extract(py)?;
            let out_descs: Vec<TensorDesc> = tup
                .0
                .iter()
                .map(|t| {
                    use crate::codec::{Device, Dtype};
                    let dtype = match t.dtype_id {
                        0 => Dtype::F32,
                        1 => Dtype::F16,
                        2 => Dtype::BF16,
                        3 => Dtype::F8E4M3,
                        10 => Dtype::I64,
                        11 => Dtype::I32,
                        12 => Dtype::U8,
                        13 => Dtype::Bool,
                        _ => Dtype::U8,
                    };
                    let device = match t.device_id {
                        0 => Device::Cpu,
                        other => Device::Cuda(other - 1),
                    };
                    TensorDesc {
                        shape: t.shape.clone().into(),
                        dtype,
                        n_bytes: t.data.len() as u64,
                        device_hint: device,
                        fp16_cast_on_wire: false,
                    }
                })
                .collect();
            let bytes: Vec<Vec<u8>> = tup.0.into_iter().map(|t| t.data).collect();
            Ok((out_descs, bytes, tup.1))
        })
        .map_err(|e: PyErr| {
            PyWireError::Wire(WireError::HandlerPanic {
                msg: e.to_string(),
                trace: vec![],
            })
        })?;

    // Encode the response.
    let resp_header = RpcResponseHeader {
        version: WireVersion::V1,
        request_id: header.request_id,
        status: RpcStatus::Ok,
        n_result_tensors: out_tensors_descs.len() as u32,
        aux_payload_bytes: out_aux.len() as u32,
        elapsed_us: 0,
    };
    let header_bytes = postcard::to_allocvec(&resp_header).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp header".into(),
            detail: e.to_string(),
        })
    })?;
    let descs_bytes = postcard::to_allocvec(&out_tensors_descs).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp descs".into(),
            detail: e.to_string(),
        })
    })?;
    let total = 4
        + header_bytes.len()
        + 4
        + descs_bytes.len()
        + out_tensors_bytes.iter().map(Vec::len).sum::<usize>()
        + out_aux.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in &out_tensors_bytes {
        out.extend_from_slice(t);
    }
    out.extend_from_slice(&out_aux);
    Ok(out)
}

fn echo_handler(req_bytes: &[u8]) -> Result<Vec<u8>, PyWireError> {
    use crate::codec::{unpack_request, RpcResponseHeader, RpcStatus, WireVersion};
    let (header, descs, tensors, aux) = unpack_request(req_bytes).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "echo unpack".into(),
            detail: e.to_string(),
        })
    })?;
    if header.handler_id != HANDLER_ECHO {
        return Err(PyWireError::Wire(WireError::HandlerNotFound {
            handler_id: header.handler_id,
        }));
    }

    // Build the response: same descriptors + tensor bytes, identical aux.
    let resp_header = RpcResponseHeader {
        version: WireVersion::V1,
        request_id: header.request_id,
        status: RpcStatus::Ok,
        n_result_tensors: descs.len() as u32,
        aux_payload_bytes: aux.len() as u32,
        elapsed_us: 0,
    };
    let header_bytes = postcard::to_allocvec(&resp_header).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp header".into(),
            detail: e.to_string(),
        })
    })?;
    let descs_bytes = postcard::to_allocvec(&descs).map_err(|e| {
        PyWireError::Wire(WireError::DecodeFailed {
            what: "encode resp descs".into(),
            detail: e.to_string(),
        })
    })?;
    let total = 4
        + header_bytes.len()
        + 4
        + descs_bytes.len()
        + tensors.iter().map(Vec::len).sum::<usize>()
        + aux.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in &tensors {
        out.extend_from_slice(t);
    }
    out.extend_from_slice(&aux);
    Ok(out)
}
