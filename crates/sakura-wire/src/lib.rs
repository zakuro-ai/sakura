//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod codec;
pub mod protocol;
pub mod pyo3_bindings;
pub mod runtime;
pub mod supervisor;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    m.add_class::<pyo3_bindings::PyWorkerSupervisor>()?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::run_echo_server, m)?)?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::run_server, m)?)?;
    Ok(())
}
