//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod pyo3_bindings;
pub mod runtime;
pub mod transport;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyo3_bindings::PyDispatcher>()?;
    m.add_class::<pyo3_bindings::PyFuture>()?;
    m.add_class::<pyo3_bindings::PyRpcResult>()?;
    m.add_class::<pyo3_bindings::PyTlsConfig>()?;
    Ok(())
}
