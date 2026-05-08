//! sakura-wire: zero-copy tensor codec + RPC over QUIC.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

pub mod codec;
pub mod protocol;
pub mod runtime;

#[pymodule]
fn sakura_wire(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
