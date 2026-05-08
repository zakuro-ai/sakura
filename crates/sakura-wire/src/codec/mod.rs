//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod header;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
