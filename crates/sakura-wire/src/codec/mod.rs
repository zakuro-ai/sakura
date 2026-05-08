//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod cast;
pub mod header;
pub mod tensor;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
pub use tensor::{OwnedTensor, TensorView};
