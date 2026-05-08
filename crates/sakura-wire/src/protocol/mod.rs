//! RPC protocol layer: handler IDs, error types, request/response framing.

pub mod error;
pub mod handlers;

pub use error::WireError;
pub use handlers::{
    HANDLER_CUSTOM_BASE, HANDLER_ECHO, HANDLER_EXEC_CLOUDPICKLED, HANDLER_HEARTBEAT,
    HANDLER_MODEL_CACHE_GET, HANDLER_SAVE_BLOB, HANDLER_SHUTDOWN,
};
