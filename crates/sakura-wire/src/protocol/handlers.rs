//! Well-known handler IDs.

pub const HANDLER_EXEC_CLOUDPICKLED: u32 = 0x0001;
pub const HANDLER_MODEL_CACHE_GET:   u32 = 0x0002;
pub const HANDLER_HEARTBEAT:         u32 = 0x0003;
pub const HANDLER_SHUTDOWN:          u32 = 0x0004;
pub const HANDLER_SAVE_BLOB:         u32 = 0x0005;

/// First handler ID available for user-registered handlers.
pub const HANDLER_CUSTOM_BASE: u32 = 0x1000;

/// Echo handler used in tests: the worker bounces all input tensors back
/// (with byte-identical content) and an empty aux payload.
pub const HANDLER_ECHO: u32 = 0xDEAD;
