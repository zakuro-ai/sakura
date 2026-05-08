//! Wire-level error types. Maps to Python exceptions in pyo3_bindings.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum WireError {
    #[error("handler {handler_id:#x} not found")]
    HandlerNotFound { handler_id: u32 },

    #[error("decode failed in {what}: {detail}")]
    DecodeFailed { what: String, detail: String },

    #[error("handler panicked: {msg}")]
    HandlerPanic { msg: String, trace: Vec<String> },

    #[error("timeout after {deadline_ms}ms")]
    Timeout { deadline_ms: u32 },

    #[error("worker crashed")]
    WorkerCrashed,

    #[error("backpressure saturated")]
    BackpressureSaturated,
}

#[cfg(test)]
mod tests {
    use super::*;
    use postcard::{from_bytes, to_allocvec};

    #[test]
    fn wire_error_roundtrip() {
        let cases = vec![
            WireError::HandlerNotFound { handler_id: 0xCAFE },
            WireError::DecodeFailed {
                what: "header".into(),
                detail: "bad varint".into(),
            },
            WireError::HandlerPanic {
                msg: "boom".into(),
                trace: vec!["frame_a".into(), "frame_b".into()],
            },
            WireError::Timeout { deadline_ms: 5000 },
            WireError::WorkerCrashed,
            WireError::BackpressureSaturated,
        ];
        for original in cases {
            let bytes = to_allocvec(&original).unwrap();
            let decoded: WireError = from_bytes(&bytes).unwrap();
            // round-trip via Display + variant equality (Display gives a stable string)
            assert_eq!(decoded.to_string(), original.to_string());
        }
    }
}
