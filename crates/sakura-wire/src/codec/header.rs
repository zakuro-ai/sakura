//! Postcard-encoded headers for the wire format.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WireVersion {
    V1 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    I64 = 10,
    I32 = 11,
    U8 = 12,
    Bool = 13,
}

impl Dtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F8E4M3 | Dtype::U8 | Dtype::Bool => 1,
            Dtype::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u8),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDesc {
    pub shape: SmallVec<[u32; 8]>,
    pub dtype: Dtype,
    pub n_bytes: u64,
    pub device_hint: Device,
    pub fp16_cast_on_wire: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RpcStatus {
    Ok,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequestHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub handler_id: u32,
    pub n_tensors: u32,
    pub aux_payload_bytes: u32,
    pub deadline_ms: Option<u32>,
    pub trace_id: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponseHeader {
    pub version: WireVersion,
    pub request_id: u64,
    pub status: RpcStatus,
    pub n_result_tensors: u32,
    pub aux_payload_bytes: u32,
    pub elapsed_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use postcard::{from_bytes, to_allocvec};
    use smallvec::smallvec;

    #[test]
    fn rpc_request_header_roundtrip() {
        let h = RpcRequestHeader {
            version: WireVersion::V1,
            request_id: 0xDEAD_BEEF,
            handler_id: 0xDEAD,
            n_tensors: 3,
            aux_payload_bytes: 1024,
            deadline_ms: Some(5000),
            trace_id: 0x0123_4567_89AB_CDEF_0000_1111_2222_3333,
        };
        let bytes = to_allocvec(&h).unwrap();
        let decoded: RpcRequestHeader = from_bytes(&bytes).unwrap();
        assert_eq!(decoded.request_id, h.request_id);
        assert_eq!(decoded.handler_id, h.handler_id);
        assert_eq!(decoded.n_tensors, h.n_tensors);
        assert_eq!(decoded.deadline_ms, h.deadline_ms);
        assert_eq!(decoded.trace_id, h.trace_id);
    }

    #[test]
    fn tensor_desc_roundtrip() {
        let t = TensorDesc {
            shape: smallvec![768, 3, 224, 224],
            dtype: Dtype::F32,
            n_bytes: 768 * 3 * 224 * 224 * 4,
            device_hint: Device::Cuda(0),
            fp16_cast_on_wire: false,
        };
        let bytes = to_allocvec(&t).unwrap();
        let decoded: TensorDesc = from_bytes(&bytes).unwrap();
        assert_eq!(decoded, t);
    }

    #[test]
    fn rpc_response_header_roundtrip() {
        let h = RpcResponseHeader {
            version: WireVersion::V1,
            request_id: 42,
            status: RpcStatus::Ok,
            n_result_tensors: 1,
            aux_payload_bytes: 64,
            elapsed_us: 4523,
        };
        let bytes = to_allocvec(&h).unwrap();
        let decoded: RpcResponseHeader = from_bytes(&bytes).unwrap();
        assert_eq!(decoded.request_id, h.request_id);
        assert_eq!(decoded.status, h.status);
        assert_eq!(decoded.elapsed_us, h.elapsed_us);
    }

    #[test]
    fn dtype_size_bytes() {
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F16.size_bytes(), 2);
        assert_eq!(Dtype::BF16.size_bytes(), 2);
        assert_eq!(Dtype::F8E4M3.size_bytes(), 1);
        assert_eq!(Dtype::I64.size_bytes(), 8);
        assert_eq!(Dtype::I32.size_bytes(), 4);
        assert_eq!(Dtype::U8.size_bytes(), 1);
        assert_eq!(Dtype::Bool.size_bytes(), 1);
    }
}
