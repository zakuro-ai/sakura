//! Integration test: pack a request (header + descriptors + payloads + aux)
//! into a Vec<u8> and unpack it back. Verifies the public codec facade.

use sakura_wire::codec::{
    pack_request, unpack_request, Device, Dtype, RpcRequestHeader, TensorView, WireVersion,
};
use smallvec::smallvec;

#[test]
fn pack_then_unpack_preserves_request() {
    let h = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 7,
        handler_id: 0xDEAD,
        n_tensors: 2,
        aux_payload_bytes: 5,
        deadline_ms: None,
        trace_id: 0,
    };

    // Two tensors with distinguishable byte patterns.
    let t1_bytes: Vec<u8> = (0u8..16).collect();
    let t2_bytes: Vec<u8> = (32u8..40).collect();
    let t1 = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &t1_bytes);
    let t2 = TensorView::new(smallvec![2u32], Dtype::F32, Device::Cpu, &t2_bytes);
    let aux = b"hello";

    let packed = pack_request(&h, &[t1, t2], aux).expect("pack");
    let (got_header, got_descs, got_tensor_bytes, got_aux) =
        unpack_request(&packed).expect("unpack");

    assert_eq!(got_header.request_id, h.request_id);
    assert_eq!(got_header.handler_id, h.handler_id);
    assert_eq!(got_descs.len(), 2);
    assert_eq!(got_tensor_bytes.len(), 2);
    assert_eq!(got_tensor_bytes[0], t1_bytes);
    assert_eq!(got_tensor_bytes[1], t2_bytes);
    assert_eq!(got_aux, aux);
}

#[test]
fn pack_request_with_zero_tensors_works() {
    let h = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0x0001,
        n_tensors: 0,
        aux_payload_bytes: 4,
        deadline_ms: Some(1000),
        trace_id: 42,
    };
    let aux = b"NOOP";
    let packed = pack_request(&h, &[], aux).unwrap();
    let (got_h, descs, tensors, got_aux) = unpack_request(&packed).unwrap();
    assert_eq!(got_h.handler_id, 0x0001);
    assert!(descs.is_empty());
    assert!(tensors.is_empty());
    assert_eq!(got_aux, aux);
}
