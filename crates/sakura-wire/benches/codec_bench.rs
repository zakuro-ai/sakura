use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sakura_wire::codec::{
    pack_request, unpack_request, Device, Dtype, RpcRequestHeader, TensorView, WireVersion,
};
use smallvec::smallvec;

fn make_state_dict_buf(total_mb: usize) -> Vec<u8> {
    vec![0u8; total_mb * 1024 * 1024]
}

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_pack_268mb");
    group.throughput(Throughput::Bytes(268 * 1024 * 1024));
    let buf = make_state_dict_buf(268);
    let header = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0xDEAD,
        n_tensors: 1,
        aux_payload_bytes: 0,
        deadline_ms: None,
        trace_id: 0,
    };
    group.bench_function("pack", |b| {
        b.iter(|| {
            let view = TensorView::new(smallvec![268_435_456u32], Dtype::U8, Device::Cpu, &buf);
            let _ = pack_request(&header, &[view], &[]).unwrap();
        });
    });
    group.finish();
}

fn bench_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_unpack_268mb");
    group.throughput(Throughput::Bytes(268 * 1024 * 1024));
    let buf = make_state_dict_buf(268);
    let header = RpcRequestHeader {
        version: WireVersion::V1,
        request_id: 1,
        handler_id: 0xDEAD,
        n_tensors: 1,
        aux_payload_bytes: 0,
        deadline_ms: None,
        trace_id: 0,
    };
    let view = TensorView::new(smallvec![268_435_456u32], Dtype::U8, Device::Cpu, &buf);
    let packed = pack_request(&header, &[view], &[]).unwrap();
    group.bench_function("unpack", |b| {
        b.iter(|| {
            let _ = unpack_request(&packed).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pack, bench_unpack);
criterion_main!(benches);
