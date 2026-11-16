use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use quinn::Connection;
use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::oneshot;

/// Loopback QUIC echo server + one client connection. The server echoes each
/// request, so one `rpc_call` moves the payload up and back down — the shape
/// of a federation round (worker upload + leader broadcast).
fn loopback(rt: &WireRuntime) -> (Connection, oneshot::Sender<()>) {
    // All quinn endpoint setup must run inside the tokio runtime.
    rt.block_on(async {
        let pair = generate_self_signed("localhost").unwrap();
        let cert = pair.cert_der.clone();
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = bind_server(server_addr, &pair).unwrap();
        let local = endpoint.local_addr().unwrap();

        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    next = endpoint.accept() => {
                        let Some(incoming) = next else { break };
                        tokio::spawn(async move {
                            if let Ok(conn) = incoming.await {
                                while let Ok((send, req)) = accept_request(&conn).await {
                                    let _ = send_response(send, req).await;
                                }
                            }
                        });
                    }
                }
            }
        });

        let conn = connect(local, "localhost", &cert).await.unwrap();
        (conn, stop_tx)
    })
}

fn bench_rtt(c: &mut Criterion) {
    let rt = WireRuntime::shared();
    let (conn, stop_tx) = loopback(rt);

    let payload = vec![0u8; 64]; // tiny — RTT-dominated

    c.bench_function("loopback_rtt_64b", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = rpc_call(&conn, payload.clone()).await.unwrap();
            });
        });
    });

    let _ = stop_tx.send(());
}

/// Federation codec bucket (sakura/services/federation/codec.py BUCKET):
/// one fp32 scale per 4096 elements.
const BUCKET: usize = 4096;

/// Exact payload bytes of `codec_encode` for `numel` elements.
fn int8_bytes(numel: usize) -> usize {
    numel + 4 * numel.div_ceil(BUCKET)
}

fn int4_bytes(numel: usize) -> usize {
    let padded = numel.div_ceil(BUCKET) * BUCKET;
    padded / 2 + 4 * numel.div_ceil(BUCKET)
}

/// Issue #51: per-round wire payload of block mode (full int8 delta) vs
/// Streaming DiLoCo (one int4 shard of K=4). Both are echoed (up + down);
/// the time per round falls with the payload, so the peak bandwidth a round
/// needs at a fixed cadence drops by the same factor.
fn bench_round_payload(c: &mut Criterion) {
    let rt = WireRuntime::shared();
    let (conn, stop_tx) = loopback(rt);

    const NUMEL: usize = 11_689_512; // ResNet-18
    const SHARDS: usize = 4;
    let block = int8_bytes(NUMEL);
    let stream = int4_bytes(NUMEL.div_ceil(SHARDS));
    println!(
        "round payload per direction: block int8 {block} B, stream int4 K={SHARDS} {stream} B \
         ({:.2}x lower peak)",
        block as f64 / stream as f64
    );

    let mut group = c.benchmark_group("federation_round_payload");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    for (name, len) in [("block_int8_full", block), ("stream_int4_k4", stream)] {
        let payload = vec![0x5au8; len];
        group.throughput(Throughput::Bytes(2 * len as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let echoed = rpc_call(&conn, payload.clone()).await.unwrap();
                    assert_eq!(echoed.len(), len);
                });
            });
        });
    }
    group.finish();

    let _ = stop_tx.send(());
}

criterion_group!(benches, bench_rtt, bench_round_payload);
criterion_main!(benches);
