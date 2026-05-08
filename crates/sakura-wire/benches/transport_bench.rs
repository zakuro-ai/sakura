use criterion::{criterion_group, criterion_main, Criterion};
use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
};
use std::net::SocketAddr;
use tokio::sync::oneshot;

fn bench_rtt(c: &mut Criterion) {
    let rt = WireRuntime::shared();

    // All quinn endpoint setup must run inside the tokio runtime.
    let (conn, stop_tx) = rt.block_on(async {
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
    });

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

criterion_group!(benches, bench_rtt);
criterion_main!(benches);
