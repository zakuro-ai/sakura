//! End-to-end test: bind a QUIC server on loopback, connect a client,
//! send a request, server echoes it back, client receives. No PyO3 yet —
//! this verifies the Rust transport in isolation.

use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, bind_server_with, connect, connect_with, generate_self_signed,
    rpc_call, send_response, FlowControlConfig,
};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::oneshot;

#[test]
fn loopback_echo_round_trip() {
    let rt = WireRuntime::shared();
    rt.block_on(async {
        let pair = generate_self_signed("localhost").expect("cert");
        let cert_for_client = pair.cert_der.clone();

        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = bind_server(server_addr, &pair).expect("bind");
        let local_addr = endpoint.local_addr().unwrap();

        // Server task: accept one connection, echo the next request bytes.
        let (server_done_tx, server_done_rx) = oneshot::channel();
        tokio::spawn(async move {
            let incoming = endpoint.accept().await.expect("accept");
            let conn = incoming.await.expect("connection");
            let (send, req_bytes) = accept_request(&conn).await.expect("accept_request");
            // Echo back identical bytes.
            send_response(send, req_bytes).await.expect("send_response");
            let _ = server_done_tx.send(());
        });

        // Client.
        let conn = connect(local_addr, "localhost", &cert_for_client)
            .await
            .expect("connect");
        let payload: Vec<u8> = (0u8..200).cycle().take(4096).collect();
        let resp = rpc_call(&conn, payload.clone()).await.expect("rpc");
        assert_eq!(resp.len(), payload.len());
        assert_eq!(resp, payload);
        // Wait for server task to finish.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server_done_rx).await;
    });
}

/// Echo `payload` over a loopback connection where both ends use `flow`.
fn echo_with_flow(flow: FlowControlConfig, payload: Vec<u8>) -> Vec<u8> {
    let rt = WireRuntime::shared();
    rt.block_on(async move {
        let pair = generate_self_signed("localhost").expect("cert");
        let cert_for_client = pair.cert_der.clone();
        let endpoint =
            bind_server_with("127.0.0.1:0".parse().unwrap(), &pair, &flow).expect("bind");
        let local_addr = endpoint.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let conn = endpoint
                .accept()
                .await
                .expect("accept")
                .await
                .expect("conn");
            let (send, req) = accept_request(&conn).await.expect("accept_request");
            send_response(send, req).await.expect("send_response");
        });

        let conn = connect_with(local_addr, "localhost", &cert_for_client, &flow)
            .await
            .expect("connect");
        let resp =
            tokio::time::timeout(std::time::Duration::from_secs(60), rpc_call(&conn, payload))
                .await
                .expect("rpc timed out")
                .expect("rpc");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
        resp
    })
}

/// SAK-0010: BDP-scaled windows (248 ms x 12.5 MB/s -> 6.2 MB stream window,
/// well above quinn's 1.25 MB default) carry a multi-window payload intact.
#[test]
fn loopback_bdp_scaled_windows_round_trip() {
    let flow = FlowControlConfig::from_rtt_bandwidth(Duration::from_millis(248), 12_500_000);
    let w = flow.windows().expect("non-default windows");
    assert_eq!(w.stream_receive_window, 6_200_000);
    let payload: Vec<u8> = (0u8..=250).cycle().take(16 * 1024 * 1024).collect();
    let resp = echo_with_flow(flow, payload.clone());
    assert_eq!(resp, payload);
}

/// SAK-0010: a window far below the payload size forces many flow-control
/// credit updates per stream; the transfer must still complete.
#[test]
fn loopback_small_window_round_trip() {
    let flow = FlowControlConfig::from_rtt_bandwidth(Duration::from_millis(1), 1)
        .with_clamps(32 * 1024, 32 * 1024)
        .expect("clamps");
    assert_eq!(flow.windows().unwrap().stream_receive_window, 32 * 1024);
    let payload: Vec<u8> = (0u8..=200).cycle().take(2 * 1024 * 1024).collect();
    let resp = echo_with_flow(flow, payload.clone());
    assert_eq!(resp, payload);
}
