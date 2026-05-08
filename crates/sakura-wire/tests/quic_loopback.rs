//! End-to-end test: bind a QUIC server on loopback, connect a client,
//! send a request, server echoes it back, client receives. No PyO3 yet —
//! this verifies the Rust transport in isolation.

use sakura_wire::runtime::WireRuntime;
use sakura_wire::transport::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
};
use std::net::SocketAddr;
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
