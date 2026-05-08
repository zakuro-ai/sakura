//! QUIC transport via quinn. Self-signed TLS for loopback by default.

use quinn::{Connection, Endpoint, ServerConfig, TransportConfig};
use rcgen::generate_simple_self_signed;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::protocol::WireError;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("quinn: {0}")]
    Quinn(String),
    #[error("rcgen: {0}")]
    Rcgen(String),
    #[error("rustls: {0}")]
    Rustls(String),
    #[error("connect: {0}")]
    Connect(String),
    #[error("read: {0}")]
    Read(String),
    #[error("write: {0}")]
    Write(String),
    #[error("wire: {0}")]
    Wire(#[from] WireError),
}

impl From<rcgen::Error> for TransportError {
    fn from(e: rcgen::Error) -> Self {
        TransportError::Rcgen(e.to_string())
    }
}

impl From<rustls::Error> for TransportError {
    fn from(e: rustls::Error) -> Self {
        TransportError::Rustls(e.to_string())
    }
}

impl From<quinn::ConnectionError> for TransportError {
    fn from(e: quinn::ConnectionError) -> Self {
        TransportError::Quinn(e.to_string())
    }
}

impl From<quinn::ConnectError> for TransportError {
    fn from(e: quinn::ConnectError) -> Self {
        TransportError::Connect(e.to_string())
    }
}

impl From<quinn::ReadExactError> for TransportError {
    fn from(e: quinn::ReadExactError) -> Self {
        TransportError::Read(e.to_string())
    }
}

impl From<quinn::WriteError> for TransportError {
    fn from(e: quinn::WriteError) -> Self {
        TransportError::Write(e.to_string())
    }
}

/// A self-signed TLS pair (DER-encoded cert + key) for loopback testing.
#[derive(Clone)]
pub struct SelfSignedPair {
    pub cert_der: Vec<u8>,
    pub key_der: Vec<u8>,
}

pub fn generate_self_signed(subject: &str) -> Result<SelfSignedPair, TransportError> {
    let cert = generate_simple_self_signed(vec![subject.into()])?;
    Ok(SelfSignedPair {
        cert_der: cert.serialize_der()?,
        key_der: cert.serialize_private_key_der(),
    })
}

fn server_config(pair: &SelfSignedPair) -> Result<ServerConfig, TransportError> {
    let cert_chain = vec![rustls::Certificate(pair.cert_der.clone())];
    let key = rustls::PrivateKey(pair.key_der.clone());
    let mut cfg = ServerConfig::with_single_cert(cert_chain, key)
        .map_err(|e| TransportError::Rustls(e.to_string()))?;
    let mut transport = TransportConfig::default();
    transport.max_concurrent_uni_streams(0u8.into());
    cfg.transport = Arc::new(transport);
    Ok(cfg)
}

fn client_config(trusted_cert: &[u8]) -> Result<quinn::ClientConfig, TransportError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(&rustls::Certificate(trusted_cert.to_vec()))?;
    let crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let mut cfg = quinn::ClientConfig::new(Arc::new(crypto));
    let mut transport = TransportConfig::default();
    transport.max_concurrent_uni_streams(0u8.into());
    cfg.transport_config(Arc::new(transport));
    Ok(cfg)
}

/// Bind a QUIC server to `bind_addr`, returning the actual address it listens on.
pub fn bind_server(
    bind_addr: SocketAddr,
    pair: &SelfSignedPair,
) -> Result<Endpoint, TransportError> {
    let cfg = server_config(pair)?;
    let endpoint = Endpoint::server(cfg, bind_addr)?;
    Ok(endpoint)
}

/// Connect to a server with a known cert (loopback / pinned-cert use case).
pub async fn connect(
    server_addr: SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
) -> Result<Connection, TransportError> {
    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config(trusted_cert)?);
    let conn = endpoint.connect(server_addr, server_name)?.await?;
    Ok(conn)
}

/// Open a bidirectional stream, write the request bytes, signal end-of-write,
/// then read the entire response and return it.
pub async fn rpc_call(
    conn: &Connection,
    request_bytes: Vec<u8>,
) -> Result<Vec<u8>, TransportError> {
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&request_bytes).await?;
    send.finish()
        .await
        .map_err(|e| TransportError::Write(e.to_string()))?;
    let resp = recv
        .read_to_end(64 * 1024 * 1024 * 1024)
        .await
        .map_err(|e| TransportError::Read(e.to_string()))?;
    Ok(resp)
}

/// Server side: accept the next bidirectional stream and read the entire request.
pub async fn accept_request(
    conn: &Connection,
) -> Result<(quinn::SendStream, Vec<u8>), TransportError> {
    let (send, mut recv) = conn
        .accept_bi()
        .await
        .map_err(|e| TransportError::Quinn(e.to_string()))?;
    let req = recv
        .read_to_end(64 * 1024 * 1024 * 1024)
        .await
        .map_err(|e| TransportError::Read(e.to_string()))?;
    Ok((send, req))
}

/// Server side: write a response and finish the stream.
pub async fn send_response(
    mut send: quinn::SendStream,
    bytes: Vec<u8>,
) -> Result<(), TransportError> {
    send.write_all(&bytes).await?;
    send.finish()
        .await
        .map_err(|e| TransportError::Write(e.to_string()))?;
    Ok(())
}
