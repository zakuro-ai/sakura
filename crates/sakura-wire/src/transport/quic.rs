//! QUIC transport via quinn 0.11 + rustls 0.23. Self-signed TLS for loopback by default.

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{Connection, Endpoint, ServerConfig, TransportConfig, VarInt};
use rcgen::generate_simple_self_signed;
use rustls_pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

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
    #[error("config: {0}")]
    Config(String),
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

// ────────────────────────────────────────────────────────────────────────────
// Flow-control windows (SAK-0010)
// ────────────────────────────────────────────────────────────────────────────

/// quinn 0.11's default per-stream receive window: 12.5 MB/s x 100 ms = 1.25 MB.
/// At 248 ms RTT this caps one stream at about 5 MB/s (SAK-0010 / E7).
pub const QUINN_DEFAULT_STREAM_WINDOW: u64 = 1_250_000;
/// Default lower clamp for a BDP-scaled window: never shrink below quinn's default.
pub const DEFAULT_MIN_WINDOW: u64 = QUINN_DEFAULT_STREAM_WINDOW;
/// Default upper clamp for a BDP-scaled window (per stream): 128 MiB of buffering.
pub const DEFAULT_MAX_WINDOW: u64 = 128 * 1024 * 1024;
/// Headroom multiplier on the raw BDP, so the window is not the bottleneck
/// while the congestion controller probes above the estimate or recovers a loss.
pub const BDP_HEADROOM: u64 = 2;
/// Connection-level receive and send windows are this multiple of the stream
/// window (the same ratio quinn uses between its default stream and send windows).
pub const CONNECTION_WINDOW_STREAMS: u64 = 8;

/// A link estimate: round-trip time and bottleneck bandwidth.
///
/// Today this comes from configuration (the operator states the path); the
/// SAK-0013 link probe is expected to fill the same struct from a measurement
/// and hand it to [`FlowControlConfig::from_link`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkEstimate {
    pub rtt: Duration,
    /// Bottleneck bandwidth in bytes per second (not bits).
    pub bandwidth_bytes_per_s: u64,
}

impl LinkEstimate {
    /// Bandwidth-delay product in bytes (saturating).
    pub fn bdp_bytes(&self) -> u64 {
        let bdp = (self.bandwidth_bytes_per_s as u128) * self.rtt.as_micros() / 1_000_000;
        u64::try_from(bdp).unwrap_or(u64::MAX)
    }
}

/// The three quinn flow-control knobs, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowWindows {
    /// Per-stream receive window (`TransportConfig::stream_receive_window`).
    pub stream_receive_window: u64,
    /// Connection-level receive window (`TransportConfig::receive_window`).
    pub receive_window: u64,
    /// Connection-level send window (`TransportConfig::send_window`).
    pub send_window: u64,
}

/// Flow-control input for [`bind_server_with`] / [`connect_with`].
///
/// `FlowControlConfig::default()` (no link estimate) leaves quinn's defaults
/// untouched, so existing callers see no change. With a [`LinkEstimate`] the
/// windows are sized as:
///
/// ```text
/// stream_receive_window = clamp(BDP_HEADROOM x rtt x bandwidth, min_window, max_window)
/// receive_window = send_window = CONNECTION_WINDOW_STREAMS x stream_receive_window
/// ```
///
/// Each side only controls what it *advertises* (receive windows) and what it
/// is willing to have in flight (send window), so both endpoints need the
/// config for a bulk transfer in either direction to benefit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowControlConfig {
    /// `None` = keep quinn's defaults.
    pub link: Option<LinkEstimate>,
    pub min_window: u64,
    pub max_window: u64,
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            link: None,
            min_window: DEFAULT_MIN_WINDOW,
            max_window: DEFAULT_MAX_WINDOW,
        }
    }
}

impl FlowControlConfig {
    /// BDP-scaled windows for `link`, with the default clamps.
    pub fn from_link(link: LinkEstimate) -> Self {
        Self {
            link: Some(link),
            ..Self::default()
        }
    }

    /// Convenience: `from_link` from an RTT and a bandwidth in bytes/s.
    pub fn from_rtt_bandwidth(rtt: Duration, bandwidth_bytes_per_s: u64) -> Self {
        Self::from_link(LinkEstimate {
            rtt,
            bandwidth_bytes_per_s,
        })
    }

    /// Override the clamps (`min_window <= max_window` is required).
    pub fn with_clamps(mut self, min_window: u64, max_window: u64) -> Result<Self, TransportError> {
        if min_window == 0 || min_window > max_window {
            return Err(TransportError::Config(format!(
                "flow-control clamps must satisfy 0 < min_window <= max_window, got {min_window}..{max_window}"
            )));
        }
        self.min_window = min_window;
        self.max_window = max_window;
        Ok(self)
    }

    /// The windows this config applies, or `None` when quinn's defaults are kept.
    pub fn windows(&self) -> Option<FlowWindows> {
        let link = self.link?;
        // Windows must fit a QUIC varint (< 2^62).
        let cap = VarInt::MAX.into_inner() / CONNECTION_WINDOW_STREAMS;
        let max = self.max_window.min(cap);
        let min = self.min_window.min(max);
        let stream = link
            .bdp_bytes()
            .saturating_mul(BDP_HEADROOM)
            .clamp(min, max);
        let conn = stream * CONNECTION_WINDOW_STREAMS;
        Some(FlowWindows {
            stream_receive_window: stream,
            receive_window: conn,
            send_window: conn,
        })
    }

    /// Build the quinn `TransportConfig` used by both server and client:
    /// flow-control windows from this config, uni streams disabled (sakura's
    /// RPCs are bidirectional only).
    pub fn transport_config(&self) -> TransportConfig {
        let mut transport = TransportConfig::default();
        transport.max_concurrent_uni_streams(0u8.into());
        if let Some(w) = self.windows() {
            // `windows()` keeps every value below VarInt::MAX, so these cannot fail.
            transport
                .stream_receive_window(
                    VarInt::from_u64(w.stream_receive_window).unwrap_or(VarInt::MAX),
                )
                .receive_window(VarInt::from_u64(w.receive_window).unwrap_or(VarInt::MAX))
                .send_window(w.send_window);
        }
        transport
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
        cert_der: cert.cert.der().to_vec(),
        // rcgen 0.14 renamed CertifiedKey::key_pair to signing_key.
        key_der: cert.signing_key.serialize_der(),
    })
}

fn server_config(
    pair: &SelfSignedPair,
    flow: &FlowControlConfig,
) -> Result<ServerConfig, TransportError> {
    // rustls 0.23: install a default crypto provider once per process.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain = vec![CertificateDer::from(pair.cert_der.clone())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pair.key_der.clone()));

    let crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .map_err(|e| TransportError::Rustls(e.to_string()))?;

    let quic_crypto =
        QuicServerConfig::try_from(crypto).map_err(|e| TransportError::Rustls(e.to_string()))?;

    let mut cfg = ServerConfig::with_crypto(Arc::new(quic_crypto));
    cfg.transport = Arc::new(flow.transport_config());
    Ok(cfg)
}

fn client_config(
    trusted_cert: &[u8],
    flow: &FlowControlConfig,
) -> Result<quinn::ClientConfig, TransportError> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from(trusted_cert.to_vec()))
        .map_err(|e| TransportError::Rustls(e.to_string()))?;

    let crypto = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let quic_crypto =
        QuicClientConfig::try_from(crypto).map_err(|e| TransportError::Rustls(e.to_string()))?;

    let mut cfg = quinn::ClientConfig::new(Arc::new(quic_crypto));
    cfg.transport_config(Arc::new(flow.transport_config()));
    Ok(cfg)
}

/// Bind a QUIC server to `bind_addr`, returning the actual address it listens on.
/// Uses quinn's default flow-control windows; see [`bind_server_with`].
pub fn bind_server(
    bind_addr: SocketAddr,
    pair: &SelfSignedPair,
) -> Result<Endpoint, TransportError> {
    bind_server_with(bind_addr, pair, &FlowControlConfig::default())
}

/// [`bind_server`] with explicit flow-control windows (SAK-0010).
pub fn bind_server_with(
    bind_addr: SocketAddr,
    pair: &SelfSignedPair,
    flow: &FlowControlConfig,
) -> Result<Endpoint, TransportError> {
    let cfg = server_config(pair, flow)?;
    let endpoint = Endpoint::server(cfg, bind_addr)?;
    Ok(endpoint)
}

/// Connect to a server with a known cert (loopback / pinned-cert use case).
/// Uses quinn's default flow-control windows; see [`connect_with`].
pub async fn connect(
    server_addr: SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
) -> Result<Connection, TransportError> {
    connect_with(
        server_addr,
        server_name,
        trusted_cert,
        &FlowControlConfig::default(),
    )
    .await
}

/// [`connect`] with explicit flow-control windows (SAK-0010).
pub async fn connect_with(
    server_addr: SocketAddr,
    server_name: &str,
    trusted_cert: &[u8],
    flow: &FlowControlConfig,
) -> Result<Connection, TransportError> {
    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config(trusted_cert, flow)?);
    let conn = endpoint.connect(server_addr, server_name)?.await?;
    Ok(conn)
}

/// Open a bidirectional stream, write the request bytes, signal end-of-write,
/// then read the entire response and return it.
///
/// quinn 0.11: `SendStream::finish()` is now sync (returns `Result<(), ClosedStream>`)
/// — closes the write side immediately. We then await `stopped()` to wait for the
/// peer to ack — but for our request/response shape, we just rely on `read_to_end`
/// on the recv side to drive completion.
pub async fn rpc_call(
    conn: &Connection,
    request_bytes: Vec<u8>,
) -> Result<Vec<u8>, TransportError> {
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&request_bytes).await?;
    send.finish()
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
///
/// quinn 0.11: `finish()` is sync (just closes the local write side); we then
/// await `stopped()` so the spawned task doesn't drop the connection before
/// the peer has acked the data — otherwise the client sees "connection lost"
/// mid-read.
pub async fn send_response(
    mut send: quinn::SendStream,
    bytes: Vec<u8>,
) -> Result<(), TransportError> {
    send.write_all(&bytes).await?;
    send.finish()
        .map_err(|e| TransportError::Write(e.to_string()))?;
    let _ = send.stopped().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn debug_of(t: &TransportConfig) -> String {
        format!("{t:?}")
    }

    #[test]
    fn default_config_keeps_quinn_defaults() {
        let flow = FlowControlConfig::default();
        assert_eq!(flow.link, None);
        assert_eq!(flow.windows(), None);
        // Same windows as an untouched quinn TransportConfig (only uni streams differ).
        let mut expected = TransportConfig::default();
        expected.max_concurrent_uni_streams(0u8.into());
        assert_eq!(debug_of(&flow.transport_config()), debug_of(&expected));
        assert!(debug_of(&expected).contains(&format!(
            "stream_receive_window: {QUINN_DEFAULT_STREAM_WINDOW}"
        )));
    }

    #[test]
    fn bdp_is_rtt_times_bandwidth() {
        let link = LinkEstimate {
            rtt: Duration::from_millis(248),
            bandwidth_bytes_per_s: 12_500_000, // 100 Mbit/s
        };
        assert_eq!(link.bdp_bytes(), 3_100_000);
        let w = FlowControlConfig::from_link(link).windows().unwrap();
        assert_eq!(w.stream_receive_window, 6_200_000);
        assert_eq!(w.receive_window, 6_200_000 * CONNECTION_WINDOW_STREAMS);
        assert_eq!(w.send_window, w.receive_window);
    }

    #[test]
    fn bdp_scales_linearly_between_clamps() {
        let at = |ms: u64| {
            FlowControlConfig::from_rtt_bandwidth(Duration::from_millis(ms), 50_000_000)
                .windows()
                .unwrap()
                .stream_receive_window
        };
        assert_eq!(at(200), 2 * at(100));
        assert_eq!(at(100), 10_000_000);
    }

    #[test]
    fn windows_are_clamped() {
        // Tiny BDP (LAN) -> floor at quinn's default.
        let lan = FlowControlConfig::from_rtt_bandwidth(Duration::from_micros(200), 1_000_000);
        assert_eq!(
            lan.windows().unwrap().stream_receive_window,
            DEFAULT_MIN_WINDOW
        );
        // Huge BDP -> ceiling.
        let huge = FlowControlConfig::from_rtt_bandwidth(Duration::from_secs(10), u64::MAX);
        assert_eq!(
            huge.windows().unwrap().stream_receive_window,
            DEFAULT_MAX_WINDOW
        );
        // Custom clamps.
        let custom = FlowControlConfig::from_rtt_bandwidth(Duration::from_millis(248), 12_500_000)
            .with_clamps(1_000, 4_000_000)
            .unwrap();
        assert_eq!(custom.windows().unwrap().stream_receive_window, 4_000_000);
        // max_window beyond the varint range is capped, not a panic.
        let unbounded = FlowControlConfig::from_rtt_bandwidth(Duration::from_secs(10), u64::MAX)
            .with_clamps(1, u64::MAX)
            .unwrap();
        let w = unbounded.windows().unwrap();
        assert!(w.receive_window <= VarInt::MAX.into_inner());
        let _ = unbounded.transport_config();
    }

    #[test]
    fn invalid_clamps_are_rejected() {
        let base = FlowControlConfig::default();
        assert!(base.with_clamps(0, 10).is_err());
        assert!(base.with_clamps(10, 5).is_err());
        assert!(base.with_clamps(5, 5).is_ok());
    }

    #[test]
    fn transport_config_applies_windows() {
        let flow = FlowControlConfig::from_rtt_bandwidth(Duration::from_millis(248), 12_500_000);
        let dbg = debug_of(&flow.transport_config());
        assert!(dbg.contains("stream_receive_window: 6200000"), "{dbg}");
        assert!(dbg.contains("receive_window: 49600000"), "{dbg}");
        assert!(dbg.contains("send_window: 49600000"), "{dbg}");
        // The uni-stream cap is kept.
        assert!(dbg.contains("max_concurrent_uni_streams: 0"), "{dbg}");
    }
}
