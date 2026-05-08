//! Transport layer: QUIC over UDP via quinn (default).

pub mod quic;

pub use quic::{
    accept_request, bind_server, connect, generate_self_signed, rpc_call, send_response,
    SelfSignedPair, TransportError,
};
