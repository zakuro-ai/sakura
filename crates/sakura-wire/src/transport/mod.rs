//! Transport layer: QUIC over UDP via quinn (default).

pub mod quic;

pub use quic::{
    accept_request, bind_server, bind_server_with, connect, connect_with, generate_self_signed,
    rpc_call, send_response, FlowControlConfig, FlowWindows, LinkEstimate, SelfSignedPair,
    TransportError,
};
