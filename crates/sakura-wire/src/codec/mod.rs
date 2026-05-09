//! Wire-format codec: postcard headers + raw zero-copy tensor payloads.

pub mod cast;
pub mod header;
pub mod tensor;

pub use header::{
    Device, Dtype, RpcRequestHeader, RpcResponseHeader, RpcStatus, TensorDesc, WireVersion,
};
pub use tensor::{OwnedTensor, TensorView};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("postcard encode failed: {0}")]
    Encode(String),
    #[error("postcard decode failed: {0}")]
    Decode(String),
    #[error("truncated payload: expected {expected} bytes, got {got}")]
    Truncated { expected: usize, got: usize },
    #[error("descriptor count mismatch: header.n_tensors = {header}, descriptors = {got}")]
    DescriptorCountMismatch { header: u32, got: usize },
}

impl From<postcard::Error> for CodecError {
    fn from(e: postcard::Error) -> Self {
        CodecError::Decode(e.to_string())
    }
}

/// Pack a request into a single contiguous buffer. Used by the client side.
///
/// Layout: [postcard(header)] [postcard(Vec<TensorDesc>)] [tensor bytes...] [aux bytes].
pub fn pack_request(
    header: &RpcRequestHeader,
    tensors: &[TensorView<'_>],
    aux: &[u8],
) -> Result<Vec<u8>, CodecError> {
    let descs: Vec<TensorDesc> = tensors.iter().map(|t| t.desc.clone()).collect();
    let header_bytes =
        postcard::to_allocvec(header).map_err(|e| CodecError::Encode(e.to_string()))?;
    let descs_bytes =
        postcard::to_allocvec(&descs).map_err(|e| CodecError::Encode(e.to_string()))?;

    let total: usize = header_bytes.len()
        + descs_bytes.len()
        + tensors.iter().map(|t| t.bytes.len()).sum::<usize>()
        + aux.len();
    let mut out = Vec::with_capacity(total);

    // Length-prefix each postcard chunk so the unpacker knows its size.
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&(descs_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&descs_bytes);
    for t in tensors {
        out.extend_from_slice(t.bytes);
    }
    out.extend_from_slice(aux);
    Ok(out)
}

/// Unpacked components of a wire request: header, descriptors, tensor payloads, aux bytes.
pub type UnpackedRequest = (RpcRequestHeader, Vec<TensorDesc>, Vec<Vec<u8>>, Vec<u8>);

/// Like pack_request, but returns Vec<Bytes> chunks for quinn::SendStream::write_all_chunks.
///
/// Each tensor's bytes become a separate `Bytes` wrapping a copy of the buffer.
/// Compared to `pack_request` (which `Vec::extend_from_slice` consolidates everything
/// into one giant `Vec<u8>`), this avoids the final consolidation memcpy. quinn writes
/// the chunks directly to the QUIC stream — under the hood quinn coalesces but avoids
/// the full O(N) `Vec::with_capacity + extend`.
///
/// Note: true zero-copy would require the caller to supply owned `Bytes` from the source
/// (the PyO3 buffer protocol gives borrowed `&[u8]`, so one copy per tensor is unavoidable).
pub fn pack_request_zero_copy(
    header: &RpcRequestHeader,
    tensors: &[TensorView<'_>],
    aux: &[u8],
) -> Result<Vec<bytes::Bytes>, CodecError> {
    let descs: Vec<TensorDesc> = tensors.iter().map(|t| t.desc.clone()).collect();
    let header_bytes =
        postcard::to_allocvec(header).map_err(|e| CodecError::Encode(e.to_string()))?;
    let descs_bytes =
        postcard::to_allocvec(&descs).map_err(|e| CodecError::Encode(e.to_string()))?;

    // Slots: 4-byte header-len prefix, header body, 4-byte descs-len prefix, descs body,
    //        one chunk per tensor body, aux body.
    let mut chunks: Vec<bytes::Bytes> = Vec::with_capacity(4 + tensors.len() + 1);
    chunks.push(bytes::Bytes::copy_from_slice(
        &(header_bytes.len() as u32).to_le_bytes(),
    ));
    chunks.push(bytes::Bytes::from(header_bytes));
    chunks.push(bytes::Bytes::copy_from_slice(
        &(descs_bytes.len() as u32).to_le_bytes(),
    ));
    chunks.push(bytes::Bytes::from(descs_bytes));
    for t in tensors {
        chunks.push(bytes::Bytes::copy_from_slice(t.bytes));
    }
    chunks.push(bytes::Bytes::copy_from_slice(aux));
    Ok(chunks)
}

/// Unpack a request buffer into header + descriptors + per-tensor byte slices + aux bytes.
/// Returns owned bytes for tensors so the caller can reuse the input buffer.
pub fn unpack_request(buf: &[u8]) -> Result<UnpackedRequest, CodecError> {
    let mut cursor = 0usize;
    let read_u32 = |cur: &mut usize, buf: &[u8]| -> Result<u32, CodecError> {
        if buf.len() < *cur + 4 {
            return Err(CodecError::Truncated {
                expected: *cur + 4,
                got: buf.len(),
            });
        }
        let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };

    let header_len = read_u32(&mut cursor, buf)? as usize;
    if buf.len() < cursor + header_len {
        return Err(CodecError::Truncated {
            expected: cursor + header_len,
            got: buf.len(),
        });
    }
    let header: RpcRequestHeader = postcard::from_bytes(&buf[cursor..cursor + header_len])?;
    cursor += header_len;

    let descs_len = read_u32(&mut cursor, buf)? as usize;
    if buf.len() < cursor + descs_len {
        return Err(CodecError::Truncated {
            expected: cursor + descs_len,
            got: buf.len(),
        });
    }
    let descs: Vec<TensorDesc> = postcard::from_bytes(&buf[cursor..cursor + descs_len])?;
    cursor += descs_len;

    if descs.len() != header.n_tensors as usize {
        return Err(CodecError::DescriptorCountMismatch {
            header: header.n_tensors,
            got: descs.len(),
        });
    }

    let mut tensors = Vec::with_capacity(descs.len());
    for d in &descs {
        let n = d.n_bytes as usize;
        if buf.len() < cursor + n {
            return Err(CodecError::Truncated {
                expected: cursor + n,
                got: buf.len(),
            });
        }
        tensors.push(buf[cursor..cursor + n].to_vec());
        cursor += n;
    }

    let aux_len = header.aux_payload_bytes as usize;
    if buf.len() < cursor + aux_len {
        return Err(CodecError::Truncated {
            expected: cursor + aux_len,
            got: buf.len(),
        });
    }
    let aux = buf[cursor..cursor + aux_len].to_vec();
    Ok((header, descs, tensors, aux))
}
