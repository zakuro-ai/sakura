//! fp32 ↔ fp16 / bf16 conversion. Bit-identical to torch's `.to(torch.float16)`
//! (IEEE 754 round-to-nearest-even via the `half` crate).

use half::{bf16, f16};

/// Cast a slice of f32 bytes into a Vec<u8> of fp16 bytes (length halves).
pub fn cast_f32_to_f16(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len().is_multiple_of(4), "input must be 4-byte aligned (f32)");
    let n = input.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for chunk in input.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Cast a slice of fp16 bytes back to f32 bytes (length doubles).
pub fn cast_f16_to_f32(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len().is_multiple_of(2), "input must be 2-byte aligned (f16)");
    let n = input.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in input.chunks_exact(2) {
        let h = f16::from_le_bytes([chunk[0], chunk[1]]);
        let v = h.to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Cast a slice of f32 bytes into a Vec<u8> of bf16 bytes (length halves).
pub fn cast_f32_to_bf16(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len().is_multiple_of(4), "input must be 4-byte aligned (f32)");
    let n = input.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for chunk in input.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = bf16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Cast a slice of bf16 bytes back to f32 bytes (length doubles).
pub fn cast_bf16_to_f32(input: &[u8]) -> Vec<u8> {
    debug_assert!(input.len().is_multiple_of(2), "input must be 2-byte aligned (bf16)");
    let n = input.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in input.chunks_exact(2) {
        let h = bf16::from_le_bytes([chunk[0], chunk[1]]);
        let v = h.to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn bytes_to_f32(input: &[u8]) -> Vec<f32> {
        input
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn fp16_roundtrip_preserves_representable_values() {
        let original = [0.0_f32, 1.0, -1.0, 0.5, -0.5, 256.0, 1e-4];
        let bytes = f32_bytes(&original);
        let half = cast_f32_to_f16(&bytes);
        assert_eq!(half.len(), bytes.len() / 2);
        let back = cast_f16_to_f32(&half);
        let recovered = bytes_to_f32(&back);
        for (orig, got) in original.iter().zip(recovered.iter()) {
            assert!(
                (orig - got).abs() < 1e-3,
                "fp16 round-trip drift: {} vs {}",
                orig,
                got
            );
        }
    }

    #[test]
    fn bf16_roundtrip_preserves_dynamic_range() {
        let original = [0.0_f32, 1.0, -1.0, 1e10, -1e10, 1e-30];
        let bytes = f32_bytes(&original);
        let bf = cast_f32_to_bf16(&bytes);
        assert_eq!(bf.len(), bytes.len() / 2);
        let back = cast_bf16_to_f32(&bf);
        let recovered = bytes_to_f32(&back);
        for (orig, got) in original.iter().zip(recovered.iter()) {
            // bf16 has 7 mantissa bits — relative precision ~1%.
            let rel = if orig.abs() > 1e-6 { (orig - got).abs() / orig.abs() } else { (orig - got).abs() };
            assert!(rel < 1e-2, "bf16 round-trip drift: {} vs {}", orig, got);
        }
    }

    #[test]
    fn empty_slice_yields_empty_output() {
        assert!(cast_f32_to_f16(&[]).is_empty());
        assert!(cast_f16_to_f32(&[]).is_empty());
        assert!(cast_f32_to_bf16(&[]).is_empty());
        assert!(cast_bf16_to_f32(&[]).is_empty());
    }
}
