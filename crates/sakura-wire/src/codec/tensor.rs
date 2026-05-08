//! Owning + borrowing tensor types used by the codec.
//!
//! `TensorView` borrows external bytes (zero-copy from the buffer protocol);
//! `OwnedTensor` owns its bytes (used on the receiver side after assembling
//! a buffer from the wire).

use crate::codec::header::{Device, Dtype, TensorDesc};
use smallvec::SmallVec;

/// A borrowed view of tensor bytes, paired with a descriptor.
/// Producers create these from PyO3 buffer protocol slices — no copy.
pub struct TensorView<'a> {
    pub desc: TensorDesc,
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(
        shape: impl Into<SmallVec<[u32; 8]>>,
        dtype: Dtype,
        device_hint: Device,
        bytes: &'a [u8],
    ) -> Self {
        let desc = TensorDesc {
            shape: shape.into(),
            dtype,
            n_bytes: bytes.len() as u64,
            device_hint,
            fp16_cast_on_wire: false,
        };
        Self { desc, bytes }
    }

    pub fn with_fp16_cast(mut self) -> Self {
        self.desc.fp16_cast_on_wire = true;
        self
    }

    pub fn n_elements(&self) -> usize {
        self.desc.shape.iter().product::<u32>() as usize
    }
}

/// A heap-allocated tensor blob used after assembling from the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    pub desc: TensorDesc,
    pub bytes: Vec<u8>,
}

impl OwnedTensor {
    pub fn new(desc: TensorDesc, bytes: Vec<u8>) -> Self {
        debug_assert_eq!(bytes.len() as u64, desc.n_bytes);
        Self { desc, bytes }
    }

    pub fn from_view(view: &TensorView<'_>) -> Self {
        Self::new(view.desc.clone(), view.bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn view_descriptor_matches_byte_length() {
        let bytes = vec![0u8; 4 * 12];
        let v = TensorView::new(smallvec![3u32, 4], Dtype::F32, Device::Cpu, &bytes);
        assert_eq!(v.desc.shape.as_slice(), &[3, 4]);
        assert_eq!(v.desc.dtype, Dtype::F32);
        assert_eq!(v.desc.n_bytes, 48);
        assert_eq!(v.n_elements(), 12);
        assert!(!v.desc.fp16_cast_on_wire);
    }

    #[test]
    fn view_fp16_cast_flag_flips() {
        let bytes = vec![0u8; 16];
        let v = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &bytes).with_fp16_cast();
        assert!(v.desc.fp16_cast_on_wire);
    }

    #[test]
    fn owned_tensor_roundtrips_from_view() {
        let bytes: Vec<u8> = (0..16).collect();
        let v = TensorView::new(smallvec![4u32], Dtype::F32, Device::Cpu, &bytes);
        let owned = OwnedTensor::from_view(&v);
        assert_eq!(owned.bytes, bytes);
        assert_eq!(owned.desc.dtype, Dtype::F32);
    }
}
