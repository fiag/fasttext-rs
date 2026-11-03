// Shared SIMD kernels for dot product, vector addition, and row averaging.
//
// Organized by target feature level:
//   - scalar   — portable fallback
//   - neon     — aarch64 NEON (128-bit with FMA)
//   - sse2     — x86_64 baseline (128-bit)
//   - avx      — x86_64 AVX (256-bit, no FMA)
//   - avx2     — x86_64 AVX2 + FMA (256-bit with fused multiply-add)
//   - avx512   — x86_64 AVX-512F (512-bit with FMA)
//
// The `dot_impl` and `add_vector_impl` dispatch functions select the best
// available implementation at runtime on x86_64 via `is_x86_feature_detected!`.

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "x86_64")]
mod sse2;

use crate::matrix::DenseMatrix;
use crate::vector::Vector;

// Scalar fallback implementations

/// Scalar fallback for dot product of two slices.
#[allow(dead_code)]
#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Scalar fallback for dest += scale * src.
#[allow(dead_code)]
#[inline]
pub(crate) fn add_vector_scalar(dest: &mut [f32], src: &[f32], scale: f32) {
    dest.iter_mut()
        .zip(src.iter())
        .for_each(|(d, &s)| *d += scale * s);
}

/// Scalar fallback for matrix-vector product `out[i] = dot(mat[i], x)`.
///
/// Four rows are processed per iteration so the compiler can keep four
/// independent accumulator chains (and shared `x` loads) in registers.
#[allow(dead_code)]
pub(crate) fn gemv_scalar(mat: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    let n = out.len();
    if cols == 0 || n == 0 {
        return;
    }
    let mut i = 0;
    while i + 4 <= n {
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let r0 = &mat[i * cols..];
        let r1 = &mat[(i + 1) * cols..];
        let r2 = &mat[(i + 2) * cols..];
        let r3 = &mat[(i + 3) * cols..];
        for j in 0..cols {
            let xv = x[j];
            s0 += r0[j] * xv;
            s1 += r1[j] * xv;
            s2 += r2[j] * xv;
            s3 += r3[j] * xv;
        }
        out[i] = s0;
        out[i + 1] = s1;
        out[i + 2] = s2;
        out[i + 3] = s3;
        i += 4;
    }
    while i < n {
        out[i] = dot_scalar(&mat[i * cols..(i + 1) * cols], x);
        i += 1;
    }
}

/// Scalar average_rows implementation.
#[allow(dead_code)]
pub(crate) fn average_rows_scalar(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    x.zero();
    for &row_idx in rows {
        let row = mat.row(row_idx as i64);
        add_vector_impl(x.data_mut(), row, 1.0);
    }
    if !rows.is_empty() {
        x.mul(1.0 / rows.len() as f32);
    }
}

// Dispatch functions

/// Dispatch dot product to best available implementation.
#[inline]
pub(crate) fn dot_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::dot(a, b) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::dot(a, b) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::dot(a, b) }
        } else {
            sse2::dot(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

/// Dispatch add_vector to best available implementation.
#[inline]
pub(crate) fn add_vector_impl(dest: &mut [f32], src: &[f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::add_vector(dest, src, scale)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::add_vector(dest, src, scale) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::add_vector(dest, src, scale) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::add_vector(dest, src, scale) }
        } else {
            sse2::add_vector(dest, src, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        add_vector_scalar(dest, src, scale)
    }
}

/// Dispatch matrix-vector product to the best available implementation.
#[inline]
pub(crate) fn gemv_impl(mat: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::gemv(mat, cols, x, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemv_scalar(mat, cols, x, out);
    }
}

/// Dispatch maximum-value scan to the best available implementation.
#[inline]
pub(crate) fn max_impl(data: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::max_value(data)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Dispatch in-place shifted exponential to the best available implementation.
#[inline]
pub(crate) fn exp_shifted_impl(data: &mut [f32], shift: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::exp_shifted_in_place(data, shift);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in data.iter_mut() {
            *v = (*v - shift).exp();
        }
    }
}

/// Dispatch average_rows to best available implementation.
pub(crate) fn average_rows_impl(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::average_rows(x, rows, mat);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::average_rows(x, rows, mat) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::average_rows(x, rows, mat) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::average_rows(x, rows, mat) }
        } else {
            sse2::average_rows(x, rows, mat)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        average_rows_scalar(x, rows, mat);
    }
}
