// NEON SIMD kernels (aarch64).

use std::arch::aarch64::*;

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let a0 = vld1q_f32(a_ptr.add(offset));
            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let a2 = vld1q_f32(a_ptr.add(offset + 8));
            let a3 = vld1q_f32(a_ptr.add(offset + 12));
            let b0 = vld1q_f32(b_ptr.add(offset));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            let b2 = vld1q_f32(b_ptr.add(offset + 8));
            let b3 = vld1q_f32(b_ptr.add(offset + 12));
            sum0 = vfmaq_f32(sum0, a0, b0);
            sum1 = vfmaq_f32(sum1, a1, b1);
            sum2 = vfmaq_f32(sum2, a2, b2);
            sum3 = vfmaq_f32(sum3, a3, b3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        // Vectorize the tail with a 4-wide accumulator.  This matters for
        // small dimensions (e.g. dim=10), where the 16-wide main loop never
        // runs and the scalar tail would otherwise dominate.
        let rem_start = chunks * 16;
        let mut tail_sum = vdupq_n_f32(0.0);
        let mut i = rem_start;
        while i + 4 <= n {
            tail_sum = vfmaq_f32(tail_sum, vld1q_f32(a_ptr.add(i)), vld1q_f32(b_ptr.add(i)));
            i += 4;
        }

        let mut result = vaddvq_f32(sum0) + vaddvq_f32(tail_sum);
        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }
}

#[inline]
pub(crate) fn add_vector(dest: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dest.len(), src.len());
    let n = dest.len();
    let chunks = n / 16;

    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let d_ptr = dest.as_mut_ptr();
        let s_ptr = src.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let d0 = vld1q_f32(d_ptr.add(offset));
            let d1 = vld1q_f32(d_ptr.add(offset + 4));
            let d2 = vld1q_f32(d_ptr.add(offset + 8));
            let d3 = vld1q_f32(d_ptr.add(offset + 12));
            let s0 = vld1q_f32(s_ptr.add(offset));
            let s1 = vld1q_f32(s_ptr.add(offset + 4));
            let s2 = vld1q_f32(s_ptr.add(offset + 8));
            let s3 = vld1q_f32(s_ptr.add(offset + 12));
            vst1q_f32(d_ptr.add(offset), vfmaq_f32(d0, scale_v, s0));
            vst1q_f32(d_ptr.add(offset + 4), vfmaq_f32(d1, scale_v, s1));
            vst1q_f32(d_ptr.add(offset + 8), vfmaq_f32(d2, scale_v, s2));
            vst1q_f32(d_ptr.add(offset + 12), vfmaq_f32(d3, scale_v, s3));
        }

        let rem_start = chunks * 16;
        let mut i = rem_start;
        while i + 4 <= n {
            let d = vld1q_f32(d_ptr.add(i));
            let s = vld1q_f32(s_ptr.add(i));
            vst1q_f32(d_ptr.add(i), vfmaq_f32(d, scale_v, s));
            i += 4;
        }
        while i < n {
            dest[i] += scale * src[i];
            i += 1;
        }
    }
}

/// Compute `out[i] = dot(mat[i], x)` for a row-major matrix.
///
/// Processes four rows at a time so each load of `x` feeds four independent
/// FMA chains.  Small output layers (e.g. a 10-dim hidden vector over 735
/// labels) are dominated by per-row call and load overhead, and this kernel
/// keeps the inner loop latency-bound instead.
pub(crate) fn gemv(mat: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    let n = out.len();
    if cols == 0 || n == 0 {
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= n {
            let base = mat.as_ptr().add(i * cols);
            let r0 = base;
            let r1 = r0.add(cols);
            let r2 = r1.add(cols);
            let r3 = r2.add(cols);
            let r4 = r3.add(cols);
            let r5 = r4.add(cols);
            let r6 = r5.add(cols);
            let r7 = r6.add(cols);
            let xp = x.as_ptr();
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut acc4 = vdupq_n_f32(0.0);
            let mut acc5 = vdupq_n_f32(0.0);
            let mut acc6 = vdupq_n_f32(0.0);
            let mut acc7 = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 4 <= cols {
                let xv = vld1q_f32(xp.add(j));
                acc0 = vfmaq_f32(acc0, vld1q_f32(r0.add(j)), xv);
                acc1 = vfmaq_f32(acc1, vld1q_f32(r1.add(j)), xv);
                acc2 = vfmaq_f32(acc2, vld1q_f32(r2.add(j)), xv);
                acc3 = vfmaq_f32(acc3, vld1q_f32(r3.add(j)), xv);
                acc4 = vfmaq_f32(acc4, vld1q_f32(r4.add(j)), xv);
                acc5 = vfmaq_f32(acc5, vld1q_f32(r5.add(j)), xv);
                acc6 = vfmaq_f32(acc6, vld1q_f32(r6.add(j)), xv);
                acc7 = vfmaq_f32(acc7, vld1q_f32(r7.add(j)), xv);
                j += 4;
            }
            let mut s0 = vaddvq_f32(acc0);
            let mut s1 = vaddvq_f32(acc1);
            let mut s2 = vaddvq_f32(acc2);
            let mut s3 = vaddvq_f32(acc3);
            let mut s4 = vaddvq_f32(acc4);
            let mut s5 = vaddvq_f32(acc5);
            let mut s6 = vaddvq_f32(acc6);
            let mut s7 = vaddvq_f32(acc7);
            while j < cols {
                let xv = *xp.add(j);
                s0 += *r0.add(j) * xv;
                s1 += *r1.add(j) * xv;
                s2 += *r2.add(j) * xv;
                s3 += *r3.add(j) * xv;
                s4 += *r4.add(j) * xv;
                s5 += *r5.add(j) * xv;
                s6 += *r6.add(j) * xv;
                s7 += *r7.add(j) * xv;
                j += 1;
            }
            out[i] = s0;
            out[i + 1] = s1;
            out[i + 2] = s2;
            out[i + 3] = s3;
            out[i + 4] = s4;
            out[i + 5] = s5;
            out[i + 6] = s6;
            out[i + 7] = s7;
            i += 8;
        }
    }
    while i < n {
        out[i] = dot(&mat[i * cols..(i + 1) * cols], x);
        i += 1;
    }
}

/// Sum of a slice.
pub(crate) fn sum_value(data: &[f32]) -> f32 {
    let n = data.len();
    let mut i = 0;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            acc = vaddq_f32(acc, vld1q_f32(data.as_ptr().add(i)));
            i += 4;
        }
        let mut result = vaddvq_f32(acc);
        while i < n {
            result += data[i];
            i += 1;
        }
        result
    }
}

/// Maximum value in a slice (NaN is ignored, matching `f32::max` folding).
pub(crate) fn max_value(data: &[f32]) -> f32 {
    let n = data.len();
    let mut i = 0;
    unsafe {
        let mut m = vdupq_n_f32(f32::NEG_INFINITY);
        while i + 4 <= n {
            m = vmaxq_f32(m, vld1q_f32(data.as_ptr().add(i)));
            i += 4;
        }
        let mut result = vmaxvq_f32(m);
        while i < n {
            result = result.max(data[i]);
            i += 1;
        }
        result
    }
}

/// Vectorized `exp` for four f32 lanes.
///
/// `exp(x) = 2^n * exp(r)` with `n = round(x / ln2)` and `r = x - n*ln2`.
/// `exp(r)` is evaluated with a degree-6 Taylor polynomial for `|r| <= ln2/2`
/// (relative error below 1e-7).  The caller is responsible for clamping `x`
/// to the representable range.
#[inline]
unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(std::f32::consts::LN_2);
    let ln2_lo = vdupq_n_f32(-1.904_654_3e-9_f32);

    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vfmsq_f32(x, n, ln2_hi);
    let r = vfmsq_f32(r, n, ln2_lo);

    // 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    let mut p = vdupq_n_f32(1.0 / 720.0);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), r, p);
    p = vfmaq_f32(vdupq_n_f32(0.5), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0), r, p);

    // Scale by 2^n via the IEEE-754 exponent field.
    let ni = vcvtq_s32_f32(n);
    let bits = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    vmulq_f32(p, vreinterpretq_f32_s32(bits))
}

/// In-place `data[i] = exp(data[i] - shift)`.
///
/// Intended for softmax where `shift` is the maximum input value, so every
/// argument is non-positive.  Arguments below -87 are clamped to 0 (they
/// underflow in the scalar path as well); inputs above 88 saturate to infinity.
pub(crate) fn exp_shifted_in_place(data: &mut [f32], shift: f32) {
    let n = data.len();
    let mut i = 0;
    unsafe {
        let neg_shift = vdupq_n_f32(-shift);
        let lo = vdupq_n_f32(-87.0);
        let hi = vdupq_n_f32(88.0);
        while i + 4 <= n {
            let x = vld1q_f32(data.as_ptr().add(i));
            let x = vaddq_f32(x, neg_shift);
            let x = vmaxq_f32(x, lo);
            let x = vminq_f32(x, hi);
            vst1q_f32(data.as_mut_ptr().add(i), exp_ps(x));
            i += 4;
        }
    }
    while i < n {
        data[i] = (data[i] - shift).exp();
        i += 1;
    }
}

pub(crate) fn average_rows(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.cols() as usize;

    let first_row = mat.row(rows[0] as i64);
    x.data_mut().copy_from_slice(first_row);

    for &row_idx in &rows[1..] {
        let row = mat.row(row_idx as i64);
        add_vector(x.data_mut(), row, 1.0);
    }

    let scale = 1.0 / rows.len() as f32;
    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let chunks = n / 16;
        let remainder = n % 16;
        let d_ptr = x.data_mut().as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let v0 = vld1q_f32(d_ptr.add(offset));
            let v1 = vld1q_f32(d_ptr.add(offset + 4));
            let v2 = vld1q_f32(d_ptr.add(offset + 8));
            let v3 = vld1q_f32(d_ptr.add(offset + 12));
            vst1q_f32(d_ptr.add(offset), vmulq_f32(v0, scale_v));
            vst1q_f32(d_ptr.add(offset + 4), vmulq_f32(v1, scale_v));
            vst1q_f32(d_ptr.add(offset + 8), vmulq_f32(v2, scale_v));
            vst1q_f32(d_ptr.add(offset + 12), vmulq_f32(v3, scale_v));
        }

        let rem_start = chunks * 16;
        for i in 0..remainder {
            x.data_mut()[rem_start + i] *= scale;
        }
    }
}
