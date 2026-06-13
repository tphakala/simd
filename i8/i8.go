// Package i8 provides SIMD-accelerated operations on int8 slices.
//
// int8 is the 8-bit signed integer workhorse of quantized numeric pipelines.
// Its narrow range (-128..127) makes element-wise arithmetic overflow almost
// immediately, so this package does not mirror the wrapping arithmetic of the
// i16/i32 packages one-to-one. Instead it ships the operations that are
// genuinely high-impact and well-defined at 8-bit width:
//
//   - Saturating arithmetic (AddSaturate, SubSaturate, and the scalar-broadcast
//     AddScalarSaturate, SubScalarSaturate): single hardware instructions
//     (PADDSB/PSUBSB, SQADD/SQSUB) that clamp to [-128, 127] instead of wrapping,
//     which is what 8-bit arithmetic almost always wants.
//   - int32-accumulated reductions (Sum, DotProduct, SumAbs, SAD): widen to
//     int32 so the running total has headroom. DotProduct is the inner loop of
//     quantized matmul/conv; it uses ARM64 SDOT (FEAT_DotProd) where available
//     and AVX2 VPMADDWD otherwise. SumAbs is the L1 norm and SAD the sum of
//     absolute differences (block matching), both via PSADBW on AVX2.
//   - Signed min/max (MinMax reduction; element-wise two-slice Min/Max).
//   - Element-wise Clamp (activation clipping) and saturating Abs/Neg, where
//     -128 maps to 127 (SQABS/SQNEG on NEON; saturating constructions on AVX2).
//   - Saturating AbsDiff (|a-b| clamped to [0,127]) and MaxAbs (the per-tensor
//     abs-max for dynamic quantization, returned as int because |-128| = 128).
//   - Sign-extending widening (ToInt16, ToInt32) to hand off to the wider
//     integer or float packages.
//
// Sum and DotProduct accumulate in int32 with two's-complement wraparound,
// exactly like their pure-Go references. int32 wrapping addition is associative
// and commutative modulo 2^32, so the lane-parallel SIMD reductions are
// bit-identical to the scalar reference regardless of summation order. The
// intermediate products never overflow their SIMD lane (|int8 * int8| <= 16384),
// so only the final running total can wrap, and it wraps identically.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection and fall back to a pure-Go implementation on
// unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package i8

// AddSaturate writes dst[i] = clamp(int(a[i]) + int(b[i]), -128, 127) for i in
// [0, n), n = min(len(dst), len(a), len(b)). The add saturates to the int8
// range instead of wrapping, so 100 + 100 = 127 and -100 + -100 = -128. Any
// trailing capacity in dst is left untouched.
func AddSaturate(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	addSatI8(dst[:n], a[:n], b[:n])
}

// SubSaturate writes dst[i] = clamp(int(a[i]) - int(b[i]), -128, 127) for i in
// [0, n), n = min(len(dst), len(a), len(b)). The subtract saturates to the int8
// range instead of wrapping. Any trailing capacity in dst is left untouched.
func SubSaturate(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	subSatI8(dst[:n], a[:n], b[:n])
}

// SumAbs returns sum_i |a[i]| (the L1 norm), accumulated in int32 with
// two's-complement wraparound. |-128| = 128 contributes the full 128. An empty a
// returns 0. a is read-only; the call allocates nothing.
//
// On AVX2 it uses PABSB then PSADBW (sum of absolute differences against zero);
// on NEON, ABS then UADDLP/UADALP widen-accumulate.
func SumAbs(a []int8) int32 {
	if len(a) == 0 {
		return 0
	}
	return sumAbsI8(a)
}

// SAD returns sum_i |a[i] - b[i]| (the sum of absolute differences) over i in
// [0, n), n = min(len(a), len(b)), accumulated in int32 with two's-complement
// wraparound. The per-element difference is the true |a-b| in [0, 255] (not
// saturated), so |127 - (-128)| contributes 255. An empty operand returns 0.
// a and b are read-only; the call allocates nothing.
//
// SAD is the block-matching / feature-distance reduction (the scalar companion
// to AbsDiff). On AVX2 it offsets both operands by 128 and uses PSADBW; on NEON,
// SABD then UADDLP/UADALP widen-accumulate.
func SAD(a, b []int8) int32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	return sadI8(a[:n], b[:n])
}

// AddScalarSaturate writes dst[i] = clamp(int(a[i]) + int(s), -128, 127) for i
// in [0, n), n = min(len(dst), len(a)). It broadcasts the scalar s and adds with
// signed saturation (VPADDSB on AVX2, SQADD on NEON). Any trailing capacity in
// dst is left untouched.
func AddScalarSaturate(dst, a []int8, s int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	addScalarSatI8(dst[:n], a[:n], s)
}

// SubScalarSaturate writes dst[i] = clamp(int(a[i]) - int(s), -128, 127) for i
// in [0, n), n = min(len(dst), len(a)). It broadcasts the scalar s and subtracts
// with signed saturation (VPSUBSB on AVX2, SQSUB on NEON). Any trailing capacity
// in dst is left untouched.
func SubScalarSaturate(dst, a []int8, s int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	subScalarSatI8(dst[:n], a[:n], s)
}

// ToInt16 sign-extends src into dst: dst[i] = int16(src[i]) for i in [0, n),
// n = min(len(dst), len(src)). It is exact (int8 fits in int16). Any trailing
// capacity in dst is left untouched.
func ToInt16(dst []int16, src []int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	toI16(dst[:n], src[:n])
}

// ToInt32 sign-extends src into dst: dst[i] = int32(src[i]) for i in [0, n),
// n = min(len(dst), len(src)). It is exact (int8 fits in int32). Any trailing
// capacity in dst is left untouched.
func ToInt32(dst []int32, src []int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	toI32(dst[:n], src[:n])
}

// Sum returns the sum of all elements of a, accumulated in int32 with
// two's-complement wraparound. An empty a returns 0. a is read-only; the call
// allocates nothing.
func Sum(a []int8) int32 {
	if len(a) == 0 {
		return 0
	}
	return sumI8(a)
}

// DotProduct returns sum_i int32(a[i]) * int32(b[i]) over i in [0, n),
// n = min(len(a), len(b)), accumulated in int32 with two's-complement
// wraparound. An empty operand returns 0. a and b are read-only; the call
// allocates nothing.
//
// This is the inner loop of quantized matmul/convolution. On ARM64 with
// FEAT_DotProd it uses SDOT (16 int8 multiply-accumulates per instruction); on
// AVX2 it widens with VPMOVSXBW and reduces with VPMADDWD.
func DotProduct(a, b []int8) int32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	return dotI8(a[:n], b[:n])
}

// MinMax returns the smallest and largest int8 in a:
//
//	minVal = min_i a[i],  maxVal = max_i a[i]
//
// Both are signed comparisons. An empty a returns (0, 0). a is read-only; the
// call allocates nothing.
func MinMax(a []int8) (minVal, maxVal int8) {
	if len(a) == 0 {
		return 0, 0
	}
	return minMaxI8(a)
}

// Min writes dst[i] = min(a[i], b[i]) (signed) for i in [0, n),
// n = min(len(dst), len(a), len(b)). This is the element-wise two-slice minimum
// (PMINSB/SMIN), distinct from the MinMax reduction. Any trailing capacity in
// dst is left untouched.
func Min(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	minI8(dst[:n], a[:n], b[:n])
}

// Max writes dst[i] = max(a[i], b[i]) (signed) for i in [0, n),
// n = min(len(dst), len(a), len(b)). This is the element-wise two-slice maximum
// (PMAXSB/SMAX), distinct from the MinMax reduction. Any trailing capacity in
// dst is left untouched.
func Max(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	maxI8(dst[:n], a[:n], b[:n])
}

// Clamp writes dst[i] = min(max(src[i], lo), hi) (signed) for i in [0, n),
// n = min(len(dst), len(src)). It is the activation-clipping primitive. If
// lo > hi every element maps to hi (max-then-min ordering). Any trailing
// capacity in dst is left untouched.
func Clamp(dst, src []int8, lo, hi int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	clampElemI8(dst[:n], src[:n], lo, hi)
}

// Abs writes the saturating absolute value dst[i] = |a[i]| for i in [0, n),
// n = min(len(dst), len(a)). abs(-128) saturates to 127 (SQABS on NEON; on AVX2
// max(a, saturating(0-a))). Any trailing capacity in dst is left untouched.
func Abs(dst, a []int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absI8(dst[:n], a[:n])
}

// Neg writes the saturating negation dst[i] = -a[i] for i in [0, n),
// n = min(len(dst), len(a)). neg(-128) saturates to 127 (SQNEG on NEON;
// saturating(0-a) via VPSUBSB on AVX2). Any trailing capacity in dst is left
// untouched.
func Neg(dst, a []int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	negI8(dst[:n], a[:n])
}

// MaxAbs returns max_i |a[i]| accumulated as int (range [0, 128], since
// |-128| = 128 does not fit int8). It is the per-tensor scale for dynamic
// quantization (PABSB+PMAXUB on AVX2; ABS+UMAXV on NEON). An empty a returns 0.
// a is read-only; the call allocates nothing.
func MaxAbs(a []int8) int {
	if len(a) == 0 {
		return 0
	}
	return maxAbsI8(a)
}

// AbsDiff writes the saturating absolute difference dst[i] = |a[i] - b[i]|,
// clamped to [0, 127], for i in [0, n), n = min(len(dst), len(a), len(b)).
// |127 - (-128)| = 255 saturates to 127, consistent with Abs. It uses
// max(saturating(a-b), saturating(b-a)) on AVX2 and SABD then an unsigned min
// with 127 on NEON. Any trailing capacity in dst is left untouched.
func AbsDiff(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	absDiffI8(dst[:n], a[:n], b[:n])
}
