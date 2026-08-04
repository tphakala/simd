package i8

import "math"

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.

// clampI8 saturates a widened sum/difference to the signed 8-bit range.
func clampI8(v int) int8 {
	switch {
	case v > math.MaxInt8:
		return math.MaxInt8
	case v < math.MinInt8:
		return math.MinInt8
	default:
		return int8(v)
	}
}

func addSatGo(dst, a, b []int8) {
	for i := range dst {
		dst[i] = clampI8(int(a[i]) + int(b[i]))
	}
}

func subSatGo(dst, a, b []int8) {
	for i := range dst {
		dst[i] = clampI8(int(a[i]) - int(b[i]))
	}
}

// sumAbsGo accumulates sum_i |a[i]| (the L1 norm) into int32 with
// two's-complement wraparound. It is the source of truth for the SumAbs kernels.
func sumAbsGo(a []int8) int32 {
	var s int32
	for _, v := range a {
		s += int32(absInt(int(v)))
	}
	return s
}

// sadGo accumulates sum_i |a[i] - b[i]| (sum of absolute differences) into int32
// with two's-complement wraparound. a and b are equal length (guaranteed by the
// public SAD clamp). It is the source of truth for the SAD kernels.
func sadGo(a, b []int8) int32 {
	var s int32
	for i := range a {
		s += int32(absInt(int(a[i]) - int(b[i])))
	}
	return s
}

// addScalarSatGo / subScalarSatGo broadcast a scalar with signed saturation;
// they reuse clampI8, so they are bit-exact with AddSaturate/SubSaturate.
func addScalarSatGo(dst, a []int8, s int8) {
	for i := range dst {
		dst[i] = clampI8(int(a[i]) + int(s))
	}
}

func subScalarSatGo(dst, a []int8, s int8) {
	for i := range dst {
		dst[i] = clampI8(int(a[i]) - int(s))
	}
}

func minGo(dst, a, b []int8) {
	for i := range dst {
		dst[i] = min(a[i], b[i])
	}
}

func maxGo(dst, a, b []int8) {
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

// clampGo clamps each element to [lo, hi]. With lo > hi every element maps to
// hi, matching the SIMD kernels' max-then-min ordering.
func clampGo(dst, src []int8, lo, hi int8) {
	for i := range dst {
		dst[i] = min(max(src[i], lo), hi)
	}
}

// absGo writes the saturating absolute value: abs(-128) clamps to 127.
func absGo(dst, a []int8) {
	for i := range dst {
		dst[i] = clampI8(absInt(int(a[i])))
	}
}

// negGo writes the saturating negation: -(-128) clamps to 127.
func negGo(dst, a []int8) {
	for i := range dst {
		dst[i] = clampI8(-int(a[i]))
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// maxAbsGo returns max_i |a[i]| as int. |-128| = 128 does not fit int8, hence
// the int return. It is the bit-exact source of truth for the MaxAbs kernels.
func maxAbsGo(a []int8) int {
	m := 0
	for _, v := range a {
		m = max(m, absInt(int(v)))
	}
	return m
}

// absDiffGo writes the saturating absolute difference clamped to [0, 127], so
// |127 - (-128)| = 255 maps to 127.
func absDiffGo(dst, a, b []int8) {
	for i := range dst {
		dst[i] = int8(min(absInt(int(a[i])-int(b[i])), math.MaxInt8))
	}
}

func toI16Go(dst []int16, src []int8) {
	for i := range dst {
		dst[i] = int16(src[i])
	}
}

func toI32Go(dst []int32, src []int8) {
	for i := range dst {
		dst[i] = int32(src[i])
	}
}

// sumGo accumulates a into int32 with two's-complement wraparound. It is the
// bit-exact source of truth the SIMD Sum kernels are validated against.
func sumGo(a []int8) int32 {
	var s int32
	for _, v := range a {
		s += int32(v)
	}
	return s
}

// dotGo computes the int32-accumulated dot product of a and b (equal length,
// guaranteed by the public DotProduct clamp) with two's-complement wraparound.
func dotGo(a, b []int8) int32 {
	var s int32
	for i := range a {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}

// minMaxGo returns the smallest and largest int8 in a via a single signed scan.
// a must be non-empty (the public MinMax guards the empty case); it is the
// bit-exact source of truth the SIMD MinMax kernels are validated against.
func minMaxGo(a []int8) (minVal, maxVal int8) {
	lo, hi := a[0], a[0]
	for _, v := range a[1:] {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return lo, hi
}

// -----------------------------------------------------------------------------
// Quantization references (Part of #132).
//
// These pure-Go implementations are the source of truth for the Quantize,
// Dequantize and Requantize kernels; every SIMD path is validated bit-exact
// against them. The semantics follow the ONNX / PyTorch / TFLite per-tensor
// affine convention: a real value r maps to a quantized value q via
// q = clamp(round(r/scale) + zeroPoint, -128, 127), and back via
// r = (q - zeroPoint) * scale.
// -----------------------------------------------------------------------------

// srdhmNudge is the rounding constant added before the Q31 shift in
// SaturatingRoundingDoublingHighMul: it rounds the doubled high-multiply to
// nearest, ties toward +infinity (gemmlowp semantics).
const srdhmNudge = 1 << 30

// srdhmShift is the Q31 fixed-point position: the doubled 64-bit product is
// shifted right by 31 to land its high half back in int32 range.
const srdhmShift = 31

// srdhm is gemmlowp's SaturatingRoundingDoublingHighMul: the high 32 bits of
// 2*x*multiplier, rounded to nearest with ties toward +infinity. The only
// saturating case is x == multiplier == math.MinInt32, whose true value 2^31
// saturates to math.MaxInt32. For every other input the doubled product fits in
// int64, and the Go arithmetic shift computes the floor gemmlowp specifies.
func srdhm(x, multiplier int32) int32 {
	if x == math.MinInt32 && multiplier == math.MinInt32 {
		return math.MaxInt32
	}
	return int32((int64(x)*int64(multiplier) + srdhmNudge) >> srdhmShift)
}

// rdbpot is gemmlowp's RoundingDivideByPOT: x divided by 2^exponent, rounded to
// nearest with ties AWAY from zero. exponent is in [0, 31]. At exponent == 0 it
// is the identity (the SIMD kernels depend on this being an exact pass-through,
// including for negative x; see the NEON rounding-shift fixup).
func rdbpot(x int32, exponent int) int32 {
	if exponent == 0 {
		return x
	}
	mask := int32(1)<<uint(exponent) - 1
	remainder := x & mask
	threshold := mask >> 1
	if x < 0 {
		threshold++
	}
	q := x >> uint(exponent)
	if remainder > threshold {
		q++
	}
	return q
}

// requantizeOutOfContract reports whether (multiplier, shift) fall outside the
// domain the SIMD kernels support. multiplier == math.MinInt32 would trip the
// srdhm saturation guard the kernels omit. shift is restricted to [-31, 30]: the
// vector shift instructions leave a count of 32 or more undefined, and the upper
// bound is held at 30, one step below the largest count (31) they still accept,
// as a conservative margin rather than a hard hardware requirement. Out-of-domain
// inputs route to requantizeGo, which handles them with full-width Go arithmetic.
func requantizeOutOfContract(multiplier int32, shift int) bool {
	return multiplier == math.MinInt32 || shift < -31 || shift > 30
}

// quantizeGo is the source of truth for Quantize: divide by scale, round to
// nearest even, add the zero point and saturate to int8. NaN maps to zeroPoint;
// +Inf saturates to 127 and -Inf to -128. Rounding then clamping is equivalent
// to the kernels' clamp-then-round because round-to-even is monotone and the
// bounds are exactly representable integers.
func quantizeGo(dst []int8, src []float32, scale float32, zeroPoint int8) {
	zp := int(zeroPoint)
	lo, hi := float64(math.MinInt8-zp), float64(math.MaxInt8-zp)
	for i := range dst {
		q := src[i] / scale
		if q != q { // NaN
			dst[i] = zeroPoint
			continue
		}
		r := math.RoundToEven(float64(q))
		dst[i] = int8(int(min(max(r, lo), hi)) + zp)
	}
}

// dequantizeGo is the source of truth for Dequantize: subtract the zero point
// (exact int) then multiply by scale (the single rounding). |q - zeroPoint| is
// at most 255, so the int-to-float conversion is exact.
func dequantizeGo(dst []float32, src []int8, scale float32, zeroPoint int8) {
	zp := int32(zeroPoint)
	for i := range dst {
		dst[i] = float32(int32(src[i])-zp) * scale
	}
}

// requantizeClampAdd finishes the requantize epilogue in int32 space: it clamps
// the rounded accumulator z to [MinInt8-zeroPoint, MaxInt8-zeroPoint] and only
// then adds zeroPoint, so the result lands in [-128, 127] with no wraparound.
// The whole computation stays in int32 and is bit-identical to the SIMD kernels,
// which reach the same int8 by different routes: the AVX2 kernel clamps z to
// [MinInt8-zeroPoint, MaxInt8-zeroPoint] then adds (VPMAXSD/VPMINSD then VPADDD),
// while the NEON kernel adds with a saturating SQADD then narrow-saturates in
// SQXTN. Adding first in a platform int (as clampI8(int(z)+zp) did) overflows on
// 32-bit-int targets (386, 32-bit ARM) when z is near MaxInt32 and zeroPoint is
// positive: the sum wraps negative and clamps to -128 instead of +127. Doing the
// arithmetic in int32 keeps every intermediate in range on every platform, so the
// Go reference stays bit-exact with the kernels.
func requantizeClampAdd(z int32, zeroPoint int8) int8 {
	lo := int32(math.MinInt8) - int32(zeroPoint)
	hi := int32(math.MaxInt8) - int32(zeroPoint)
	if z < lo {
		z = lo
	} else if z > hi {
		z = hi
	}
	return int8(z + int32(zeroPoint))
}

// requantizeGo is the source of truth for Requantize: the gemmlowp
// double-rounding epilogue (left shift, SaturatingRoundingDoublingHighMul,
// RoundingDivideByPOT), then add the zero point and saturate to int8. shift > 0
// left-shifts the accumulator before the multiply; shift < 0 rounding-divides
// after it.
func requantizeGo(dst []int8, acc []int32, multiplier int32, shift int, zeroPoint int8) {
	left := uint(0)
	if shift > 0 {
		left = uint(shift)
	}
	right := 0
	if shift < 0 {
		right = -shift
	}
	for i := range dst {
		x := acc[i] << left // wrapping int32 left shift
		y := srdhm(x, multiplier)
		z := rdbpot(y, right)
		dst[i] = requantizeClampAdd(z, zeroPoint)
	}
}
