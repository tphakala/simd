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
