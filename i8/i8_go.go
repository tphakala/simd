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
