//go:build amd64 || arm64

package i16

import "math"

// Shared helpers for the architecture-specific SIMD parity tests
// (i16_amd64_test.go, i16_arm64_test.go).

// fillPattern fills a/b with values that exercise sign and high bits so a
// lane-corrupting kernel cannot hide behind small positive integers.
func fillPattern(a, b []int16) {
	for i := range a {
		a[i] = int16(i*2) ^ math.MinInt16
		b[i] = int16(-(i*2 + 1))
	}
	if len(a) > 0 {
		a[0] = math.MinInt16
		b[0] = math.MaxInt16
	}
}

// paritySizes straddle every SIMD block size in this package (8 lanes on NEON
// and SSE2, 16 on AVX2) so each vector body and its scalar tail are covered.
var paritySizes = []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 100, 1000, 1023, 1024, 1025}
