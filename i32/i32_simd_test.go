//go:build amd64 || arm64

package i32

import "math"

// Shared helpers for the architecture-specific SIMD parity tests
// (i32_amd64_test.go, i32_arm64_test.go).

// fillPattern fills a/b with values that exercise sign and high bits so a
// lane-corrupting kernel cannot hide behind small positive integers.
func fillPattern(a, b []int32) {
	for i := range a {
		a[i] = int32(i*2) ^ math.MinInt32
		b[i] = int32(-(i*2 + 1))
	}
	if len(a) > 0 {
		a[0] = math.MinInt32
		b[0] = math.MaxInt32
	}
}

// paritySizes straddle both SIMD block sizes (4 lanes on NEON, 8 on AVX) so the
// vector body and the scalar tail are both covered on either architecture.
var paritySizes = []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000, 1023, 1024, 1025}

// fillRiceRes fills res with sign-exercising values, pinning the extremes so the
// zigzag overflow at math.MinInt32 (-> 2^32-1) and the unsigned widening are
// covered alongside the block-straddling sizes.
func fillRiceRes(res []int32) {
	for i := range res {
		res[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(res) > 3 {
		res[0] = math.MinInt32
		res[1] = math.MaxInt32
		res[2] = -1
		res[3] = 0
	}
}
