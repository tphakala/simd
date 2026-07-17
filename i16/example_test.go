package i16_test

import (
	"fmt"

	"github.com/tphakala/simd/i16"
)

func ExampleInterleave2() {
	left := []int16{1, 2, 3}
	right := []int16{-1, -2, -3}
	stereo := make([]int16, len(left)*2)

	i16.Interleave2(stereo, left, right)
	fmt.Println(stereo)
	// Output: [1 -1 2 -2 3 -3]
}

func ExampleDeinterleave2() {
	stereo := []int16{1, -1, 2, -2, 3, -3}
	left := make([]int16, len(stereo)/2)
	right := make([]int16, len(stereo)/2)

	i16.Deinterleave2(left, right, stereo)
	fmt.Println(left, right)
	// Output: [1 2 3] [-1 -2 -3]
}

func ExampleDotProduct() {
	// Q15 filter coefficients against a frame of int16 samples: the products
	// are widened and accumulated in int32.
	samples := []int16{1000, -2000, 3000, -4000}
	coeffs := []int16{16384, 8192, -16384, 4096}

	fmt.Println(i16.DotProduct(samples, coeffs))
	// Output: -65536000
}

func ExampleMulQ15() {
	// Apply a Q15 gain of 0.5 (16384/32768) to a frame of samples. The
	// product rounds to nearest: 1 * 16384 rounds up to 1 where a
	// truncating Q15 multiply would give 0.
	samples := []int16{1000, -1000, 32767, -32768, 1}
	gain := []int16{16384, 16384, 16384, 16384, 16384}
	out := make([]int16, len(samples))

	i16.MulQ15(out, samples, gain)
	fmt.Println(out)
	// Output: [500 -500 16384 -16384 1]
}

func ExampleAbs() {
	// Rectify a frame for an envelope follower. The negation wraps rather
	// than saturating, so -32768 maps to itself: |-32768| = 32768 does not
	// fit int16. This is the opposite of i8.Abs, which saturates by design.
	samples := []int16{-1000, 3000, -32768, 0}
	out := make([]int16, len(samples))

	i16.Abs(out, samples)
	fmt.Println(out)
	// Output: [1000 3000 -32768 0]
}

func ExampleMaxAbs() {
	// Headroom probe before applying gain: |-32768| = 32768 does not fit
	// int16, which is why the result is an int.
	frame := []int16{100, -32768, 3000, 15}

	fmt.Println(i16.MaxAbs(frame))
	// Output: 32768
}

func ExampleXCorr() {
	// Correlate a short pattern against a longer signal at every lag. The
	// pattern occurs at lag 2, which is where the correlation peaks.
	pattern := []int16{1000, -2000, 1000}
	signal := []int16{0, 0, 1000, -2000, 1000, 0}

	lags := make([]int32, len(signal)-len(pattern)+1)
	i16.XCorr(lags, pattern, signal)

	fmt.Println(lags)
	// Output: [1000000 -4000000 6000000 -4000000]
}
