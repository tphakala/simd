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
