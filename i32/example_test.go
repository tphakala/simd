package i32_test

import (
	"fmt"

	"github.com/tphakala/simd/i32"
)

func ExampleInterleave2() {
	left := []int32{1, 2, 3}
	right := []int32{-1, -2, -3}
	stereo := make([]int32, len(left)*2)

	i32.Interleave2(stereo, left, right)
	fmt.Println(stereo)
	// Output: [1 -1 2 -2 3 -3]
}

func ExampleDeinterleave2() {
	stereo := []int32{1, -1, 2, -2, 3, -3}
	left := make([]int32, len(stereo)/2)
	right := make([]int32, len(stereo)/2)

	i32.Deinterleave2(left, right, stereo)
	fmt.Println(left, right)
	// Output: [1 2 3] [-1 -2 -3]
}

func ExampleMidSideEncode() {
	left := []int32{5, 2, -3}
	right := []int32{5, 4, 1}
	mid := make([]int32, len(left))
	side := make([]int32, len(left))

	i32.MidSideEncode(mid, side, left, right)
	fmt.Println(mid, side)
	// Output: [5 3 -1] [0 -2 -4]
}

func ExampleMidSideDecode() {
	mid := []int32{5, 3, -1}
	side := []int32{0, -2, -4}
	left := make([]int32, len(mid))
	right := make([]int32, len(mid))

	i32.MidSideDecode(left, right, mid, side)
	fmt.Println(left, right)
	// Output: [5 2 -3] [5 4 1]
}

// ExampleSub shows the FLAC left/side encode: the side channel is left - right.
func ExampleSub() {
	left := []int32{10, 20, 30}
	right := []int32{1, 2, 3}
	side := make([]int32, len(left))

	i32.Sub(side, left, right)
	fmt.Println(side)
	// Output: [9 18 27]
}

// ExampleDiff1 shows the order-1 fixed-predictor residual. dst[0] is the
// verbatim warm-up sample; the rest are src[n]-src[n-1].
func ExampleDiff1() {
	samples := []int32{10, 13, 13, 8, 20}
	res := make([]int32, len(samples))

	i32.Diff1(res, samples)
	fmt.Println(res)
	// Output: [10 3 0 -5 12]
}

// ExampleRestore1 shows decode-side restoration: Restore1 inverts Diff1,
// reconstructing the samples from the [warm-up | residuals] layout.
func ExampleRestore1() {
	res := []int32{10, 3, 0, -5, 12}
	samples := make([]int32, len(res))

	i32.Restore1(samples, res)
	fmt.Println(samples)
	// Output: [10 13 13 8 20]
}

// ExampleLPCResidualEncode shows the quantized-LPC encode FIR. With order-2
// coefficients {2,-1} and shift 1 the prediction is (2*s[i-1] - s[i-2]) >> 1;
// res holds the order-2 warm-up samples verbatim, then samples[i] - prediction.
func ExampleLPCResidualEncode() {
	samples := []int32{10, 20, 34, 50}
	coeffs := []int32{2, -1}
	res := make([]int32, len(samples))

	i32.LPCResidualEncode(res, samples, coeffs, 1)
	fmt.Println(res)
	// Output: [10 20 19 26]
}

// ExampleLPCRestore shows decode-side restoration: LPCRestore inverts
// LPCResidualEncode with the same coefficients and shift, reconstructing the
// samples from the [warm-up | residual] layout.
func ExampleLPCRestore() {
	res := []int32{10, 20, 19, 26}
	coeffs := []int32{2, -1}
	samples := make([]int32, len(res))

	i32.LPCRestore(samples, res, coeffs, 1)
	fmt.Println(samples)
	// Output: [10 20 34 50]
}

// ExampleRiceSums shows the per-parameter unary-bit sums. The residuals fold to
// the zigzag symbols 0,1,2,3,4; sums[k] is the sum of those symbols shifted
// right by k, the data-dependent part of the Rice code length for parameter k.
func ExampleRiceSums() {
	res := []int32{0, -1, 1, -2, 2} // zigzag -> 0,1,2,3,4
	sums := make([]uint64, 4)

	i32.RiceSums(sums, res)
	fmt.Println(sums)
	// Output: [10 4 1 0]
}

// ExampleRiceBestParam shows the Rice parameter search. For these residuals the
// total code cost is minimized at parameter 1, which needs 14 bits:
// cost(0)=15, cost(1)=14, cost(2)=16.
func ExampleRiceBestParam() {
	res := []int32{0, -1, 1, -2, 2}

	param, bits := i32.RiceBestParam(res, 14)
	fmt.Println(param, bits)
	// Output: 1 14
}
