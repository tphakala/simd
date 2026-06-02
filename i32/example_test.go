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
