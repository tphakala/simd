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

// ExampleSub shows the element-wise difference dst[i] = a[i] - b[i].
func ExampleSub() {
	a := []int32{10, 20, 30}
	b := []int32{1, 2, 3}
	dst := make([]int32, len(a))

	i32.Sub(dst, a, b)
	fmt.Println(dst)
	// Output: [9 18 27]
}

// ExampleMinMax shows the signed min/max reduction over a slice.
func ExampleMinMax() {
	res := []int32{3, -7, 0, 42, -1}

	minVal, maxVal := i32.MinMax(res)
	fmt.Println(minVal, maxVal)
	// Output: -7 42
}
