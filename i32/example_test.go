package i32_test

import (
	"fmt"
	"math"

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

// ExampleSum shows the wrapping int32 total of a slice.
func ExampleSum() {
	a := []int32{5, -3, 10, 2}

	fmt.Println(i32.Sum(a))
	// Output: 14
}

// ExampleAbs shows the wrapping absolute value: |MinInt32| does not fit
// int32, so that one input wraps in place.
func ExampleAbs() {
	a := []int32{-5, 3, math.MinInt32}
	dst := make([]int32, len(a))

	i32.Abs(dst, a)
	fmt.Println(dst)
	// Output: [5 3 -2147483648]
}

// ExampleMinMax shows the signed min/max reduction over a slice.
func ExampleMinMax() {
	res := []int32{3, -7, 0, 42, -1}

	minVal, maxVal := i32.MinMax(res)
	fmt.Println(minVal, maxVal)
	// Output: -7 42
}
