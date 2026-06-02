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
