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
