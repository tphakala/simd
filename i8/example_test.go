package i8_test

import (
	"fmt"

	"github.com/tphakala/simd/i8"
)

func ExampleDotProduct() {
	a := []int8{1, 2, 3, 4, -5}
	b := []int8{10, 20, 30, 40, 50}
	// int32 accumulation: 10 + 40 + 90 + 160 - 250 = 50.
	fmt.Println(i8.DotProduct(a, b))
	// Output: 50
}

func ExampleAddSaturate() {
	dst := make([]int8, 3)
	i8.AddSaturate(dst, []int8{100, -100, 1}, []int8{100, -100, 2})
	// Saturates to the int8 range instead of wrapping.
	fmt.Println(dst)
	// Output: [127 -128 3]
}

func ExampleMinMax() {
	lo, hi := i8.MinMax([]int8{0, -128, 127, 3, -1})
	fmt.Println(lo, hi)
	// Output: -128 127
}
