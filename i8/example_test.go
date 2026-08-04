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

func ExampleQuantize() {
	// scale 0.5, zeroPoint 0: q = round_to_even(src/0.5) + 0, clamped.
	src := []float32{0.25, 0.75, 1.25, -0.25, 64.0, -100.0}
	dst := make([]int8, len(src))
	i8.Quantize(dst, src, 0.5, 0)
	// 0.5 and 2.5 round to even (0 and 2); 128 and -200 saturate.
	fmt.Println(dst)
	// Output: [0 2 2 0 127 -128]
}

func ExampleDequantize() {
	// dst = float32(src - zeroPoint) * scale.
	dst := make([]float32, 4)
	i8.Dequantize(dst, []int8{5, -128, 127, 3}, 0.5, 3)
	fmt.Println(dst)
	// Output: [1 -65.5 62 0]
}

func ExampleRequantize() {
	// multiplier 0x40000000 is 0.5 in Q31 and shift 0, so each accumulator is
	// halved with round-half-up, then the zero point is added.
	acc := []int32{5, 4, -3, -4, 200}
	dst := make([]int8, len(acc))
	i8.Requantize(dst, acc, 0x40000000, 0, 0)
	fmt.Println(dst)
	// Output: [3 2 -1 -2 100]
}
