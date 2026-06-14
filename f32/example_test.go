package f32_test

import (
	"fmt"

	"github.com/tphakala/simd/f32"
)

func ExampleDotProduct() {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}

	result := f32.DotProduct(a, b)
	fmt.Printf("%.0f\n", result)
	// Output: 70
}

func ExampleAdd() {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	dst := make([]float32, len(a))

	f32.Add(dst, a, b)
	fmt.Println(dst)
	// Output: [6 8 10 12]
}

func ExampleMul() {
	a := []float32{1, 2, 3, 4}
	b := []float32{2, 2, 2, 2}
	dst := make([]float32, len(a))

	f32.Mul(dst, a, b)
	fmt.Println(dst)
	// Output: [2 4 6 8]
}

func ExampleScale() {
	a := []float32{1, 2, 3, 4}
	dst := make([]float32, len(a))

	f32.Scale(dst, a, 3.0)
	fmt.Println(dst)
	// Output: [3 6 9 12]
}

func ExampleSum() {
	a := []float32{1, 2, 3, 4, 5}

	result := f32.Sum(a)
	fmt.Printf("%.0f\n", result)
	// Output: 15
}

func ExampleClamp() {
	a := []float32{-5, 0, 5, 10, 15}
	dst := make([]float32, len(a))

	f32.Clamp(dst, a, 0, 10)
	fmt.Println(dst)
	// Output: [0 0 5 10 10]
}

func ExampleFMA() {
	a := []float32{1, 2, 3}
	b := []float32{2, 2, 2}
	c := []float32{1, 1, 1}
	dst := make([]float32, len(a))

	// dst[i] = a[i] * b[i] + c[i]
	f32.FMA(dst, a, b, c)
	fmt.Println(dst)
	// Output: [3 5 7]
}

func ExampleMaxAbs() {
	a := []float32{1, -7, 3, -2}

	result := f32.MaxAbs(a)
	fmt.Printf("%.0f\n", result)
	// Output: 7
}

func ExampleConvolveValidMaxAbs() {
	signal := []float32{1, -2, 3, -4, 5}
	kernel := []float32{1, -1}

	// Peak of the valid-correlation output, no scratch buffer.
	peak := f32.ConvolveValidMaxAbs(signal, kernel)
	fmt.Printf("%.0f\n", peak)
	// Output: 9
}

func ExampleConvolveValidMaxAbsMulti() {
	signal := []float32{1, -2, 3, -4, 5}
	kernels := [][]float32{
		{1, 1},
		{1, -1},
	}

	// Single peak across every kernel's output (polyphase true-peak).
	peak := f32.ConvolveValidMaxAbsMulti(signal, kernels)
	fmt.Printf("%.0f\n", peak)
	// Output: 9
}
