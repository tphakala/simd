package f64_test

import (
	"fmt"

	"github.com/tphakala/simd/f64"
)

func ExampleDotProduct() {
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}

	result := f64.DotProduct(a, b)
	fmt.Printf("%.0f\n", result)
	// Output: 70
}

func ExampleAdd() {
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	dst := make([]float64, len(a))

	f64.Add(dst, a, b)
	fmt.Println(dst)
	// Output: [6 8 10 12]
}

func ExampleMul() {
	a := []float64{1, 2, 3, 4}
	b := []float64{2, 2, 2, 2}
	dst := make([]float64, len(a))

	f64.Mul(dst, a, b)
	fmt.Println(dst)
	// Output: [2 4 6 8]
}

func ExampleScale() {
	a := []float64{1, 2, 3, 4}
	dst := make([]float64, len(a))

	f64.Scale(dst, a, 3.0)
	fmt.Println(dst)
	// Output: [3 6 9 12]
}

func ExampleSum() {
	a := []float64{1, 2, 3, 4, 5}

	result := f64.Sum(a)
	fmt.Printf("%.0f\n", result)
	// Output: 15
}

func ExampleMean() {
	a := []float64{2, 4, 6, 8, 10}

	result := f64.Mean(a)
	fmt.Printf("%.0f\n", result)
	// Output: 6
}

func ExampleNormalize() {
	a := []float64{3, 4} // 3-4-5 triangle
	dst := make([]float64, len(a))

	f64.Normalize(dst, a)
	fmt.Printf("[%.1f %.1f]\n", dst[0], dst[1])
	// Output: [0.6 0.8]
}

func ExampleEuclideanDistance() {
	a := []float64{0, 0}
	b := []float64{3, 4}

	result := f64.EuclideanDistance(a, b)
	fmt.Printf("%.0f\n", result)
	// Output: 5
}

func ExampleClamp() {
	a := []float64{-5, 0, 5, 10, 15}
	dst := make([]float64, len(a))

	f64.Clamp(dst, a, 0, 10)
	fmt.Println(dst)
	// Output: [0 0 5 10 10]
}

func ExampleFMA() {
	a := []float64{1, 2, 3}
	b := []float64{2, 2, 2}
	c := []float64{1, 1, 1}
	dst := make([]float64, len(a))

	// dst[i] = a[i] * b[i] + c[i]
	f64.FMA(dst, a, b, c)
	fmt.Println(dst)
	// Output: [3 5 7]
}

func ExampleMaxAbs() {
	a := []float64{1, -7, 3, -2}

	result := f64.MaxAbs(a)
	fmt.Printf("%.0f\n", result)
	// Output: 7
}

func ExampleConvolveValidMaxAbs() {
	signal := []float64{1, -2, 3, -4, 5}
	kernel := []float64{1, -1}

	// Peak of the valid-correlation output, no scratch buffer.
	peak := f64.ConvolveValidMaxAbs(signal, kernel)
	fmt.Printf("%.0f\n", peak)
	// Output: 9
}

func ExampleConvolveValidMaxAbsMulti() {
	signal := []float64{1, -2, 3, -4, 5}
	kernels := [][]float64{
		{1, 1},
		{1, -1},
	}

	// Single peak across every kernel's output (polyphase true-peak).
	peak := f64.ConvolveValidMaxAbsMulti(signal, kernels)
	fmt.Printf("%.0f\n", peak)
	// Output: 9
}
