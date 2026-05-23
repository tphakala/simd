package f64

import (
	"math"
	"testing"
)

func TestSubFromScalar(t *testing.T) {
	src := []float64{1, -2, 3.5, 10}
	dst := make([]float64, len(src))
	SubFromScalar(dst, src, 7.0)

	want := []float64{6, 9, 3.5, -3}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("SubFromScalar()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestSubFromScalar_EmptySlices(t *testing.T) {
	SubFromScalar(nil, nil, 1)
	SubFromScalar([]float64{}, []float64{1, 2}, 1)
	SubFromScalar([]float64{1, 2}, nil, 1)
}

func TestSubFromScalar_LongerInputs(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 127, 128} {
		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i) - 50
		}
		dst := make([]float64, n)
		SubFromScalar(dst, src, 3.5)
		for i, v := range src {
			want := 3.5 - v
			if math.Abs(dst[i]-want) > 1e-12 {
				t.Fatalf("n=%d: SubFromScalar[%d]=%v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestWeightedSum(t *testing.T) {
	weights := []float64{0.5, -1, 2, 0.25}
	src := []float64{8, 3, -2, 4}

	got := WeightedSum(weights, src)
	want := 0.5*8 + (-1)*3 + 2*(-2) + 0.25*4
	if got != want {
		t.Errorf("WeightedSum() = %v, want %v", got, want)
	}

	if WeightedSum(nil, src) != 0 {
		t.Errorf("WeightedSum(nil, src) should return 0")
	}
	if WeightedSum(weights, nil) != 0 {
		t.Errorf("WeightedSum(w, nil) should return 0")
	}
}

func TestSumOfSquares(t *testing.T) {
	src := []float64{3, 4, -12}
	got := SumOfSquares(src)
	want := 9.0 + 16.0 + 144.0
	if got != want {
		t.Errorf("SumOfSquares() = %v, want %v", got, want)
	}

	if SumOfSquares(nil) != 0 {
		t.Errorf("SumOfSquares(nil) should return 0")
	}
	if SumOfSquares([]float64{}) != 0 {
		t.Errorf("SumOfSquares([]) should return 0")
	}
}
