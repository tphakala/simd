package f32

import (
	"math"
	"testing"
)

func TestSubFromScalar(t *testing.T) {
	src := []float32{1, -2, 3.5, 10}
	dst := make([]float32, len(src))
	SubFromScalar(dst, src, 7.0)

	want := []float32{6, 9, 3.5, -3}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("SubFromScalar()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestSubFromScalar_EmptySlices(_ *testing.T) {
	SubFromScalar(nil, nil, 1)
	SubFromScalar([]float32{}, []float32{1, 2}, 1)
	SubFromScalar([]float32{1, 2}, nil, 1)
}

func TestSubFromScalar_LongerInputs(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 127, 128} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) - 50
		}
		dst := make([]float32, n)
		SubFromScalar(dst, src, 3.5)
		for i, v := range src {
			want := float32(3.5) - v
			if math.Abs(float64(dst[i]-want)) > 1e-4 {
				t.Fatalf("n=%d: SubFromScalar[%d]=%v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestWeightedSum(t *testing.T) {
	weights := []float32{0.5, -1, 2, 0.25}
	src := []float32{8, 3, -2, 4}

	got := WeightedSum(weights, src)
	want := float32(0.5*8 + (-1)*3 + 2*(-2) + 0.25*4)
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
	src := []float32{3, 4, -12}
	got := SumOfSquares(src)
	want := float32(9.0 + 16.0 + 144.0)
	if got != want {
		t.Errorf("SumOfSquares() = %v, want %v", got, want)
	}

	if SumOfSquares(nil) != 0 {
		t.Errorf("SumOfSquares(nil) should return 0")
	}
	if SumOfSquares([]float32{}) != 0 {
		t.Errorf("SumOfSquares([]) should return 0")
	}
}

// TestNewOps32_AllocFree pins the zero-allocation guarantee for the three
// f64-parity primitives added to f32. They compose already-dispatched kernels
// (dotProduct, neg32, addScalar), none of which allocate.
func TestNewOps32_AllocFree(t *testing.T) {
	src := make([]float32, 1024)
	weights := make([]float32, 1024)
	dst := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i%17) - 8
		weights[i] = float32(i%5) - 2
	}

	cases := []struct {
		name string
		fn   func()
	}{
		{"SubFromScalar", func() { SubFromScalar(dst, src, 3.5) }},
		{"WeightedSum", func() { _ = WeightedSum(weights, src) }},
		{"SumOfSquares", func() { _ = SumOfSquares(src) }},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, a)
		}
	}
}
