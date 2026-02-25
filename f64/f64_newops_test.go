package f64

import (
	"math"
	"testing"
)

const trigApproxTol = 1e-12

func TestSin(t *testing.T) {
	src := []float64{0, math.Pi / 6, math.Pi / 2, math.Pi}
	dst := make([]float64, len(src))

	Sin(dst, src)

	for i := range src {
		want := math.Sin(src[i])
		if math.Abs(dst[i]-want) > trigApproxTol {
			t.Errorf("Sin()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestCos(t *testing.T) {
	src := []float64{0, math.Pi / 3, math.Pi / 2, math.Pi}
	dst := make([]float64, len(src))

	Cos(dst, src)

	for i := range src {
		want := math.Cos(src[i])
		if math.Abs(dst[i]-want) > trigApproxTol {
			t.Errorf("Cos()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestSinCos(t *testing.T) {
	src := []float64{0, math.Pi / 4, math.Pi / 2, math.Pi}
	sinDst := make([]float64, len(src))
	cosDst := make([]float64, len(src))

	SinCos(sinDst, cosDst, src)

	for i := range src {
		wantSin, wantCos := math.Sincos(src[i])
		if math.Abs(sinDst[i]-wantSin) > trigApproxTol {
			t.Errorf("SinCos() sin[%d] = %v, want %v", i, sinDst[i], wantSin)
		}
		if math.Abs(cosDst[i]-wantCos) > trigApproxTol {
			t.Errorf("SinCos() cos[%d] = %v, want %v", i, cosDst[i], wantCos)
		}
	}
}

func TestSinCos_DifferentLengths(t *testing.T) {
	src := []float64{0, math.Pi / 4, math.Pi / 2, math.Pi}
	sinDst := make([]float64, 2)
	cosDst := make([]float64, 3)

	SinCos(sinDst, cosDst, src)

	for i := range sinDst {
		wantSin, wantCos := math.Sincos(src[i])
		if math.Abs(sinDst[i]-wantSin) > trigApproxTol {
			t.Errorf("SinCos() sin[%d] = %v, want %v", i, sinDst[i], wantSin)
		}
		if math.Abs(cosDst[i]-wantCos) > trigApproxTol {
			t.Errorf("SinCos() cos[%d] = %v, want %v", i, cosDst[i], wantCos)
		}
	}
}

func TestSinCos_SpecialValues(t *testing.T) {
	src := []float64{0, 1e7, -1e7, math.NaN(), math.Inf(1), math.Inf(-1)}
	sinDst := make([]float64, len(src))
	cosDst := make([]float64, len(src))

	SinCos(sinDst, cosDst, src)

	for i := range src {
		wantSin, wantCos := math.Sincos(src[i])
		if math.IsNaN(wantSin) {
			if !math.IsNaN(sinDst[i]) {
				t.Errorf("SinCos() sin[%d] = %v, want NaN", i, sinDst[i])
			}
		} else if math.Abs(sinDst[i]-wantSin) > trigApproxTol {
			t.Errorf("SinCos() sin[%d] = %v, want %v", i, sinDst[i], wantSin)
		}

		if math.IsNaN(wantCos) {
			if !math.IsNaN(cosDst[i]) {
				t.Errorf("SinCos() cos[%d] = %v, want NaN", i, cosDst[i])
			}
		} else if math.Abs(cosDst[i]-wantCos) > trigApproxTol {
			t.Errorf("SinCos() cos[%d] = %v, want %v", i, cosDst[i], wantCos)
		}
	}
}

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

func TestRound(t *testing.T) {
	src := []float64{-2.5, -1.5, -0.5, 0.4, 0.5, 1.5, 2.5, 3.49, 3.5}
	dst := make([]float64, len(src))

	Round(dst, src)

	for i := range src {
		want := math.Round(src[i])
		if dst[i] != want {
			t.Errorf("Round()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestRound_SpecialValues(t *testing.T) {
	src := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	dst := make([]float64, len(src))

	Round(dst, src)

	if !math.IsNaN(dst[0]) {
		t.Errorf("Round(NaN) = %v, want NaN", dst[0])
	}
	if !math.IsInf(dst[1], 1) {
		t.Errorf("Round(+Inf) = %v, want +Inf", dst[1])
	}
	if !math.IsInf(dst[2], -1) {
		t.Errorf("Round(-Inf) = %v, want -Inf", dst[2])
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
}

func TestGather(t *testing.T) {
	src := []float64{10, 20, 30, 40, 50}
	indices := []int{4, 0, 3, 1}
	dst := make([]float64, len(indices))

	Gather(dst, src, indices)

	want := []float64{50, 10, 40, 20}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Gather()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestGather_DifferentLengths(t *testing.T) {
	src := []float64{1, 2, 3, 4}
	indices := []int{3, 2, 1}
	dst := make([]float64, 2)

	Gather(dst, src, indices)

	want := []float64{4, 3}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Gather()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestGather_PanicsNegativeIndex(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Gather() did not panic for negative index")
		}
	}()

	Gather(make([]float64, 1), []float64{1, 2, 3}, []int{-1})
}

func TestGather_PanicsOutOfRange(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Gather() did not panic for out-of-range index")
		}
	}()

	Gather(make([]float64, 1), []float64{1, 2, 3}, []int{3})
}
