package f64

import (
	"math"
	"testing"
)

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

func TestRound_EmptySlices(t *testing.T) {
	Round(nil, nil)
	Round([]float64{}, []float64{1, 2, 3})
	Round([]float64{1, 2, 3}, nil)
}

func TestRound_FPEdgeCases(t *testing.T) {
	// Inputs that would break a naive trunc(abs(x)+0.5) implementation
	// because abs(x)+0.5 rounds to a value whose trunc is one too large.
	belowHalf := math.Nextafter(0.5, 0)              // largest float < 0.5
	negBelowHalf := math.Nextafter(-0.5, 0)          // smallest-magnitude float > -0.5
	belowOneHalf := math.Nextafter(1.5, 1)           // largest float < 1.5
	negBelowOneHalf := math.Nextafter(-1.5, -1)      // smallest-magnitude float > -1.5

	src := []float64{
		belowHalf,
		negBelowHalf,
		belowOneHalf,
		negBelowOneHalf,
		0.5,
		-0.5,
		1.5,
		-1.5,
		math.Copysign(0, 1),
		math.Copysign(0, -1),
	}
	dst := make([]float64, len(src))
	Round(dst, src)

	for i, v := range src {
		want := math.Round(v)
		if dst[i] != want || math.Signbit(dst[i]) != math.Signbit(want) {
			t.Errorf("Round(%g)=%g (signbit=%v), want %g (signbit=%v)",
				v, dst[i], math.Signbit(dst[i]), want, math.Signbit(want))
		}
	}
}

func TestRound_LongerInputs(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 127, 128} {
		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i)*0.5 - 4
		}
		dst := make([]float64, n)
		Round(dst, src)
		for i, v := range src {
			want := math.Round(v)
			if dst[i] != want {
				t.Fatalf("n=%d: Round[%d]=%v, want %v", n, i, dst[i], want)
			}
		}
	}
}
