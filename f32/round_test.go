package f32

import (
	"math"
	"testing"
)

func TestRound(t *testing.T) {
	src := []float32{-2.5, -1.5, -0.5, 0.4, 0.5, 1.5, 2.5, 3.49, 3.5}
	dst := make([]float32, len(src))

	Round(dst, src)

	for i := range src {
		want := float32(math.Round(float64(src[i])))
		if dst[i] != want {
			t.Errorf("Round()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestRound_SpecialValues(t *testing.T) {
	src := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	dst := make([]float32, len(src))

	Round(dst, src)

	if !math.IsNaN(float64(dst[0])) {
		t.Errorf("Round(NaN) = %v, want NaN", dst[0])
	}
	if !math.IsInf(float64(dst[1]), 1) {
		t.Errorf("Round(+Inf) = %v, want +Inf", dst[1])
	}
	if !math.IsInf(float64(dst[2]), -1) {
		t.Errorf("Round(-Inf) = %v, want -Inf", dst[2])
	}
}

func TestRound_EmptySlices(_ *testing.T) {
	Round(nil, nil)
	Round([]float32{}, []float32{1, 2, 3})
	Round([]float32{1, 2, 3}, nil)
}

func TestRound_FPEdgeCases(t *testing.T) {
	// Inputs that would break a naive trunc(abs(x)+0.5) implementation because
	// abs(x)+0.5 rounds to a value whose trunc is one too large. Nextafter is
	// computed in float64 then narrowed; the narrowing stays below the tie so
	// the float32 value is still strictly below 0.5/1.5.
	belowHalf := math.Float32frombits(math.Float32bits(0.5) - 1)        // largest float32 < 0.5
	negBelowHalf := math.Float32frombits(math.Float32bits(-0.5) - 1)    // smallest-magnitude float32 > -0.5
	belowOneHalf := math.Float32frombits(math.Float32bits(1.5) - 1)     // largest float32 < 1.5
	negBelowOneHalf := math.Float32frombits(math.Float32bits(-1.5) - 1) // smallest-magnitude float32 > -1.5

	src := []float32{
		belowHalf,
		negBelowHalf,
		belowOneHalf,
		negBelowOneHalf,
		0.5,
		-0.5,
		1.5,
		-1.5,
		float32(math.Copysign(0, 1)),
		float32(math.Copysign(0, -1)),
	}
	dst := make([]float32, len(src))
	Round(dst, src)

	for i, v := range src {
		want := float32(math.Round(float64(v)))
		if dst[i] != want || math.Signbit(float64(dst[i])) != math.Signbit(float64(want)) {
			t.Errorf("Round(%g)=%g (signbit=%v), want %g (signbit=%v)",
				v, dst[i], math.Signbit(float64(dst[i])), want, math.Signbit(float64(want)))
		}
	}
}

func TestRound_LongerInputs(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 127, 128} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i)*0.5 - 4
		}
		dst := make([]float32, n)
		Round(dst, src)
		for i, v := range src {
			want := float32(math.Round(float64(v)))
			if dst[i] != want {
				t.Fatalf("n=%d: Round[%d]=%v, want %v", n, i, dst[i], want)
			}
		}
	}
}

// TestRound_AllocFree pins the zero-allocation contract: Round writes into the
// caller-provided destination and must not allocate.
func TestRound_AllocFree(t *testing.T) {
	src := make([]float32, 1024)
	dst := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i%2400-1200) / 7.0
	}
	if a := testing.AllocsPerRun(50, func() { Round(dst, src) }); a != 0 {
		t.Errorf("Round allocated %v times per run, want 0", a)
	}
}
