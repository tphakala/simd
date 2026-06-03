package f64

import (
	"math"
	"testing"
)

// These tests exercise the Go fallback implementations and the BCE-hint
// guard branches that aren't normally hit on AVX/NEON-capable test hosts.

func TestRound64Go(t *testing.T) {
	src := []float64{-2.5, -1.5, -0.5, 0.4, 0.5, 1.5, 2.5}
	dst := make([]float64, len(src))
	round64Go(dst, src)
	for i, v := range src {
		want := math.Round(v)
		if dst[i] != want {
			t.Errorf("round64Go[%d](%v) = %v, want %v", i, v, dst[i], want)
		}
	}
}

func TestSubFromScalarGo(t *testing.T) {
	a := []float64{1, -2, 3.5}
	dst := make([]float64, len(a))
	subFromScalarGo(dst, a, 7.0)
	want := []float64{6, 9, 3.5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("subFromScalarGo[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// emptyDstGuards verifies that every BCE-hint-guarded Go fallback returns
// safely (no panic) when len(dst) == 0, even with a non-empty src argument.
// This exercises the `if len(dst) == 0 { return }` branch that's otherwise
// only hit at boundaries.
func TestGoFallbacks_EmptyDst(_ *testing.T) {
	src := []float64{1, 2, 3}
	dst := []float64{}

	addGo(dst, src, src)
	subGo(dst, src, src)
	mulGo(dst, src, src)
	divGo(dst, src, src)
	scaleGo(dst, src, 2)
	addScalarGo(dst, src, 1)
	subFromScalarGo(dst, src, 1)
	absGo(dst, src)
	negGo(dst, src)
	fmaGo(dst, src, src, src)
	clampGo(dst, src, 0, 1)
	sqrt64Go(dst, src)
	round64Go(dst, src)
	reciprocal64Go(dst, src)
	addScaledGo64(dst, 1, src)
	sigmoid64Go(dst, src)
	relu64Go(dst, src)
	clampScale64Go(dst, src, 0, 1, 1)
	tanh64Go(dst, src)
	exp64Go(dst, src)
}

func TestRound64Go_EmptyDst(_ *testing.T) {
	// Hit the BCE-hint empty-slice branch with len(src) != 0.
	round64Go(nil, []float64{1, 2})
}

func TestSubFromScalarGo_EmptyDst(_ *testing.T) {
	subFromScalarGo(nil, []float64{1, 2}, 0)
}

// TestSigmoid64Go_Saturation hits both exponent-clamp branches (|x| beyond
// expOverflowThreshold) in the sigmoid64Go fallback and confirms it saturates
// smoothly: +large -> 1, -large -> a tiny positive value (NOT a hard 0).
func TestSigmoid64Go_Saturation(t *testing.T) {
	src := []float64{expOverflowThreshold + 1, -(expOverflowThreshold + 1), 0}
	dst := make([]float64, len(src))
	sigmoid64Go(dst, src)
	if dst[0] != 1.0 {
		t.Errorf("sigmoid64Go(+large) = %v, want 1", dst[0])
	}
	// With z = -x clamped to expOverflowThreshold, sigmoid(-large) is
	// 1/(1+exp(709)) ~ 1.2e-308: vanishingly small but strictly positive.
	wantNeg := 1.0 / (1.0 + math.Exp(expOverflowThreshold))
	if dst[1] != wantNeg {
		t.Errorf("sigmoid64Go(-large) = %v, want %v", dst[1], wantNeg)
	}
	if dst[2] != 0.5 {
		t.Errorf("sigmoid64Go(0) = %v, want 0.5", dst[2])
	}
}

// TestSigmoid64Go_NoFlushToZero guards the regression fixed here: the scalar
// fallback used to hard-clamp to 0 for x < -20, diverging from the accurate
// exp-based SIMD kernels (issue #33). It calls sigmoid64Go directly so it runs
// on every host, unlike the dispatched TestSigmoidAccuracy which only exercises
// whichever kernel the CPU selects (the AVX2 kernel on CI, never this fallback).
// sigmoid(-25) is ~1.39e-11; flushing it to 0 is a relative error of 1.
func TestSigmoid64Go_NoFlushToZero(t *testing.T) {
	src := []float64{-30, -27.5, -25, -22, -20.5}
	dst := make([]float64, len(src))
	sigmoid64Go(dst, src)
	for i, x := range src {
		want := 1.0 / (1.0 + math.Exp(-x))
		if dst[i] <= 0 {
			t.Errorf("sigmoid64Go(%v) = %v, want positive ~%v", x, dst[i], want)
			continue
		}
		if relErr := math.Abs(dst[i]-want) / want; relErr > 1e-9 {
			t.Errorf("sigmoid64Go(%v) = %v, want %v (relErr %v)", x, dst[i], want, relErr)
		}
	}
}

// TestExp64Go_Saturation hits the overflow / underflow clamp branches.
func TestExp64Go_Saturation(t *testing.T) {
	src := []float64{expOverflowThreshold + 1, -expOverflowThreshold - 1, 0}
	dst := make([]float64, len(src))
	exp64Go(dst, src)
	if dst[0] != math.Exp(expOverflowThreshold) {
		t.Errorf("exp64Go(+large) = %v, want exp(709)", dst[0])
	}
	if dst[1] != 0.0 {
		t.Errorf("exp64Go(-large) = %v, want 0", dst[1])
	}
	if dst[2] != 1.0 {
		t.Errorf("exp64Go(0) = %v, want 1", dst[2])
	}
}

func TestRelu64Go(t *testing.T) {
	src := []float64{-1, 0, 1, 2.5}
	dst := make([]float64, len(src))
	relu64Go(dst, src)
	want := []float64{0, 0, 1, 2.5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("relu64Go[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestClampScale64Go(t *testing.T) {
	src := []float64{-2, 0.5, 5}
	dst := make([]float64, len(src))
	clampScale64Go(dst, src, 0, 1, 2)
	want := []float64{0, 1, 2}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("clampScale64Go[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestTanh64Go(t *testing.T) {
	src := []float64{-1, 0, 1}
	dst := make([]float64, len(src))
	tanh64Go(dst, src)
	for i, v := range src {
		want := math.Tanh(v)
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Errorf("tanh64Go[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

