package f64

import (
	"math"
	"testing"
)

// These tests exercise the SIMD-dispatched paths (NEON on arm64; SSE2/AVX/AVX-512
// on amd64) at and beyond the vector width, including non-multiples of the width
// so both the main assembly loop and the scalar remainder run. Every result is
// cross-checked against the pure-Go fallback, which is the trusted reference and
// the code that actually runs on architectures without SIMD.
//
// This guards the gap described in issue #45: a kernel bug in the looping body
// stays invisible when a test uses fewer elements than the SIMD width, because
// the dispatch falls through to the Go fallback and silently produces the right
// answer. The core arithmetic and reductions are also pinned by the C-reference
// tests (f64_cref_test.go / reference_test.go); this file adds the missing direct
// Go-fallback parity for them and for the activation, scale, clamp, interleave and
// resampling kernels.

// vpLens spans every f64 SIMD stride: NEON (2), AVX (4) and AVX-512 (8), with
// non-multiples so the remainder path is always taken too.
var vpLens = []int{2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33}

func vpClose64(got, want float64) bool {
	d := math.Abs(got - want)
	return d <= 1e-9*math.Abs(want)+1e-12
}

// vpCloseActiv64 is the looser tolerance for the activation kernels, whose SIMD
// paths are polynomial approximations while the Go fallback calls the accurate
// math.Exp/math.Tanh. It is still far tighter than any block-boundary bug.
func vpCloseActiv64(got, want float64) bool {
	d := math.Abs(got - want)
	return d <= 1e-3*math.Abs(want)+1e-4
}

func vpSeq64(n, seed int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = float64((i*7+seed)%23) - 11
	}
	return s
}

func vpPos64(n, seed int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = float64((i*5+seed)%13) + 0.5
	}
	return s
}

func vpActiv64(n, seed int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = -4.0 + 0.37*float64((i*3+seed)%23)
	}
	return s
}

func vpCheck64(t *testing.T, name string, n int, got, want []float64, cmp func(a, b float64) bool) {
	t.Helper()
	for i := range got {
		if !cmp(got[i], want[i]) {
			t.Errorf("%s n=%d [%d] = %v, want %v (Go fallback)", name, n, i, got[i], want[i])
		}
	}
}

func TestVectorPathBinary64(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a, b []float64)
		gold func(dst, a, b []float64)
		posB bool
	}{
		{"Add", Add, addGo, false},
		{"Sub", Sub, subGo, false},
		{"Mul", Mul, mulGo, false},
		{"Div", Div, divGo, true},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpSeq64(n, 1)
			var b []float64
			if op.posB {
				b = vpPos64(n, 2)
			} else {
				b = vpSeq64(n, 2)
			}
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, a, b)
			op.gold(want, a, b)
			vpCheck64(t, op.name, n, got, want, vpClose64)
		}
	}
}

func TestVectorPathUnary64(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a []float64)
		gold func(dst, a []float64)
		pos  bool
	}{
		{"Abs", Abs, absGo, false},
		{"Neg", Neg, negGo, false},
		{"Sqrt", Sqrt, sqrt64Go, true},
		{"Reciprocal", Reciprocal, reciprocal64Go, true},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			var a []float64
			if op.pos {
				a = vpPos64(n, 3)
			} else {
				a = vpSeq64(n, 3)
			}
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, a)
			op.gold(want, a)
			vpCheck64(t, op.name, n, got, want, vpClose64)
		}
	}
}

// TestVectorPathRound64 uses non-tie fractional inputs so the result is identical
// regardless of the half-way rounding rule, and an exact match is required.
func TestVectorPathRound64(t *testing.T) {
	for _, n := range vpLens {
		src := make([]float64, n)
		for i := range src {
			frac := 0.3
			if i%2 == 1 {
				frac = 0.7
			}
			src[i] = float64((i*3+1)%17-8) + frac
		}
		got := make([]float64, n)
		want := make([]float64, n)
		Round(got, src)
		round64Go(want, src)
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("Round n=%d [%d] = %v, want %v (in %v)", n, i, got[i], want[i], src[i])
			}
		}
	}
}

func TestVectorPathActivations64(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, src []float64)
		gold func(dst, src []float64)
		cmp  func(a, b float64) bool
	}{
		{"ReLU", ReLU, relu64Go, vpClose64},
		{"Sigmoid", Sigmoid, sigmoid64Go, vpCloseActiv64},
		{"Tanh", Tanh, tanh64Go, vpCloseActiv64},
		{"Exp", Exp, exp64Go, vpCloseActiv64},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			src := vpActiv64(n, 4)
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, src)
			op.gold(want, src)
			vpCheck64(t, op.name, n, got, want, op.cmp)
		}
	}
}

func TestVectorPathScalar64(t *testing.T) {
	const s = 1.75
	ops := []struct {
		name string
		simd func(dst, a []float64)
		gold func(dst, a []float64)
	}{
		{"Scale", func(dst, a []float64) { Scale(dst, a, s) }, func(dst, a []float64) { scaleGo(dst, a, s) }},
		{"AddScalar", func(dst, a []float64) { AddScalar(dst, a, s) }, func(dst, a []float64) { addScalarGo(dst, a, s) }},
		{"SubFromScalar", func(dst, a []float64) { SubFromScalar(dst, a, s) }, func(dst, a []float64) { subFromScalarGo(dst, a, s) }},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpSeq64(n, 5)
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, a)
			op.gold(want, a)
			vpCheck64(t, op.name, n, got, want, vpClose64)
		}
	}
}

func TestVectorPathClamp64(t *testing.T) {
	const lo, hi = -3.0, 4.0
	for _, n := range vpLens {
		a := vpSeq64(n, 6)
		got := make([]float64, n)
		want := make([]float64, n)
		Clamp(got, a, lo, hi)
		clampGo(want, a, lo, hi)
		vpCheck64(t, "Clamp", n, got, want, vpClose64)
	}
}

func TestVectorPathClampScale64(t *testing.T) {
	const lo, hi, sc = -3.0, 4.0, 0.25
	for _, n := range vpLens {
		src := vpSeq64(n, 7)
		got := make([]float64, n)
		want := make([]float64, n)
		ClampScale(got, src, lo, hi, sc)
		clampScale64Go(want, src, lo, hi, sc)
		vpCheck64(t, "ClampScale", n, got, want, vpClose64)
	}
}

func TestVectorPathAddScaled64(t *testing.T) {
	const alpha = 0.6
	for _, n := range vpLens {
		s := vpSeq64(n, 8)
		got := vpSeq64(n, 9)
		want := append([]float64(nil), got...)
		AddScaled(got, alpha, s)
		addScaledGo64(want, alpha, s)
		vpCheck64(t, "AddScaled", n, got, want, vpClose64)
	}
}

func TestVectorPathAccumulateAdd64(t *testing.T) {
	for _, n := range vpLens {
		src := vpSeq64(n, 10)
		got := vpSeq64(n, 11)
		want := append([]float64(nil), got...)
		AccumulateAdd(got, src, 0)
		accumulateAdd64Go(want, src)
		vpCheck64(t, "AccumulateAdd", n, got, want, vpClose64)
	}
}

func TestVectorPathFMA64(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq64(n, 12)
		b := vpSeq64(n, 13)
		c := vpSeq64(n, 14)
		got := make([]float64, n)
		want := make([]float64, n)
		FMA(got, a, b, c)
		fmaGo(want, a, b, c)
		vpCheck64(t, "FMA", n, got, want, vpClose64)
	}
}

// TestVectorPathReductions64 covers scalar-returning reductions. For Min/Max the
// true extremum is planted in the final block so a kernel that fails to fold its
// later lanes into the accumulator cannot find it.
func TestVectorPathReductions64(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq64(n, 15)
		if got, want := Sum(a), sumGo(a); !vpClose64(got, want) {
			t.Errorf("Sum n=%d = %v, want %v (Go fallback)", n, got, want)
		}

		ext := vpSeq64(n, 16)
		ext[n-2] = -123.5 // global min, last block
		ext[n-1] = 234.5  // global max, last block
		if got, want := Min(ext), minGo(ext); got != want {
			t.Errorf("Min n=%d = %v, want %v (Go fallback)", n, got, want)
		}
		if got, want := Max(ext), maxGo(ext); got != want {
			t.Errorf("Max n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVectorPathDot64(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq64(n, 17)
		b := vpSeq64(n, 18)
		if got, want := DotProduct(a, b), dotProductGo(a, b); !vpClose64(got, want) {
			t.Errorf("DotProduct n=%d = %v, want %v (Go fallback)", n, got, want)
		}
		if got, want := EuclideanDistance(a, b), euclideanDistance64Go(a, b); !vpClose64(got, want) {
			t.Errorf("EuclideanDistance n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVectorPathVariance64(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq64(n, 19)
		var sum float64
		for _, v := range a {
			sum += v
		}
		mean := sum / float64(n)
		if got, want := variance64(a, mean), variance64Go(a, mean); !vpClose64(got, want) {
			t.Errorf("variance64 n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVectorPathInterleave2_64(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq64(n, 20)
		b := vpSeq64(n, 21)
		got := make([]float64, 2*n)
		want := make([]float64, 2*n)
		Interleave2(got, a, b)
		interleave2Go(want, a, b)
		vpCheck64(t, "Interleave2", 2*n, got, want, vpClose64)

		src := vpSeq64(2*n, 22)
		gotA := make([]float64, n)
		gotB := make([]float64, n)
		wantA := make([]float64, n)
		wantB := make([]float64, n)
		Deinterleave2(gotA, gotB, src)
		deinterleave2Go(wantA, wantB, src)
		vpCheck64(t, "Deinterleave2.a", n, gotA, wantA, vpClose64)
		vpCheck64(t, "Deinterleave2.b", n, gotB, wantB, vpClose64)
	}
}

// TestVectorPathInterleaveN_64 covers the N=3 (ST3/LD3) and N=4 (ST4/LD4) NEON
// structured-store kernels and the amd64 N=4 transpose against the generic Go path.
func TestVectorPathInterleaveN_64(t *testing.T) {
	for _, nc := range []int{3, 4} {
		for _, n := range vpLens {
			srcs := make([][]float64, nc)
			for c := range srcs {
				srcs[c] = vpSeq64(n, 30+c)
			}
			got := make([]float64, n*nc)
			want := make([]float64, n*nc)
			InterleaveN(got, srcs)
			interleaveNGo(want, srcs, n)
			vpCheck64(t, "InterleaveN", n*nc, got, want, vpClose64)

			src := vpSeq64(n*nc, 40)
			gotD := make([][]float64, nc)
			wantD := make([][]float64, nc)
			for c := range gotD {
				gotD[c] = make([]float64, n)
				wantD[c] = make([]float64, n)
			}
			DeinterleaveN(gotD, src)
			deinterleaveNGo(wantD, src, n)
			for c := range gotD {
				vpCheck64(t, "DeinterleaveN", n, gotD[c], wantD[c], vpClose64)
			}
		}
	}
}

func TestVectorPathCubicInterpDot64(t *testing.T) {
	const x = 0.375
	for _, n := range vpLens {
		hist := vpSeq64(n, 50)
		a := vpSeq64(n, 51)
		b := vpSeq64(n, 52)
		c := vpSeq64(n, 53)
		d := vpSeq64(n, 54)
		got := CubicInterpDot(hist, a, b, c, d, x)
		want := cubicInterpDotGo(hist, a, b, c, d, x)
		if !vpClose64(got, want) {
			t.Errorf("CubicInterpDot n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}
