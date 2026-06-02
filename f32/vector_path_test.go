package f32

import (
	"math"
	"testing"
)

// These tests exercise the SIMD-dispatched paths (NEON on arm64; SSE/AVX/AVX-512
// on amd64) at and beyond the vector width, including non-multiples of the width
// so both the main assembly loop and the scalar remainder run. Every result is
// cross-checked against the pure-Go fallback, which is the trusted reference and
// the code that actually runs on architectures without SIMD.
//
// This guards the gap described in issue #45: a kernel bug in the looping body
// stays invisible when a test uses fewer elements than the SIMD width, because
// the dispatch falls through to the Go fallback and silently produces the right
// answer (that is how three broken f16 NEON encodings shipped). On amd64 several
// elementwise ops only take the AVX path at len >= 8, so a single len-6 or len-7
// fixture never executes the vectorized code at all.
//
// Ops that already have dedicated SIMD-vs-Go parity sweeps are not duplicated
// here: the split-format complex ops (TestMulComplex_SIMDvsGo and friends),
// Reverse (TestReverse_GoVsSIMD) and AddSub (TestAddSub_GoVsSIMD). The core
// arithmetic and reductions are additionally pinned by the C-reference tests
// (f32_cref_test.go / reference_test.go); this file adds the missing direct
// Go-fallback parity for them and for the activation, scale, clamp, conversion,
// interleave and resampling kernels.

// vpLens spans every SIMD stride: NEON (4), AVX (8) and AVX-512 (16), with
// non-multiples so the remainder path is always taken too. Lengths below an
// architecture's threshold simply compare the Go path against itself, which is
// harmless; the larger lengths are what exercise the assembly.
var vpLens = []int{4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}

func vpClose32(got, want float32) bool {
	d := math.Abs(float64(got - want))
	return d <= 1e-4*math.Abs(float64(want))+1e-5
}

// vpCloseActiv32 is the looser tolerance used for the activation kernels, whose
// SIMD paths are polynomial approximations while the Go fallback calls the
// accurate math.Exp/math.Tanh. It is still far tighter than any block-boundary
// or encoding bug (those leave whole lanes at zero or garbage).
func vpCloseActiv32(got, want float32) bool {
	d := math.Abs(float64(got - want))
	return d <= 1e-3*math.Abs(float64(want))+1e-4
}

func vpSeq32(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*7+seed)%23) - 11
	}
	return s
}

// vpPos32 returns strictly positive values, for divisors and for Sqrt/Reciprocal.
func vpPos32(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*5+seed)%13) + 0.5
	}
	return s
}

// vpActiv32 returns moderate values in roughly [-4, 4] for the activation kernels.
func vpActiv32(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(-4.0 + 0.37*float64((i*3+seed)%23))
	}
	return s
}

func vpCheck32(t *testing.T, name string, n int, got, want []float32, cmp func(a, b float32) bool) {
	t.Helper()
	for i := range got {
		if !cmp(got[i], want[i]) {
			t.Errorf("%s n=%d [%d] = %v, want %v (Go fallback)", name, n, i, got[i], want[i])
		}
	}
}

func TestVectorPathBinary32(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a, b []float32)
		gold func(dst, a, b []float32)
		posB bool // divisor must be non-zero
	}{
		{"Add", Add, addGo, false},
		{"Sub", Sub, subGo, false},
		{"Mul", Mul, mulGo, false},
		{"Div", Div, divGo, true},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpSeq32(n, 1)
			var b []float32
			if op.posB {
				b = vpPos32(n, 2)
			} else {
				b = vpSeq32(n, 2)
			}
			got := make([]float32, n)
			want := make([]float32, n)
			op.simd(got, a, b)
			op.gold(want, a, b)
			vpCheck32(t, op.name, n, got, want, vpClose32)
		}
	}
}

func TestVectorPathUnary32(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a []float32)
		gold func(dst, a []float32)
		pos  bool
	}{
		{"Abs", Abs, absGo, false},
		{"Neg", Neg, negGo, false},
		{"Sqrt", Sqrt, sqrt32Go, true},
		{"Reciprocal", Reciprocal, reciprocal32Go, true},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			var a []float32
			if op.pos {
				a = vpPos32(n, 3)
			} else {
				a = vpSeq32(n, 3)
			}
			got := make([]float32, n)
			want := make([]float32, n)
			op.simd(got, a)
			op.gold(want, a)
			vpCheck32(t, op.name, n, got, want, vpClose32)
		}
	}
}

// TestVectorPathActivations32 covers the (dst, src) activation kernels. ReLU is
// exact; Sigmoid/Tanh/Exp use the looser approximation tolerance.
func TestVectorPathActivations32(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, src []float32)
		gold func(dst, src []float32)
		cmp  func(a, b float32) bool
	}{
		{"ReLU", ReLU, relu32Go, vpClose32},
		{"Sigmoid", Sigmoid, sigmoid32Go, vpCloseActiv32},
		{"Tanh", Tanh, tanh32Go, vpCloseActiv32},
		{"Exp", Exp, exp32Go, vpCloseActiv32},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			src := vpActiv32(n, 4)
			got := make([]float32, n)
			want := make([]float32, n)
			op.simd(got, src)
			op.gold(want, src)
			vpCheck32(t, op.name, n, got, want, op.cmp)
		}
	}
}

func TestVectorPathScalar32(t *testing.T) {
	const s = float32(1.75)
	ops := []struct {
		name string
		simd func(dst, a []float32)
		gold func(dst, a []float32)
	}{
		{"Scale", func(dst, a []float32) { Scale(dst, a, s) }, func(dst, a []float32) { scaleGo(dst, a, s) }},
		{"AddScalar", func(dst, a []float32) { AddScalar(dst, a, s) }, func(dst, a []float32) { addScalarGo(dst, a, s) }},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpSeq32(n, 5)
			got := make([]float32, n)
			want := make([]float32, n)
			op.simd(got, a)
			op.gold(want, a)
			vpCheck32(t, op.name, n, got, want, vpClose32)
		}
	}
}

func TestVectorPathClamp32(t *testing.T) {
	const lo, hi = float32(-3), float32(4)
	for _, n := range vpLens {
		a := vpSeq32(n, 6)
		got := make([]float32, n)
		want := make([]float32, n)
		Clamp(got, a, lo, hi)
		clampGo(want, a, lo, hi)
		vpCheck32(t, "Clamp", n, got, want, vpClose32)
	}
}

func TestVectorPathClampScale32(t *testing.T) {
	const lo, hi, sc = float32(-3), float32(4), float32(0.25)
	for _, n := range vpLens {
		src := vpSeq32(n, 7)
		got := make([]float32, n)
		want := make([]float32, n)
		ClampScale(got, src, lo, hi, sc)
		clampScale32Go(want, src, lo, hi, sc)
		vpCheck32(t, "ClampScale", n, got, want, vpClose32)
	}
}

func TestVectorPathAddScaled32(t *testing.T) {
	const alpha = float32(0.6)
	for _, n := range vpLens {
		s := vpSeq32(n, 8)
		got := vpSeq32(n, 9)
		want := append([]float32(nil), got...)
		AddScaled(got, alpha, s)
		addScaledGo(want, alpha, s)
		vpCheck32(t, "AddScaled", n, got, want, vpClose32)
	}
}

func TestVectorPathAccumulateAdd32(t *testing.T) {
	for _, n := range vpLens {
		src := vpSeq32(n, 10)
		got := vpSeq32(n, 11)
		want := append([]float32(nil), got...)
		AccumulateAdd(got, src, 0)
		accumulateAdd32Go(want, src)
		vpCheck32(t, "AccumulateAdd", n, got, want, vpClose32)
	}
}

func TestVectorPathFMA32(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq32(n, 12)
		b := vpSeq32(n, 13)
		c := vpSeq32(n, 14)
		got := make([]float32, n)
		want := make([]float32, n)
		FMA(got, a, b, c)
		fmaGo(want, a, b, c)
		vpCheck32(t, "FMA", n, got, want, vpClose32)
	}
}

// TestVectorPathReductions32 covers scalar-returning reductions. For Min/Max the
// true extremum is planted in the final block so a kernel that fails to fold its
// later lanes into the accumulator cannot find it.
func TestVectorPathReductions32(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq32(n, 15)
		if got, want := Sum(a), sumGo(a); !vpClose32(got, want) {
			t.Errorf("Sum n=%d = %v, want %v (Go fallback)", n, got, want)
		}

		ext := vpSeq32(n, 16)
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

func TestVectorPathDot32(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq32(n, 17)
		b := vpSeq32(n, 18)
		if got, want := DotProduct(a, b), dotProductGo(a, b); !vpClose32(got, want) {
			t.Errorf("DotProduct n=%d = %v, want %v (Go fallback)", n, got, want)
		}
		if got, want := EuclideanDistance(a, b), euclideanDistance32Go(a, b); !vpClose32(got, want) {
			t.Errorf("EuclideanDistance n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVectorPathVariance32(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq32(n, 19)
		// Use one shared mean (computed in float64 for accuracy) so the SIMD and
		// Go variance kernels are compared on identical inputs.
		var sum float64
		for _, v := range a {
			sum += float64(v)
		}
		mean := float32(sum / float64(n))
		if got, want := variance32(a, mean), variance32Go(a, mean); !vpClose32(got, want) {
			t.Errorf("variance32 n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVectorPathInterleave2_32(t *testing.T) {
	for _, n := range vpLens {
		a := vpSeq32(n, 20)
		b := vpSeq32(n, 21)
		got := make([]float32, 2*n)
		want := make([]float32, 2*n)
		Interleave2(got, a, b)
		interleave2Go(want, a, b)
		vpCheck32(t, "Interleave2", 2*n, got, want, vpClose32)

		src := vpSeq32(2*n, 22)
		gotA := make([]float32, n)
		gotB := make([]float32, n)
		wantA := make([]float32, n)
		wantB := make([]float32, n)
		Deinterleave2(gotA, gotB, src)
		deinterleave2Go(wantA, wantB, src)
		vpCheck32(t, "Deinterleave2.a", n, gotA, wantA, vpClose32)
		vpCheck32(t, "Deinterleave2.b", n, gotB, wantB, vpClose32)
	}
}

// TestVectorPathInterleaveN_32 covers the N=3 (ST3/LD3) and N=4 (ST4/LD4) NEON
// structured-store kernels and the amd64 N=4 transpose against the generic Go path.
func TestVectorPathInterleaveN_32(t *testing.T) {
	for _, nc := range []int{3, 4} {
		for _, n := range vpLens {
			srcs := make([][]float32, nc)
			for c := range srcs {
				srcs[c] = vpSeq32(n, 30+c)
			}
			got := make([]float32, n*nc)
			want := make([]float32, n*nc)
			InterleaveN(got, srcs)
			interleaveNGo(want, srcs, n)
			vpCheck32(t, "InterleaveN", n*nc, got, want, vpClose32)

			src := vpSeq32(n*nc, 40)
			gotD := make([][]float32, nc)
			wantD := make([][]float32, nc)
			for c := range gotD {
				gotD[c] = make([]float32, n)
				wantD[c] = make([]float32, n)
			}
			DeinterleaveN(gotD, src)
			deinterleaveNGo(wantD, src, n)
			for c := range gotD {
				vpCheck32(t, "DeinterleaveN", n, gotD[c], wantD[c], vpClose32)
			}
		}
	}
}

func TestVectorPathCubicInterpDot32(t *testing.T) {
	const x = float32(0.375)
	for _, n := range vpLens {
		hist := vpSeq32(n, 50)
		a := vpSeq32(n, 51)
		b := vpSeq32(n, 52)
		c := vpSeq32(n, 53)
		d := vpSeq32(n, 54)
		got := CubicInterpDot(hist, a, b, c, d, x)
		want := cubicInterpDotGo(hist, a, b, c, d, x)
		if !vpClose32(got, want) {
			t.Errorf("CubicInterpDot n=%d = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

// TestVectorPathConvert32 covers the int<->float conversion kernels. Their Go
// fallbacks are documented to be bit-identical to the SIMD paths, so an exact
// match is required.
func TestVectorPathConvert32(t *testing.T) {
	for _, n := range vpLens {
		// int32 -> float32
		s32 := make([]int32, n)
		for i := range s32 {
			s32[i] = int32((i*131+7)%2000 - 1000)
		}
		g32 := make([]float32, n)
		w32 := make([]float32, n)
		Int32ToFloat32Scale(g32, s32, 0.5)
		int32ToFloat32ScaleGo(w32, s32, 0.5)
		for i := range g32 {
			if g32[i] != w32[i] {
				t.Errorf("Int32ToFloat32Scale n=%d [%d] = %v, want %v", n, i, g32[i], w32[i])
			}
		}

		// int16 -> float32
		s16 := make([]int16, n)
		for i := range s16 {
			s16[i] = int16((i*97+3)%2000 - 1000)
		}
		gi := make([]float32, n)
		wi := make([]float32, n)
		Int16ToFloat32Scale(gi, s16, 0.25)
		int16ToFloat32ScaleGo(wi, s16, 0.25)
		for i := range gi {
			if gi[i] != wi[i] {
				t.Errorf("Int16ToFloat32Scale n=%d [%d] = %v, want %v", n, i, gi[i], wi[i])
			}
		}

		// float32 -> int16 (includes saturation at both ends)
		sf := make([]float32, n)
		for i := range sf {
			sf[i] = float32((i*1103+17)%80000-40000) + 0.3
		}
		go16 := make([]int16, n)
		wo16 := make([]int16, n)
		Float32ToInt16Scale(go16, sf, 1.0)
		float32ToInt16ScaleGo(wo16, sf, 1.0)
		for i := range go16 {
			if go16[i] != wo16[i] {
				t.Errorf("Float32ToInt16Scale n=%d [%d] = %v, want %v (in %v)", n, i, go16[i], wo16[i], sf[i])
			}
		}
	}
}
