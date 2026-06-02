package c64

import (
	"math"
	"testing"
)

// These tests exercise the SIMD-dispatched paths (NEON on arm64; AVX/AVX-512/SSE2
// on amd64) at and beyond the vector width, including non-multiples of the width
// so both the main assembly loop and the scalar remainder run. Every result is
// cross-checked against the pure-Go fallback, which is the trusted reference.
//
// The c64 package previously had no test that compared a SIMD-dispatched public
// function against its *Go fallback at len >= the vector width: the *Go tests
// exercise only the fallback, and the size-sweep tests do not cross-check. That
// is exactly the gap that let three broken NEON encodings ship in f16 (see issue
// #45): a kernel bug in the looping body stays invisible when the inputs are
// shorter than the SIMD width because only the Go path runs.
//
// complex64 is laid out as interleaved [re, im] float32 pairs. The widest kernel
// stride here is 8 elements (AVX-512), so the length set covers one, two and
// several blocks plus odd remainders.

// vpLens are lengths at or beyond every SIMD stride (NEON/AVX = 4, AVX-512 = 8),
// with non-multiples so the remainder path is always taken too.
var vpLens = []int{4, 5, 6, 7, 8, 9, 11, 16, 17, 23, 32, 33}

// vpClose32 is a relative+absolute tolerance comparison. The SIMD kernels reduce
// in a different order than the scalar Go reference (and Abs uses a float32 sqrt
// versus the fallback's float64 math.Hypot), so an exact match is not expected;
// the tolerance is still far tighter than any block-boundary or encoding bug.
func vpClose32(got, want float32) bool {
	d := math.Abs(float64(got - want))
	return d <= 1e-4*math.Abs(float64(want))+1e-5
}

func vpCClose(got, want complex64) bool {
	return vpClose32(real(got), real(want)) && vpClose32(imag(got), imag(want))
}

// vpMakeC64 builds a deterministic, non-trivial complex64 slice. Values vary per
// index and per seed so neighbouring lanes differ (a kernel that shuffles or
// drops a lane cannot accidentally match).
func vpMakeC64(n, seed int) []complex64 {
	s := make([]complex64, n)
	for i := range s {
		re := float32((i*7+seed)%23) - 11
		im := float32((i*5+seed*3)%17) - 8
		s[i] = complex(re, im)
	}
	return s
}

func vpMakeF32(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*3+seed)%19) - 9
	}
	return s
}

func TestVectorPathBinaryC64(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a, b []complex64)
		gold func(dst, a, b []complex64)
	}{
		{"Mul", Mul, mulGo},
		{"MulConj", MulConj, mulConjGo},
		{"Add", Add, addGo},
		{"Sub", Sub, subGo},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpMakeC64(n, 1)
			b := vpMakeC64(n, 2)
			got := make([]complex64, n)
			want := make([]complex64, n)
			op.simd(got, a, b)
			op.gold(want, a, b)
			for i := range got {
				if !vpCClose(got[i], want[i]) {
					t.Errorf("%s n=%d [%d] = %v, want %v (Go fallback)", op.name, n, i, got[i], want[i])
				}
			}
		}
	}
}

func TestVectorPathScaleC64(t *testing.T) {
	s := complex64(complex(1.5, -0.5))
	for _, n := range vpLens {
		a := vpMakeC64(n, 3)
		got := make([]complex64, n)
		want := make([]complex64, n)
		Scale(got, a, s)
		scaleGo(want, a, s)
		for i := range got {
			if !vpCClose(got[i], want[i]) {
				t.Errorf("Scale n=%d [%d] = %v, want %v (Go fallback)", n, i, got[i], want[i])
			}
		}
	}
}

func TestVectorPathConjC64(t *testing.T) {
	for _, n := range vpLens {
		a := vpMakeC64(n, 4)
		got := make([]complex64, n)
		want := make([]complex64, n)
		Conj(got, a)
		conjGo(want, a)
		for i := range got {
			if !vpCClose(got[i], want[i]) {
				t.Errorf("Conj n=%d [%d] = %v, want %v (Go fallback)", n, i, got[i], want[i])
			}
		}
	}
}

// TestVectorPathToRealC64 covers the complex -> real reductions, whose output is
// a float32 slice. AbsSq is exact arithmetic; Abs involves a square root, so both
// go through the same relative-tolerance check.
func TestVectorPathToRealC64(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst []float32, a []complex64)
		gold func(dst []float32, a []complex64)
	}{
		{"Abs", Abs, absGo},
		{"AbsSq", AbsSq, absSqGo},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpMakeC64(n, 5)
			got := make([]float32, n)
			want := make([]float32, n)
			op.simd(got, a)
			op.gold(want, a)
			for i := range got {
				if !vpClose32(got[i], want[i]) {
					t.Errorf("%s n=%d [%d] = %v, want %v (Go fallback, input %v)",
						op.name, n, i, got[i], want[i], a[i])
				}
			}
		}
	}
}

func TestVectorPathFromRealC64(t *testing.T) {
	for _, n := range vpLens {
		src := vpMakeF32(n, 6)
		got := make([]complex64, n)
		want := make([]complex64, n)
		FromReal(got, src)
		fromRealGo(want, src)
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("FromReal n=%d [%d] = %v, want %v (Go fallback)", n, i, got[i], want[i])
			}
		}
	}
}
