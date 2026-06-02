package c128

import (
	"math"
	"testing"
)

// These tests exercise the SIMD-dispatched paths (NEON on arm64; AVX-512/AVX/SSE2
// on amd64) at and beyond the vector width, including non-multiples of the width
// so both the main assembly loop and the scalar remainder run. Every result is
// cross-checked against the pure-Go fallback, which is the trusted reference.
//
// Before this file the only SIMD-vs-Go cross-check in the package was
// TestAbsSqKernels, which is amd64-only and calls the kernels directly, so the
// NEON path and the seven other public operations had no parity test at len >=
// the vector width. That is the gap described in issue #45: a looping-body or
// encoding bug stays invisible when the inputs are shorter than the SIMD width
// because only the Go fallback runs.
//
// complex128 is laid out as interleaved [re, im] float64 pairs. The widest kernel
// stride is 4 elements (AVX-512); the length set covers one, two and several
// blocks plus odd remainders.

// vpLens are lengths at or beyond every SIMD stride (NEON/AVX = 2, AVX-512 = 4),
// with non-multiples so the remainder path is always taken too.
var vpLens = []int{4, 5, 6, 7, 8, 9, 11, 16, 17, 23, 32, 33}

// vpClose64 is a relative+absolute tolerance comparison. The SIMD kernels reduce
// in a different order than the scalar Go reference (and Abs uses sqrt versus the
// fallback's math.Hypot), so the tolerance is set tighter than any block-boundary
// or encoding bug but loose enough for benign float64 reassociation.
func vpClose64(got, want float64) bool {
	d := math.Abs(got - want)
	return d <= 1e-12*math.Abs(want)+1e-12
}

func vpCClose(got, want complex128) bool {
	return vpClose64(real(got), real(want)) && vpClose64(imag(got), imag(want))
}

// vpMakeC128 builds a deterministic, non-trivial complex128 slice; values vary
// per index and per seed so a kernel that shuffles or drops a lane cannot match.
func vpMakeC128(n, seed int) []complex128 {
	s := make([]complex128, n)
	for i := range s {
		re := float64((i*7+seed)%23) - 11
		im := float64((i*5+seed*3)%17) - 8
		s[i] = complex(re, im)
	}
	return s
}

func TestVectorPathBinaryC128(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst, a, b []complex128)
		gold func(dst, a, b []complex128)
	}{
		{"Mul", Mul, mulGo},
		{"MulConj", MulConj, mulConjGo},
		{"Add", Add, addGo},
		{"Sub", Sub, subGo},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpMakeC128(n, 1)
			b := vpMakeC128(n, 2)
			got := make([]complex128, n)
			want := make([]complex128, n)
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

func TestVectorPathScaleC128(t *testing.T) {
	s := complex(1.5, -0.5)
	for _, n := range vpLens {
		a := vpMakeC128(n, 3)
		got := make([]complex128, n)
		want := make([]complex128, n)
		Scale(got, a, s)
		scaleGo(want, a, s)
		for i := range got {
			if !vpCClose(got[i], want[i]) {
				t.Errorf("Scale n=%d [%d] = %v, want %v (Go fallback)", n, i, got[i], want[i])
			}
		}
	}
}

func TestVectorPathConjC128(t *testing.T) {
	for _, n := range vpLens {
		a := vpMakeC128(n, 4)
		got := make([]complex128, n)
		want := make([]complex128, n)
		Conj(got, a)
		conjGo(want, a)
		for i := range got {
			if !vpCClose(got[i], want[i]) {
				t.Errorf("Conj n=%d [%d] = %v, want %v (Go fallback)", n, i, got[i], want[i])
			}
		}
	}
}

// TestVectorPathToRealC128 covers the complex -> real reductions, whose output is
// a float64 slice. AbsSq is exact arithmetic; Abs involves a square root, so both
// go through the same relative-tolerance check.
func TestVectorPathToRealC128(t *testing.T) {
	ops := []struct {
		name string
		simd func(dst []float64, a []complex128)
		gold func(dst []float64, a []complex128)
	}{
		{"Abs", Abs, absGo},
		{"AbsSq", AbsSq, absSqGo},
	}
	for _, op := range ops {
		for _, n := range vpLens {
			a := vpMakeC128(n, 5)
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, a)
			op.gold(want, a)
			for i := range got {
				if !vpClose64(got[i], want[i]) {
					t.Errorf("%s n=%d [%d] = %v, want %v (Go fallback, input %v)",
						op.name, n, i, got[i], want[i], a[i])
				}
			}
		}
	}
}
