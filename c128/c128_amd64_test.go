//go:build amd64

package c128

import (
	"math"
	"reflect"
	"runtime"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// implName returns the symbol name of a dispatch function pointer, for use in
// assertion failure messages.
func implName(f any) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

// sameImpl reports whether two dispatch function pointers refer to the same
// function. Go does not allow func == func, so compare program counters.
func sameImpl(a, b any) bool {
	return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer()
}

// Tests for init functions to ensure they properly configure function pointers
// These tests are AMD64-specific because they test x86 SIMD initialization paths.

func TestInitGo(t *testing.T) {
	savedMul := mulImpl

	initGo()

	a := []complex128{1 + 2i}
	b := []complex128{3 + 4i}
	dst := make([]complex128, 1)

	mulImpl(dst, a, b)
	want := a[0] * b[0]
	if !complexClose(dst[0], want) {
		t.Errorf("After initGo, mul = %v, want %v", dst[0], want)
	}

	mulImpl = savedMul
}

func TestInitSSE2(t *testing.T) {
	savedMul := mulImpl

	initSSE2()

	a := []complex128{1 + 2i}
	b := []complex128{3 + 4i}
	dst := make([]complex128, 1)

	mulImpl(dst, a, b)
	want := a[0] * b[0]
	if !complexClose(dst[0], want) {
		t.Errorf("After initSSE2, mul = %v, want %v", dst[0], want)
	}

	mulImpl = savedMul
}

func TestInitAVX512(t *testing.T) {
	if !cpu.X86.AVX512F || !cpu.X86.AVX512VL {
		t.Skip("AVX-512 not supported on this CPU")
	}

	savedMul := mulImpl

	initAVX512()

	a := []complex128{1 + 2i}
	b := []complex128{3 + 4i}
	dst := make([]complex128, 1)

	mulImpl(dst, a, b)
	want := a[0] * b[0]
	if !complexClose(dst[0], want) {
		t.Errorf("After initAVX512, mul = %v, want %v", dst[0], want)
	}

	mulImpl = savedMul
}

// c128Impls snapshots the dispatch function pointers so a test that mutates
// them via an init*/selectImpl call can restore them for later tests.
type c128Impls struct {
	mul, mulConj, add, sub binaryOpFunc
	scale                  scaleFunc
	abs, absSq             unaryAbsFunc
	conj                   unaryConjFunc
}

func saveImpls() c128Impls {
	return c128Impls{mulImpl, mulConjImpl, addImpl, subImpl, scaleImpl, absImpl, absSqImpl, conjImpl}
}

func restoreImpls(s c128Impls) {
	mulImpl, mulConjImpl, addImpl, subImpl = s.mul, s.mulConj, s.add, s.sub
	scaleImpl = s.scale
	absImpl, absSqImpl = s.abs, s.absSq
	conjImpl = s.conj
}

// TestSelectImpl exercises every dispatch arm of selectImpl on any host,
// including the AVX-without-FMA arm that never runs through init() on
// FMA-capable CI hardware. It only assigns function pointers (no kernel is
// executed), so it is safe regardless of the host's actual CPU features.
func TestSelectImpl(t *testing.T) {
	saved := saveImpls()
	defer restoreImpls(saved)

	cases := []struct {
		name                      string
		avx512, avxFMA, avx, sse2 bool
	}{
		{"avx512", true, false, false, false},
		{"avxFMA", false, true, false, false},
		{"avxNoFMA", false, false, true, false},
		{"sse2", false, false, false, true},
		{"go", false, false, false, false},
	}
	for _, c := range cases {
		selectImpl(c.avx512, c.avxFMA, c.avx, c.sse2)
		if mulImpl == nil || absSqImpl == nil || conjImpl == nil {
			t.Fatalf("%s: selectImpl left a nil implementation pointer", c.name)
		}
	}
}

// TestInitAVXNoFMA verifies the AVX-without-FMA dispatch path: FMA-dependent
// kernels (mul/mulConj/scale) fall back to SSE2, while FMA-free AVX kernels
// (add/sub/abs/absSq/conj) stay on AVX. Runs on any AVX-capable CPU by calling
// initAVXNoFMA directly, even when the CPU also has FMA.
func TestInitAVXNoFMA(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not supported on this CPU")
	}

	// Restore every pointer initAVXNoFMA reassigns so later tests are unaffected.
	saved := saveImpls()
	defer restoreImpls(saved)

	initAVXNoFMA()

	// Assert the dispatch targets, not just numeric results: a wrong impl
	// (e.g. mulAVX or absSqSSE2) could still produce correct numbers and hide a
	// regression. FMA-dependent kernels must fall back to SSE2; FMA-free kernels
	// must stay on AVX.
	targets := []struct {
		name      string
		got, want any
	}{
		{"mulImpl", mulImpl, mulSSE2},
		{"mulConjImpl", mulConjImpl, mulConjSSE2},
		{"scaleImpl", scaleImpl, scaleSSE2},
		{"addImpl", addImpl, addAVX},
		{"subImpl", subImpl, subAVX},
		{"absImpl", absImpl, absAVX},
		{"absSqImpl", absSqImpl, absSqAVX},
		{"conjImpl", conjImpl, conjAVX},
	}
	for _, tt := range targets {
		if !sameImpl(tt.got, tt.want) {
			t.Errorf("%s dispatch = %s, want %s", tt.name, implName(tt.got), implName(tt.want))
		}
	}

	// SSE2-routed kernel (FMA-dependent): mul.
	a := []complex128{1 + 2i, 5 + 6i}
	b := []complex128{3 + 4i, 7 + 8i}
	mdst := make([]complex128, 2)
	mulImpl(mdst, a, b)
	for i := range mdst {
		if want := a[i] * b[i]; !complexClose(mdst[i], want) {
			t.Errorf("After initAVXNoFMA, mul[%d] = %v, want %v", i, mdst[i], want)
		}
	}

	// AVX-routed kernel (FMA-free): absSq.
	sdst := make([]float64, len(a))
	absSqImpl(sdst, a)
	for i := range sdst {
		r, im := real(a[i]), imag(a[i])
		if want := r*r + im*im; math.Abs(sdst[i]-want) > epsilon {
			t.Errorf("After initAVXNoFMA, absSq[%d] = %v, want %v", i, sdst[i], want)
		}
	}
}

// TestAbsSqKernels compares each available AbsSq assembly kernel against the Go
// reference across sizes that exercise the wide loop, narrow loop, and scalar
// remainder paths. Guards #23 (loop widening) against tail-handling bugs.
func TestAbsSqKernels(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17}

	kernels := []struct {
		name string
		fn   unaryAbsFunc
		skip bool
	}{
		{"SSE2", absSqSSE2, !cpu.X86.SSE2},
		{"AVX", absSqAVX, !cpu.X86.AVX},
		{"AVX512", absSqAVX512, !cpu.X86.AVX512F || !cpu.X86.AVX512VL},
	}

	for _, k := range kernels {
		if k.skip {
			continue
		}
		t.Run(k.name, func(t *testing.T) {
			for _, n := range sizes {
				a := make([]complex128, n)
				for i := range a {
					a[i] = complex(float64(i+1), float64(2*i+3))
				}
				got := make([]float64, n)
				want := make([]float64, n)
				k.fn(got, a)
				absSqGo(want, a)
				for i := range got {
					if math.Abs(got[i]-want[i]) > epsilon {
						t.Errorf("%s n=%d: AbsSq[%d] = %v, want %v", k.name, n, i, got[i], want[i])
					}
				}
			}
		})
	}
}
