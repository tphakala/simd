//go:build amd64

package c128

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

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
