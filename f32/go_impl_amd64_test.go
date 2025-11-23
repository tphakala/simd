//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Tests for init functions to ensure they properly configure function pointers
// These tests are AMD64-specific because they test x86 SIMD initialization paths.

func TestInitGo(t *testing.T) {
	savedDotProduct := dotProductImpl

	initGo()

	a := []float32{1, 2, 3, 4}
	b := []float32{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float32(20)
	if got != want {
		t.Errorf("After initGo, dotProduct = %v, want %v", got, want)
	}

	dotProductImpl = savedDotProduct
}

func TestInitSSE(t *testing.T) {
	savedDotProduct := dotProductImpl

	initSSE()

	a := []float32{1, 2, 3, 4}
	b := []float32{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float32(20)
	if got != want {
		t.Errorf("After initSSE, dotProduct = %v, want %v", got, want)
	}

	dotProductImpl = savedDotProduct
}

func TestInitAVX512(t *testing.T) {
	if !cpu.X86.AVX512F || !cpu.X86.AVX512VL {
		t.Skip("AVX-512 not supported on this CPU")
	}

	savedDotProduct := dotProductImpl
	savedMinSIMD := minSIMDElements

	initAVX512()

	a := []float32{1, 2, 3, 4}
	b := []float32{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float32(20)
	if got != want {
		t.Errorf("After initAVX512, dotProduct = %v, want %v", got, want)
	}

	if minSIMDElements != minAVX512Elements {
		t.Errorf("initAVX512 didn't set minSIMDElements correctly")
	}

	dotProductImpl = savedDotProduct
	minSIMDElements = savedMinSIMD
}
