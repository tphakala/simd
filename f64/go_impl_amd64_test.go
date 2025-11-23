//go:build amd64

package f64

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Tests for init functions to ensure they properly configure function pointers
// These tests are AMD64-specific because they test x86 SIMD initialization paths.

func TestInitGo(t *testing.T) {
	// Save current state
	savedDotProduct := dotProductImpl

	// Call initGo
	initGo()

	// Test that operations work with Go implementations
	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20) // 1*4 + 2*3 + 3*2 + 4*1 = 20
	if got != want {
		t.Errorf("After initGo, dotProduct = %v, want %v", got, want)
	}

	// Restore
	dotProductImpl = savedDotProduct
}

func TestInitSSE2(t *testing.T) {
	savedDotProduct := dotProductImpl

	initSSE2()

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20)
	if got != want {
		t.Errorf("After initSSE2, dotProduct = %v, want %v", got, want)
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

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20)
	if got != want {
		t.Errorf("After initAVX512, dotProduct = %v, want %v", got, want)
	}

	// Verify minSIMDElements was set
	if minSIMDElements != minAVX512Elements {
		t.Errorf("initAVX512 didn't set minSIMDElements correctly")
	}

	dotProductImpl = savedDotProduct
	minSIMDElements = savedMinSIMD
}
