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
	if !cpu.X86.AVX512F || !cpu.X86.AVX512VL || !cpu.X86.AVX512DQ {
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

func TestInitAVXNoFMA(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not supported on this CPU")
	}

	savedDotProduct := dotProductImpl
	savedMinSIMD := minSIMDElements
	savedAdd := addImpl
	t.Cleanup(func() {
		dotProductImpl = savedDotProduct
		minSIMDElements = savedMinSIMD
		addImpl = savedAdd
	})

	initAVXNoFMA()

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}
	dst := make([]float64, len(a))

	// dotProduct should remain functional via SSE2 fallback in AVX-without-FMA mode
	gotDot := dotProductImpl(a, b)
	if gotDot != 20 {
		t.Errorf("After initAVXNoFMA, dotProduct = %v, want 20", gotDot)
	}

	// Non-FMA arithmetic paths should still use AVX implementations
	addImpl(dst, a, b)
	for i, want := range []float64{5, 5, 5, 5} {
		if dst[i] != want {
			t.Errorf("After initAVXNoFMA, add[%d] = %v, want %v", i, dst[i], want)
		}
	}

	if minSIMDElements != minAVXElements {
		t.Errorf("initAVXNoFMA didn't set minSIMDElements correctly")
	}
}
