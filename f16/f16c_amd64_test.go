//go:build amd64

package f16

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// These tests call the F16C kernels directly (bypassing the dispatch thresholds)
// so the kernel itself is exercised regardless of length. f16IsNaN and
// fromFloat32Corpus are shared with the cross-platform parity tests in
// conv_correctness_test.go.

// TestToFloat32SliceF16C_Exhaustive converts every one of the 65536 possible
// Float16 values through the F16C kernel and checks bit-exact parity with the
// scalar reference, comparing NaN by class.
func TestToFloat32SliceF16C_Exhaustive(t *testing.T) {
	if !cpu.X86.F16C {
		t.Skip("F16C not available on this CPU")
	}
	const total = 1 << 16 // multiple of 8, so the kernel converts every element
	src := make([]Float16, total)
	for i := range src {
		src[i] = Float16(i)
	}
	got := make([]float32, total)
	toFloat32SliceF16C(got, src)

	for i := range src {
		want := toFloat32Go(src[i])
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("h=%#04x: got %v, want NaN", src[i], got[i])
			}
			continue
		}
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("h=%#04x: got bits %#08x, want %#08x", src[i], math.Float32bits(got[i]), math.Float32bits(want))
		}
	}
}

// TestFromFloat32SliceF16C_Direct feeds the boundary corpus plus a large random
// sweep through the F16C kernel directly and asserts bit-exact parity with the
// scalar reference for non-NaN inputs, class parity for NaN.
func TestFromFloat32SliceF16C_Direct(t *testing.T) {
	if !cpu.X86.F16C {
		t.Skip("F16C not available on this CPU")
	}

	src := fromFloat32Corpus()
	seed := uint32(0x9e3779b9)
	for range 4096 {
		seed = seed*1664525 + 1013904223
		src = append(src, math.Float32frombits(seed))
	}
	for len(src)%8 != 0 { // kernel requires a multiple of 8
		src = append(src, 0)
	}

	got := make([]Float16, len(src))
	fromFloat32SliceF16C(got, src)

	for i, f := range src {
		want := fromFloat32Go(f)
		if math.IsNaN(float64(f)) {
			if !f16IsNaN(got[i]) {
				t.Fatalf("src=%v (%#08x): got %#04x, want a NaN", f, math.Float32bits(f), got[i])
			}
			continue
		}
		if got[i] != want {
			t.Fatalf("src=%v (%#08x): got %#04x, want %#04x", f, math.Float32bits(f), got[i], want)
		}
	}
}
