//go:build arm64

package f16

import (
	"math"
	"math/rand"
	"testing"
)

// TestDotProductF32_NEONOnlyGate verifies that DotProductF32 still dispatches to
// the wide-NEON kernel (dotProductWideNEON) and matches the Go reference when
// FEAT_FP16 is reported absent. This simulates a base ARMv8.0-A NEON core
// without the FP16 arithmetic extension (Cortex-A53/A72, Raspberry Pi 3/4).
//
// The FP32-widened kernel only uses FCVTL/FCVTL2 (base ARMv8.0 converts) and
// FP32 .4S arithmetic, so it must be gated on hasNEON, not hasFP16 (issue #48).
// The test forces hasFP16=false with hasNEON=true; it must not call
// t.Parallel(), since it mutates the package-level dispatch flags.
func TestDotProductF32_NEONOnlyGate(t *testing.T) {
	if !hasNEON {
		t.Skip("requires NEON")
	}
	origFP16, origNEON := hasFP16, hasNEON
	hasFP16, hasNEON = false, true
	defer func() { hasFP16, hasNEON = origFP16, origNEON }()

	// Mix of exact multiples of neonWidth and non-multiples to also cover the
	// Go remainder tail under the forced NEON-only gate.
	sizes := []int{8, 11, 16, 64, 100, 256, 1024}
	rng := rand.New(rand.NewSource(42))
	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(rng.Float32()*2 - 1) // [-1, 1]
			b[i] = FromFloat32(rng.Float32()*2 - 1)
		}

		got := DotProductF32(a, b)
		want := dotProductGo(a, b)

		// Reference and SIMD differ in summation order; bound generously to
		// absorb up to n*ULP per accumulator chain.
		tol := math.Abs(float64(want))*float64(n)*1e-6 + 1e-5
		if math.Abs(float64(got-want)) > tol {
			t.Errorf("n=%d: got %v, want %v (tol %v)", n, got, want, tol)
		}
	}
}
