//go:build amd64

package f32

import (
	"reflect"
	"testing"
)

// funcPC returns the entry program counter of a function value, for identity
// comparison of the selected kernel against the known interleave2 / Go symbols.
func funcPC(f any) uintptr { return reflect.ValueOf(f).Pointer() }

// TestInterleave2Kernels_NonAVXSelectsScalar is the regression test for the
// SIGILL reported in birdnet-go issue #3353. On an amd64 CPU without AVX (e.g.
// Celeron 887), the float32 resampler crashed because interleave2_32 /
// deinterleave2_32 dispatched straight to the VEX-encoded interleave2AVX /
// deinterleave2AVX kernels based on slice length alone. The kernels are now
// selected once at init via interleave2Kernels, exactly like every other f32
// SIMD op. This pins the invariant that got #3353 wrong: without AVX the scalar
// Go kernels must be chosen, never the AVX ones. It is host-independent (it does
// not depend on whether the test machine has AVX), so it fails on the buggy
// selection even on AVX-capable CI.
func TestInterleave2Kernels_NonAVXSelectsScalar(t *testing.T) {
	il, dl := interleave2Kernels(false)
	if funcPC(il) != funcPC(interleave2Go) {
		t.Errorf("interleave2Kernels(false) must select scalar interleave2Go; selecting the AVX kernel SIGILLs on non-AVX CPUs (birdnet-go #3353)")
	}
	if funcPC(dl) != funcPC(deinterleave2Go) {
		t.Errorf("interleave2Kernels(false) must select scalar deinterleave2Go; selecting the AVX kernel SIGILLs on non-AVX CPUs (birdnet-go #3353)")
	}
}

// TestInterleave2Kernels_AVXSelectsSIMD confirms the AVX kernels are still used
// when the CPU supports AVX, so the #3353 fix does not silently disable SIMD on
// capable hardware.
func TestInterleave2Kernels_AVXSelectsSIMD(t *testing.T) {
	il, dl := interleave2Kernels(true)
	if funcPC(il) != funcPC(interleave2AVX) {
		t.Errorf("interleave2Kernels(true) must select the AVX interleave2 kernel")
	}
	if funcPC(dl) != funcPC(deinterleave2AVX) {
		t.Errorf("interleave2Kernels(true) must select the AVX deinterleave2 kernel")
	}
}

// TestInterleave2_ScalarPathMatchesGo checks that the scalar fallback selected
// for AVX-less CPUs produces output identical to the AVX path for a length that
// would otherwise take the AVX kernel, so the #3353 fix does not change results.
func TestInterleave2_ScalarPathMatchesGo(t *testing.T) {
	const n = 4 * minAVXElements // well above the SIMD threshold
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i + 1)
		b[i] = float32(-(i + 1))
	}

	gotInter := make([]float32, 2*n)
	interleave2Go(gotInter, a, b)

	gotA := make([]float32, n)
	gotB := make([]float32, n)
	deinterleave2Go(gotA, gotB, gotInter)
	for i := range a {
		if gotA[i] != a[i] || gotB[i] != b[i] {
			t.Fatalf("scalar round-trip mismatch at %d: a=%v b=%v want a=%v b=%v", i, gotA[i], gotB[i], a[i], b[i])
		}
	}
}
