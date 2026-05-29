package f16

import (
	"math"
	"testing"
)

// These tests exercise the NEON-dispatched paths for lengths at or beyond the
// 8-lane vector width, including non-multiples of 8 so the scalar/Go remainder
// also runs. They guard against encoding bugs that only surface when the
// assembly loop actually executes: the earlier reciprocal/min/max tests used
// fewer than 16 elements, so the looping NEON code never ran and three broken
// WORD encodings shipped undetected. Each result is cross-checked against the
// pure-Go fallback, which is the trusted reference.

func relClose(got, want float32) bool {
	d := math.Abs(float64(got - want))
	return d <= 0.02*math.Abs(float64(want))+0.002
}

func TestReciprocalVectorPath(t *testing.T) {
	for _, n := range []int{8, 11, 16, 19, 24} {
		a := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(float32(i + 1)) // 1..n, never zero
		}
		gotDst := make([]Float16, n)
		wantDst := make([]Float16, n)
		Reciprocal(gotDst, a)
		reciprocalGo(wantDst, a)
		for i := range gotDst {
			got, want := ToFloat32(gotDst[i]), ToFloat32(wantDst[i])
			if !relClose(got, want) {
				t.Errorf("n=%d Reciprocal[%d] = %v, want %v (Go fallback, input %v)",
					n, i, got, want, ToFloat32(a[i]))
			}
		}
	}
}

func TestMinMaxVectorPath(t *testing.T) {
	for _, n := range []int{16, 19, 24, 32} {
		a := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(float32((i*7+3)%40) - 5)
		}
		// Put the true extremum past the first 8-lane block so a loop that
		// fails to update its accumulator cannot find it.
		a[n-4] = FromFloat32(-100)
		a[n-3] = FromFloat32(100)

		if got, want := ToFloat32(Min(a)), ToFloat32(minGo(a)); !almostEqual32(got, want, 0.01) {
			t.Errorf("n=%d Min() = %v, want %v (Go fallback)", n, got, want)
		}
		if got, want := ToFloat32(Max(a)), ToFloat32(maxGo(a)); !almostEqual32(got, want, 0.01) {
			t.Errorf("n=%d Max() = %v, want %v (Go fallback)", n, got, want)
		}
	}
}
