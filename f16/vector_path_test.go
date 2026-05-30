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

func TestEuclideanDistanceUnequalLen(t *testing.T) {
	// euclideanDistanceGo must not panic when len(a) > len(b); it computes over
	// the common prefix, consistent with the public API which slices to min.
	a := []Float16{FromFloat32(3), FromFloat32(4), FromFloat32(99)}
	b := []Float16{FromFloat32(0), FromFloat32(0)}
	if got := euclideanDistanceGo(a, b); !almostEqual32(got, 5, 0.1) {
		t.Errorf("euclideanDistanceGo(unequal len) = %v, want 5", got)
	}
}

func TestEuclideanDistanceVectorPath(t *testing.T) {
	for _, n := range []int{8, 11, 16, 19, 24, 32} {
		a := make([]Float16, n)
		b := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(float32((i*3+1)%17) - 8)
			b[i] = FromFloat32(float32((i*5+2)%13) - 6)
		}
		got := EuclideanDistance(a, b)
		want := euclideanDistanceGo(a, b)
		if !relClose(got, want) {
			t.Errorf("n=%d EuclideanDistance() = %v, want %v (Go fallback)", n, got, want)
		}
	}
}

func TestVarianceStdDevVectorPath(t *testing.T) {
	for _, n := range []int{8, 11, 16, 19, 24, 32} {
		a := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(float32((i*7+3)%50) - 20)
		}
		gotV := Variance(a)
		wantV := varianceGo(a, Mean(a))
		if !relClose(gotV, wantV) {
			t.Errorf("n=%d Variance() = %v, want %v (Go fallback)", n, gotV, wantV)
		}
		gotS := StdDev(a)
		wantS := float32(math.Sqrt(float64(wantV)))
		if !relClose(gotS, wantS) {
			t.Errorf("n=%d StdDev() = %v, want %v (Go fallback)", n, gotS, wantS)
		}
	}
}

func TestInterleave2VectorPath(t *testing.T) {
	for _, n := range []int{8, 11, 16, 19, 24} {
		a := make([]Float16, n)
		b := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(float32(i + 1))
			b[i] = FromFloat32(float32(100 + i))
		}
		got := make([]Float16, 2*n)
		want := make([]Float16, 2*n)
		Interleave2(got, a, b)
		interleave2Go(want, a, b)
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("n=%d Interleave2[%d] = 0x%04X, want 0x%04X", n, i, got[i], want[i])
			}
		}
	}
}

func TestDeinterleave2VectorPath(t *testing.T) {
	for _, n := range []int{8, 11, 16, 19, 24} {
		src := make([]Float16, 2*n)
		for i := range src {
			src[i] = FromFloat32(float32(i + 1))
		}
		gotA := make([]Float16, n)
		gotB := make([]Float16, n)
		wantA := make([]Float16, n)
		wantB := make([]Float16, n)
		Deinterleave2(gotA, gotB, src)
		deinterleave2Go(wantA, wantB, src)
		for i := range gotA {
			if gotA[i] != wantA[i] || gotB[i] != wantB[i] {
				t.Errorf("n=%d Deinterleave2[%d] = (0x%04X,0x%04X), want (0x%04X,0x%04X)",
					n, i, gotA[i], gotB[i], wantA[i], wantB[i])
			}
		}
	}
}

func TestClampScaleVectorPath(t *testing.T) {
	minV := FromFloat32(-2)
	maxV := FromFloat32(5)
	sc := FromFloat32(0.25)
	for _, n := range []int{8, 11, 16, 19, 24} {
		src := make([]Float16, n)
		for i := range src {
			src[i] = FromFloat32(float32(i%20) - 8) // below min, in range, above max
		}
		got := make([]Float16, n)
		want := make([]Float16, n)
		ClampScale(got, src, minV, maxV, sc)
		clampScaleGo(want, src, minV, maxV, sc)
		for i := range got {
			if !relClose(ToFloat32(got[i]), ToFloat32(want[i])) {
				t.Errorf("n=%d ClampScale[%d] = %v, want %v (Go fallback)",
					n, i, ToFloat32(got[i]), ToFloat32(want[i]))
			}
		}
	}
}
