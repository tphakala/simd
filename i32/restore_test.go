package i32

import (
	"math"
	"testing"
)

// Tests for the fixed-predictor decode restoration Restore1..Restore4.
//
// RestoreK is the exact inverse of DiffK: it reconstructs the original samples
// from the [K warm-up | residuals] layout. The strongest check is the
// Restore(Diff(x)) == x round-trip, since Diff (forward difference) and Restore
// (cumulative sum) are independent kernels. Parity against restoreGo, the direct
// decode recurrence, guards the cumulative-sum decomposition the SIMD path uses.

// fillRestoreSrc fills src with values that exercise the sign bit and force
// int32 wraparound at the extremes (matching fillDiffSrc's intent, but defined
// here so the cross-architecture tests do not depend on the amd64/arm64-only
// helpers).
func fillRestoreSrc(src []int32) {
	for i := range src {
		src[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(src) > 1 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
	}
}

// restoreSizes straddle both SIMD block sizes (4 lanes on NEON, 8 on AVX) and
// include inputs shorter than the predictor order (routed to the Go path).
var restoreSizes = []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 1024, 1025}

// restoreGo is the test oracle: the direct decode recurrence that inverts diffGo
// for the order-(len(c)-1) fixed predictor. The first 'order' entries are the
// verbatim warm-up (dst[i]=src[i]) and, for n>=order,
//
//	dst[n] = src[n] - sum_{j>=1} c[j]*dst[n-j]
//
// which solves src[n] = sum_j c[j]*dst[n-j] (the diffGo combination) for dst[n].
// This serial recurrence is independent of the cumulative-sum decomposition the
// package ships, so the two agreeing is a meaningful cross-check. int32 wraps.
func restoreGo(dst, src, c []int32) {
	order := len(c) - 1
	n := min(len(dst), len(src))
	w := min(order, n)
	copy(dst[:w], src[:w])
	for nn := order; nn < n; nn++ {
		acc := src[nn]
		for j := 1; j < len(c); j++ {
			acc -= c[j] * dst[nn-j]
		}
		dst[nn] = acc
	}
}

func restore1Go(dst, src []int32) { restoreGo(dst, src, fixedCoeffs1) }
func restore2Go(dst, src []int32) { restoreGo(dst, src, fixedCoeffs2) }
func restore3Go(dst, src []int32) { restoreGo(dst, src, fixedCoeffs3) }
func restore4Go(dst, src []int32) { restoreGo(dst, src, fixedCoeffs4) }

func TestRestoreRoundTrip(t *testing.T) {
	diffs := []func(dst, src []int32){Diff1, Diff2, Diff3, Diff4}
	restores := []func(dst, src []int32){Restore1, Restore2, Restore3, Restore4}
	for _, n := range restoreSizes {
		src := make([]int32, n)
		fillRestoreSrc(src)
		for order := 1; order <= 4; order++ {
			residual := make([]int32, n)
			diffs[order-1](residual, src)
			got := make([]int32, n)
			restores[order-1](got, residual)
			for i := range src {
				if got[i] != src[i] {
					t.Fatalf("n=%d order=%d Restore(Diff)[%d] = %d, want %d", n, order, i, got[i], src[i])
				}
			}
		}
	}
}

func TestRestoreMatchesDirectRecurrence(t *testing.T) {
	restores := []func(dst, src []int32){Restore1, Restore2, Restore3, Restore4}
	refs := []func(dst, src []int32){restore1Go, restore2Go, restore3Go, restore4Go}
	for _, n := range restoreSizes {
		res := make([]int32, n)
		fillRestoreSrc(res) // treat as an arbitrary residual stream
		for order := 1; order <= 4; order++ {
			got := make([]int32, n)
			want := make([]int32, n)
			restores[order-1](got, res)
			refs[order-1](want, res)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("n=%d order=%d Restore[%d] = %d, want %d (direct recurrence)", n, order, i, got[i], want[i])
				}
			}
		}
	}
}

// TestRestore1Simple checks order-1 restoration against a hand-computed result.
func TestRestore1Simple(t *testing.T) {
	// Diff1 of [10,13,13,8,20] is [10,3,0,-5,12]; Restore1 inverts it.
	src := []int32{10, 3, 0, -5, 12}
	dst := make([]int32, len(src))
	Restore1(dst, src)
	want := []int32{10, 13, 13, 8, 20}
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("Restore1[%d] = %d, want %d", i, dst[i], w)
		}
	}
}

// TestRestoreWraps verifies int32 wraparound: restoring residuals at the type
// extremes must wrap exactly like the scalar recurrence.
func TestRestoreWraps(t *testing.T) {
	src := []int32{math.MinInt32, math.MaxInt32, 1, math.MaxInt32, -1, 0}
	got := make([]int32, len(src))
	want := make([]int32, len(src))
	Restore1(got, src)
	restore1Go(want, src)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Restore1 wrap [%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestRestore_Clamp(t *testing.T) {
	src := []int32{1, 1, 2, 3, 4}
	dst := make([]int32, 100)
	Restore1(dst, src)
	want := []int32{1, 2, 4, 7, 11} // cumulative sum of src
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("Restore1 clamp dst[%d] = %d, want %d", i, dst[i], w)
		}
	}
	for i := len(want); i < len(dst); i++ {
		if dst[i] != 0 {
			t.Errorf("Restore1 wrote past clamp at dst[%d] = %d, want untouched 0", i, dst[i])
		}
	}
}

// TestRestore_ShortInput checks inputs shorter than the order: with fewer than
// K samples there is no residual, only warm-up copied verbatim.
func TestRestore_ShortInput(t *testing.T) {
	src := []int32{42, 7}
	dst := make([]int32, len(src))
	Restore4(dst, src) // order 4 but only 2 samples
	if dst[0] != 42 || dst[1] != 7 {
		t.Errorf("Restore4 short input = %v, want [42 7] warm-up", dst)
	}
}

func TestRestore_Empty(t *testing.T) {
	Restore1(nil, nil)
	Restore2([]int32{}, []int32{})
	Restore3(nil, nil)
	Restore4(nil, nil)
	dst := []int32{99}
	Restore1(dst, nil)
	if dst[0] != 99 {
		t.Errorf("Restore1 wrote on empty input: %v", dst)
	}
}

func TestRestore_AllocFree(t *testing.T) {
	src := make([]int32, 1024)
	fillRestoreSrc(src)
	dst := make([]int32, 1024)
	restores := map[string]func(dst, src []int32){"Restore1": Restore1, "Restore2": Restore2, "Restore3": Restore3, "Restore4": Restore4}
	for name, fn := range restores {
		if got := testing.AllocsPerRun(100, func() { fn(dst, src) }); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", name, got)
		}
	}
}
