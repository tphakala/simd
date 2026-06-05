package i32

import (
	"math"
	"math/rand"
	"testing"
)

// Tests for MinMax, the signed int32 per-slice min/max reduction.
//
// The interesting cases mirror the other i32 reductions: the int32 sign-bit
// extremes (MinInt32/MaxInt32), the block-straddling sizes that exercise both
// the vector body and the scalar tail, and the degenerate length-1 and empty
// slices.

// TestMinMaxMatchesOracle checks MinMax against an independent min/max scan
// across the block-straddling sizes.
func TestMinMaxMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 1024, 1025} {
		res := make([]int32, n)
		for i := range res {
			res[i] = int32(rng.Uint32())
		}
		wantMin, wantMax := res[0], res[0]
		for _, r := range res {
			if r < wantMin {
				wantMin = r
			}
			if r > wantMax {
				wantMax = r
			}
		}
		gotMin, gotMax := MinMax(res)
		if gotMin != wantMin || gotMax != wantMax {
			t.Fatalf("n=%d MinMax = (%d, %d), want (%d, %d)", n, gotMin, gotMax, wantMin, wantMax)
		}
	}
}

// TestMinMaxExtremes pins the int32 sign-bit extremes so a kernel that mishandles
// signedness (an unsigned compare, or a lane-corrupting move) is caught.
func TestMinMaxExtremes(t *testing.T) {
	res := []int32{math.MinInt32, math.MaxInt32, -1, 0, 1, math.MinInt32, math.MaxInt32}
	gotMin, gotMax := MinMax(res)
	if gotMin != math.MinInt32 || gotMax != math.MaxInt32 {
		t.Errorf("MinMax extremes = (%d, %d), want (%d, %d)", gotMin, gotMax, math.MinInt32, math.MaxInt32)
	}
}

// TestMinMaxTailPlacement puts the unique extreme at the last index so the
// scalar tail (n mod vector-width) must be folded in for the answer to be right.
// Sizes straddle both the 4-lane (NEON) and 8-lane (AVX) widths.
func TestMinMaxTailPlacement(t *testing.T) {
	for _, n := range []int{1, 2, 4, 5, 8, 9, 12, 16, 17, 31, 33, 100, 1023, 1025} {
		// Baseline values are a tame mid-range; the extremes live only at the end.
		res := make([]int32, n)
		for i := range res {
			res[i] = int32(i % 7)
		}
		res[n-1] = math.MaxInt32
		if n >= 2 {
			res[n-2] = math.MinInt32
		}
		wantMin, wantMax := res[0], res[0]
		for _, r := range res {
			if r < wantMin {
				wantMin = r
			}
			if r > wantMax {
				wantMax = r
			}
		}
		gotMin, gotMax := MinMax(res)
		if gotMin != wantMin || gotMax != wantMax {
			t.Fatalf("n=%d MinMax = (%d, %d), want (%d, %d)", n, gotMin, gotMax, wantMin, wantMax)
		}
	}
}

// TestMinMaxSingle covers the length-1 slice (min == max == the element).
func TestMinMaxSingle(t *testing.T) {
	gotMin, gotMax := MinMax([]int32{-42})
	if gotMin != -42 || gotMax != -42 {
		t.Errorf("MinMax([-42]) = (%d, %d), want (-42, -42)", gotMin, gotMax)
	}
}

// TestMinMaxEmpty pins the documented empty-input contract: (0, 0), no panic.
func TestMinMaxEmpty(t *testing.T) {
	gotMin, gotMax := MinMax(nil)
	if gotMin != 0 || gotMax != 0 {
		t.Errorf("MinMax(nil) = (%d, %d), want (0, 0)", gotMin, gotMax)
	}
	gotMin, gotMax = MinMax([]int32{})
	if gotMin != 0 || gotMax != 0 {
		t.Errorf("MinMax([]) = (%d, %d), want (0, 0)", gotMin, gotMax)
	}
}

func TestMinMaxAllocFree(t *testing.T) {
	res := make([]int32, 1024)
	for i := range res {
		res[i] = int32(i*7 - 3)
	}
	if got := testing.AllocsPerRun(100, func() { _, _ = MinMax(res) }); got != 0 {
		t.Errorf("MinMax allocated %v times per run, want 0", got)
	}
}
