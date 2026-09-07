package i16

import (
	"math"
	"testing"
)

func TestMinMax(t *testing.T) {
	if lo, hi := MinMax(nil); lo != 0 || hi != 0 {
		t.Errorf("MinMax(nil) = (%d,%d), want (0,0)", lo, hi)
	}
	if lo, hi := MinMax([]int16{}); lo != 0 || hi != 0 {
		t.Errorf("MinMax(empty) = (%d,%d), want (0,0)", lo, hi)
	}
	if lo, hi := MinMax([]int16{5}); lo != 5 || hi != 5 {
		t.Errorf("MinMax([5]) = (%d,%d), want (5,5)", lo, hi)
	}
	if lo, hi := MinMax([]int16{0, -32768, 32767, 3, -1}); lo != -32768 || hi != 32767 {
		t.Errorf("MinMax = (%d,%d), want (-32768,32767)", lo, hi)
	}

	// Parity across the sweep against the reference and an independent scan.
	for _, n := range tier3Lengths {
		if n == 0 {
			continue
		}
		a := genI16(n, 92)
		gotLo, gotHi := MinMax(a)
		wantLo, wantHi := minMaxGo(a)
		if gotLo != wantLo || gotHi != wantHi {
			t.Errorf("MinMax n=%d: got (%d,%d), want (%d,%d)", n, gotLo, gotHi, wantLo, wantHi)
		}
		oracleLo, oracleHi := a[0], a[0]
		for _, v := range a {
			if v < oracleLo {
				oracleLo = v
			}
			if v > oracleHi {
				oracleHi = v
			}
		}
		if gotLo != oracleLo || gotHi != oracleHi {
			t.Errorf("MinMax n=%d: got (%d,%d), want (%d,%d) (oracle)", n, gotLo, gotHi, oracleLo, oracleHi)
		}
	}
}

// TestMinMax_Extremes plants the type extremes at every lane position and the
// scalar tail, at lengths that reach the vector bodies: the signed min must
// report -32768 and the max +32767 rather than an unsigned or swapped result.
func TestMinMax_Extremes(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 19, 24, 31, 32, 33, 64, 100} {
		for pos := range n {
			// Plant the minimum against a tame body; the max stays inside.
			a := make([]int16, n)
			for i := range a {
				a[i] = int16(i%100 - 50)
			}
			a[pos] = math.MinInt16
			if lo, _ := MinMax(a); lo != math.MinInt16 {
				t.Fatalf("MinMax n=%d pos=%d min: got %d, want %d", n, pos, lo, math.MinInt16)
			}
			// Plant the maximum against the same tame body.
			b := make([]int16, n)
			for i := range b {
				b[i] = int16(i%100 - 50)
			}
			b[pos] = math.MaxInt16
			if _, hi := MinMax(b); hi != math.MaxInt16 {
				t.Fatalf("MinMax n=%d pos=%d max: got %d, want %d", n, pos, hi, math.MaxInt16)
			}
		}
	}
}

// TestMinMax_AllEqual: a constant slice must report (c, c). Uses INT16_MIN so a
// kernel that seeded its max accumulator from a wrong constant would show.
func TestMinMax_AllEqual(t *testing.T) {
	for _, n := range []int{1, 8, 16, 17, 33, 64} {
		for _, c := range []int16{math.MinInt16, -1, 0, 1, math.MaxInt16} {
			a := make([]int16, n)
			for i := range a {
				a[i] = c
			}
			if lo, hi := MinMax(a); lo != c || hi != c {
				t.Fatalf("MinMax n=%d all=%d: got (%d,%d), want (%d,%d)", n, c, lo, hi, c, c)
			}
		}
	}
}

// TestMinMax_AllocFree: buffers INSIDE the closure, see TestMaxAbs_AllocFree.
func TestMinMax_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a [240]int16
		_, _ = MinMax(a[:])
	}); n != 0 {
		t.Errorf("MinMax forces %v caller allocations per run, want 0", n)
	}
}
