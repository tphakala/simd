package i32

import (
	"math"
	"testing"
)

// Tests for Abs, the wrapping int32 absolute value. The load-bearing input is
// MinInt32, whose magnitude does not fit int32 and must wrap in place. Kept
// separate from arith_test.go: Abs is single-operand and its edge case is the
// wrap, not two-operand overflow.

// absOracle computes |v| in int64, independent of absGo, narrowed to int32
// the same way a 32-bit lane store truncates.
func absOracle(v int32) int32 {
	av := int64(v)
	if av < 0 {
		av = -av
	}
	return int32(av)
}

func TestAbs(t *testing.T) {
	for _, n := range tier3Lengths {
		a := genI32(n, 21)
		dst := make([]int32, n)
		ref := make([]int32, n)
		Abs(dst, a)
		absGo(ref, a)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf("Abs n=%d: dst[%d] = %d, want %d (reference)", n, i, dst[i], ref[i])
			}
			if want := absOracle(a[i]); dst[i] != want {
				t.Fatalf("Abs n=%d: dst[%d] = %d, want %d (oracle)", n, i, dst[i], want)
			}
		}
	}
}

// TestAbs_MinInt32 walks the wrap input through every lane position: a single
// planted MinInt32 must come back unchanged while its neighbours are truly
// absolute. The rotating position is what catches lane and index errors.
func TestAbs_MinInt32(t *testing.T) {
	for n := 1; n <= 24; n++ {
		for pos := range n {
			a := genI32(n, uint32(40+n))
			a[pos] = math.MinInt32
			dst := make([]int32, n)
			Abs(dst, a)
			for i := range dst {
				if want := absOracle(a[i]); dst[i] != want {
					t.Fatalf("Abs n=%d pos=%d: dst[%d] = %d, want %d", n, pos, i, dst[i], want)
				}
			}
		}
	}
}

// TestAbs_TailUntouched plants sentinels past the clamp point at n=11: one
// 8-wide AVX2 block plus a 3-tail, and two 4-wide NEON blocks plus the same
// tail, so both vector bodies run and both scalar tails must stop exactly
// at n.
func TestAbs_TailUntouched(t *testing.T) {
	const n = 11
	a := genI32(n, 42)
	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	Abs(dst[:n], a)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("Abs wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestAbs_Clamp covers mismatched lengths in both directions plus the empty
// no-op.
func TestAbs_Clamp(t *testing.T) {
	a := genI32(40, 43)
	short := make([]int32, 25)
	Abs(short, a) // dst shorter: n = 25
	for i := range short {
		if want := absOracle(a[i]); short[i] != want {
			t.Fatalf("Abs short dst: dst[%d] = %d, want %d", i, short[i], want)
		}
	}
	long := make([]int32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	Abs(long, a[:25]) // a shorter: long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("Abs wrote past clamp at dst[%d] = %d", i, long[i])
		}
	}
	Abs(nil, nil)
	one := []int32{42}
	Abs(one, nil)
	if one[0] != 42 {
		t.Errorf("Abs wrote on empty input: %v", one)
	}
}

// TestAbs_UnalignedOperands sweeps all eight element offsets, holding dst and
// a at different offsets from each other, so neither is reliably aligned and
// an aligned-load or aligned-store substitution cannot survive the suite: a
// 16-byte-aligned access needs an element offset that is a multiple of 4, and
// the sweep leaves most iterations 4-byte aligned rather than 16- or 32-byte.
// Sum rides the same windows so its loads are pinned too.
func TestAbs_UnalignedOperands(t *testing.T) {
	const span = 300
	base := genI32(span, 44)
	backing := make([]int32, span)
	for _, n := range []int{8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			a := base[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			Abs(dst, a)
			for i := range n {
				if want := absOracle(a[i]); dst[i] != want {
					t.Fatalf("Abs unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
			if got, want := Sum(a), sumOracle(a); got != want {
				t.Fatalf("Sum unaligned n=%d off=%d: got %d, want %d", n, off, got, want)
			}
		}
	}
}

// TestAbs_AllocFree declares the buffers INSIDE the measured closure, see
// TestSum_AllocFree.
func TestAbs_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a, dst [1000]int32
		Abs(dst[:], a[:])
	}); n != 0 {
		t.Errorf("Abs forces %v caller allocations per run, want 0", n)
	}
}
