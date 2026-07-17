package i16

import (
	"math"
	"testing"
)

// Tests for Abs and MaxAbs, the wrapping absolute value and the abs-max
// reduction. The load-bearing input is MinInt16: Abs must wrap it in place
// while MaxAbs must report its magnitude as the out-of-int16 value 32768,
// and the two contracts meet in no other case.

// absOracle computes |v| in int, independent of absGo, narrowed to int16 the
// same way a 16-bit lane store truncates.
func absOracle(v int16) int16 {
	av := int(v)
	if av < 0 {
		av = -av
	}
	return int16(av)
}

func TestAbs(t *testing.T) {
	for _, n := range tier3Lengths {
		a := genI16(n, 81)
		dst := make([]int16, n)
		ref := make([]int16, n)
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

// TestAbs_MinInt16 walks the wrap input through every lane position: a single
// planted MinInt16 must come back unchanged while its neighbours are truly
// absolute. The rotating position is what catches lane and index errors; an
// all-MinInt16 fill could only catch a miscount.
func TestAbs_MinInt16(t *testing.T) {
	for n := 1; n <= 40; n++ {
		for pos := range n {
			a := genI16(n, uint32(90+n))
			a[pos] = math.MinInt16
			dst := make([]int16, n)
			Abs(dst, a)
			for i := range dst {
				if want := absOracle(a[i]); dst[i] != want {
					t.Fatalf("Abs n=%d pos=%d: dst[%d] = %d, want %d", n, pos, i, dst[i], want)
				}
			}
		}
	}
}

// TestAbs_TailUntouched plants sentinels past the clamp point at n=19: one
// 16-wide AVX2 block plus a 3-tail, and two 8-wide NEON blocks plus the same
// tail, so both vector bodies run and both scalar tails must stop exactly
// at n.
func TestAbs_TailUntouched(t *testing.T) {
	const n = 19
	a := genI16(n, 83)
	dst := make([]int16, n+8)
	for i := range dst {
		dst[i] = math.MaxInt16 // sentinel
	}
	Abs(dst[:n], a)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt16 {
			t.Errorf("Abs wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestAbs_Clamp covers mismatched lengths in both directions plus the empty
// no-op.
func TestAbs_Clamp(t *testing.T) {
	a := genI16(40, 84)
	short := make([]int16, 25)
	Abs(short, a) // dst shorter: n = 25
	for i := range short {
		if want := absOracle(a[i]); short[i] != want {
			t.Fatalf("Abs short dst: dst[%d] = %d, want %d", i, short[i], want)
		}
	}
	long := make([]int16, 40)
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
	one := []int16{42}
	Abs(one, nil)
	if one[0] != 42 {
		t.Errorf("Abs wrote on empty input: %v", one)
	}
}

// TestAbs_UnalignedOperands sweeps all eight element offsets, holding dst and
// a at different offsets from each other, so neither is reliably aligned and
// an aligned-load or aligned-store substitution cannot survive the suite: a
// 16-byte-aligned access needs an element offset that is a multiple of 8,
// which only one of the eight iterations supplies. MaxAbs rides the same
// windows so its loads are pinned too.
func TestAbs_UnalignedOperands(t *testing.T) {
	const span = 300
	base := genI16(span, 87)
	backing := make([]int16, span)
	for _, n := range []int{16, 17, 19, 25, 33, 64, 240} {
		for off := range 8 {
			a := base[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			Abs(dst, a)
			for i := range n {
				if want := absOracle(a[i]); dst[i] != want {
					t.Fatalf("Abs unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
			if got, want := MaxAbs(a), maxAbsGo(a); got != want {
				t.Fatalf("MaxAbs unaligned n=%d off=%d: got %d, want %d", n, off, got, want)
			}
		}
	}
}

// TestAbs_AllocFree: buffers INSIDE the closure, see TestMulQ15_AllocFree.
func TestAbs_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a, dst [240]int16
		Abs(dst[:], a[:])
	}); n != 0 {
		t.Errorf("Abs forces %v caller allocations per run, want 0", n)
	}
}

func TestMaxAbs(t *testing.T) {
	if got := MaxAbs(nil); got != 0 {
		t.Errorf("MaxAbs(nil) = %d, want 0", got)
	}
	for _, n := range tier3Lengths {
		a := genI16(n, 85)
		got := MaxAbs(a)
		if want := maxAbsGo(a); got != want {
			t.Errorf("MaxAbs n=%d: got %d, want %d (reference)", n, got, want)
		}
		// Independent oracle: an int scan over the same data.
		oracle := 0
		for _, v := range a {
			av := int(v)
			if av < 0 {
				av = -av
			}
			oracle = max(oracle, av)
		}
		if got != oracle {
			t.Errorf("MaxAbs n=%d: got %d, want %d (oracle)", n, got, oracle)
		}
	}
}

// TestMaxAbs_MinInt16 pins the widened result: a planted -32768 must report
// 32768 (not 32767 and not -32768) from every lane position and from the
// scalar tail, at lengths that reach the vector bodies.
func TestMaxAbs_MinInt16(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 19, 24, 31, 32, 33, 64, 100} {
		for pos := range n {
			a := make([]int16, n)
			for i := range a {
				a[i] = int16(i%100 - 50) // tame body keeps the plant the unique max
			}
			a[pos] = math.MinInt16
			if got := MaxAbs(a); got != 32768 {
				t.Fatalf("MaxAbs n=%d pos=%d: got %d, want 32768", n, pos, got)
			}
		}
	}
	// Without the plant the result stays inside int16 range.
	a := genI16(33, 86)
	for i := range a {
		if a[i] == math.MinInt16 {
			a[i] = math.MaxInt16
		}
	}
	a[17] = math.MaxInt16
	if got := MaxAbs(a); got != math.MaxInt16 {
		t.Errorf("MaxAbs without MinInt16 = %d, want %d", got, math.MaxInt16)
	}
}

// TestMaxAbs_NoOverRead hands MaxAbs a prefix of a longer allocation whose
// every element past the prefix is -32768, the input with the largest
// possible magnitude. Past a standalone slice lies zeroed memory and |0| is
// the identity element of this reduction, so an over-reading kernel would
// pass every other test in this file; the planted extreme is what makes one
// visible. Unlike the wrapping sum, max is idempotent, so no repeat count of
// the extreme can cancel it and any over-read shows up.
//
// The slack past the longest case must cover a whole vector block of the
// widest kernel this can dispatch to, which is AVX2 at 16 int16. With less,
// a full-block over-read runs off the end of the backing array itself and
// reads unpoisoned memory, so the test stops reliably failing on the defect
// it exists to catch.
func TestMaxAbs_NoOverRead(t *testing.T) {
	backing := make([]int16, 128+16)
	for i := range backing {
		backing[i] = math.MinInt16
	}
	for _, n := range []int{1, 3, 7, 8, 9, 15, 16, 17, 19, 24, 31, 32, 33, 64, 100, 128} {
		a := backing[:n]
		for i := range a {
			a[i] = int16(i%50 - 25)
		}
		want := maxAbsGo(a)
		if got := MaxAbs(a); got != want {
			t.Fatalf("MaxAbs n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = math.MinInt16 // restore the poison for the next length
		}
	}
}

// TestMaxAbs_AllocFree: buffers INSIDE the closure, see TestMulQ15_AllocFree.
func TestMaxAbs_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a [240]int16
		_ = MaxAbs(a[:])
	}); n != 0 {
		t.Errorf("MaxAbs forces %v caller allocations per run, want 0", n)
	}
}
