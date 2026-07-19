package i32

import (
	"math"
	"testing"
)

// Tests for Butterfly, the in-place radix-2 Hadamard/Haar step
// lo, hi = lo+hi, lo-hi. Both the sum and the difference wrap in int32 rather
// than saturating, so the load-bearing cases are the overflow of lo+hi and the
// underflow of lo-hi: a saturating build would clamp to MaxInt32/MinInt32 and
// fail these. lo and hi must be distinct slices; overlap is unsupported.

// butterflyOracle computes the sum and difference in int64 and truncates each
// back to int32, fully independent of the direct int32 arithmetic in butterflyGo
// and the SIMD kernels. int32 add and sub are arithmetic modulo 2^32, so
// truncating the exact int64 result reproduces the wrap exactly; it pins the
// reference rather than trusting a formula shared with it.
func butterflyOracle(lo, hi int32) (s, d int32) {
	return int32(int64(lo) + int64(hi)), int32(int64(lo) - int64(hi))
}

// TestButterflyOracle confirms the oracle itself encodes the wrap, so the parity
// tests below rest on a checked foundation: MinInt32-1 wraps to MaxInt32 and
// MaxInt32+1 wraps to MinInt32, the two behaviors a saturating build gets wrong.
func TestButterflyOracle(t *testing.T) {
	if s, d := butterflyOracle(math.MinInt32, 1); s != math.MinInt32+1 || d != math.MaxInt32 {
		t.Fatalf("oracle(MinInt32,1) = (%d,%d), want (%d,%d)", s, d, int32(math.MinInt32+1), int32(math.MaxInt32))
	}
	if s, d := butterflyOracle(math.MaxInt32, 1); s != math.MinInt32 || d != math.MaxInt32-1 {
		t.Fatalf("oracle(MaxInt32,1) = (%d,%d), want (%d,%d)", s, d, int32(math.MinInt32), int32(math.MaxInt32-1))
	}
}

// TestButterfly sweeps every tier-3 length against both the pure-Go reference and
// the arbitrary-precision-free int64 oracle, so a fault cannot hide by agreeing
// with the reference alone. Index 0 carries MinInt32/1 (the difference underflows
// to MaxInt32) and the last index MaxInt32/1 (the sum overflows to MinInt32), so
// both wraps are exercised at every length, at the head and riding the tail.
func TestButterfly(t *testing.T) {
	for _, n := range tier3Lengths {
		lo := genI32(n, 51)
		hi := genI32(n, 52)
		if n > 0 {
			lo[0], hi[0] = math.MinInt32, 1
			lo[n-1], hi[n-1] = math.MaxInt32, 1
		}
		origLo := append([]int32(nil), lo...)
		origHi := append([]int32(nil), hi...)

		refLo := append([]int32(nil), lo...)
		refHi := append([]int32(nil), hi...)
		butterflyGo(refLo, refHi)

		Butterfly(lo, hi)
		for i := range lo {
			if lo[i] != refLo[i] || hi[i] != refHi[i] {
				t.Fatalf("Butterfly n=%d: at %d got (%d,%d), reference (%d,%d)", n, i, lo[i], hi[i], refLo[i], refHi[i])
			}
			if ws, wd := butterflyOracle(origLo[i], origHi[i]); lo[i] != ws || hi[i] != wd {
				t.Fatalf("Butterfly n=%d: at %d got (%d,%d), oracle (%d,%d)", n, i, lo[i], hi[i], ws, wd)
			}
		}
	}
}

// TestButterfly_ValueMatrix crosses the load-bearing samples for lo and hi and
// plants each pair in every lane position across a length that spans a vector
// block plus a scalar tail on both arches, so a lane error, an index error, and a
// value the wrap mishandles are all caught at whichever position exposes them.
func TestButterfly_ValueMatrix(t *testing.T) {
	vals := []int32{math.MinInt32, math.MaxInt32, 0, -1, 1, 2, 0x12345678, -0x12345678, 0x7FFFFFFE}
	const n = 11 // one 8-wide AVX2 block + 3 tail; two 4-wide NEON blocks + 3 tail
	fillerLo := genI32(n, 53)
	fillerHi := genI32(n, 54)
	for _, lv := range vals {
		for _, hv := range vals {
			for pos := range n {
				lo := append([]int32(nil), fillerLo...)
				hi := append([]int32(nil), fillerHi...)
				lo[pos] = lv
				hi[pos] = hv
				origLo := append([]int32(nil), lo...)
				origHi := append([]int32(nil), hi...)
				Butterfly(lo, hi)
				for i := range lo {
					ws, wd := butterflyOracle(origLo[i], origHi[i])
					if lo[i] != ws || hi[i] != wd {
						t.Fatalf("Butterfly lv=%d hv=%d pos=%d: at %d got (%d,%d), want (%d,%d)", lv, hv, pos, i, lo[i], hi[i], ws, wd)
					}
				}
			}
		}
	}
}

// TestButterfly_Wrap pins the two extreme contracts in isolation. n=11 forces
// both the vector body and the scalar tail on both arches. lo=MinInt32,hi=1 gives
// sum MinInt32+1 and difference MinInt32-1, which wraps to MaxInt32 (a saturating
// build would return MinInt32); lo=MaxInt32,hi=1 gives sum MaxInt32+1, which wraps
// to MinInt32 (a saturating build would return MaxInt32).
func TestButterfly_Wrap(t *testing.T) {
	const n = 11
	lo := make([]int32, n)
	hi := make([]int32, n)

	for i := range lo {
		lo[i], hi[i] = math.MinInt32, 1
	}
	Butterfly(lo, hi)
	for i := range lo {
		if lo[i] != math.MinInt32+1 {
			t.Fatalf("Butterfly MinInt32+1 sum: lo[%d] = %d, want %d", i, lo[i], int32(math.MinInt32+1))
		}
		if hi[i] != math.MaxInt32 {
			t.Fatalf("Butterfly MinInt32-1 diff: hi[%d] = %d, want %d (wrap, not saturate)", i, hi[i], int32(math.MaxInt32))
		}
	}

	for i := range lo {
		lo[i], hi[i] = math.MaxInt32, 1
	}
	Butterfly(lo, hi)
	for i := range lo {
		if lo[i] != math.MinInt32 {
			t.Fatalf("Butterfly MaxInt32+1 sum: lo[%d] = %d, want %d (wrap, not saturate)", i, lo[i], int32(math.MinInt32))
		}
		if hi[i] != math.MaxInt32-1 {
			t.Fatalf("Butterfly MaxInt32-1 diff: hi[%d] = %d, want %d", i, hi[i], int32(math.MaxInt32-1))
		}
	}
}

// TestButterfly_TailUntouched plants sentinels past the clamp point at n=11 (one
// 8-wide AVX2 block + 3 tail, two 4-wide NEON blocks + the same tail) in both
// slices, so both vector bodies run and both scalar tails must stop exactly at n.
func TestButterfly_TailUntouched(t *testing.T) {
	const n = 11
	lo := genI32(n+8, 55)
	hi := genI32(n+8, 56)
	for i := n; i < len(lo); i++ {
		lo[i] = math.MaxInt32
		hi[i] = math.MinInt32
	}
	Butterfly(lo[:n], hi[:n])
	for i := n; i < len(lo); i++ {
		if lo[i] != math.MaxInt32 {
			t.Errorf("Butterfly wrote past end of lo at %d = %d", i, lo[i])
		}
		if hi[i] != math.MinInt32 {
			t.Errorf("Butterfly wrote past end of hi at %d = %d", i, hi[i])
		}
	}
}

// TestButterfly_Clamp covers mismatched lengths and the empty no-op: n is the
// shorter of lo and hi, only the first n of each are updated, and nothing past it
// is touched.
func TestButterfly_Clamp(t *testing.T) {
	// lo shorter: n = len(lo) = 25.
	lo := genI32(25, 57)
	hi := genI32(40, 58)
	origLo := append([]int32(nil), lo...)
	origHi := append([]int32(nil), hi...)
	Butterfly(lo, hi)
	for i := range lo {
		if ws, wd := butterflyOracle(origLo[i], origHi[i]); lo[i] != ws || hi[i] != wd {
			t.Fatalf("Butterfly clamp lo: at %d got (%d,%d), want (%d,%d)", i, lo[i], hi[i], ws, wd)
		}
	}
	for i := 25; i < len(hi); i++ {
		if hi[i] != origHi[i] {
			t.Fatalf("Butterfly wrote past n into hi[%d] = %d, want %d", i, hi[i], origHi[i])
		}
	}

	// hi shorter: n = len(hi) = 25.
	lo2 := genI32(40, 59)
	hi2 := genI32(25, 60)
	origLo2 := append([]int32(nil), lo2...)
	origHi2 := append([]int32(nil), hi2...)
	Butterfly(lo2, hi2)
	for i := range hi2 {
		if ws, wd := butterflyOracle(origLo2[i], origHi2[i]); lo2[i] != ws || hi2[i] != wd {
			t.Fatalf("Butterfly clamp hi: at %d got (%d,%d), want (%d,%d)", i, lo2[i], hi2[i], ws, wd)
		}
	}
	for i := 25; i < len(lo2); i++ {
		if lo2[i] != origLo2[i] {
			t.Fatalf("Butterfly wrote past n into lo[%d] = %d, want %d", i, lo2[i], origLo2[i])
		}
	}

	// Empty inputs are a no-op.
	Butterfly(nil, nil)
	one := []int32{42}
	Butterfly(one, nil)
	if one[0] != 42 {
		t.Errorf("Butterfly wrote on empty input: %v", one)
	}
}

// TestButterfly_DistinctSlices confirms the documented contract that lo and hi
// are distinct backing arrays and transform correctly. It deliberately does NOT
// exercise overlapping lo/hi, which is unsupported: the SIMD kernels load a whole
// block of both before storing either, so an overlap would clobber unread input.
func TestButterfly_DistinctSlices(t *testing.T) {
	const n = 64
	lo := genI32(n, 63)
	hi := genI32(n, 64)
	origLo := append([]int32(nil), lo...)
	origHi := append([]int32(nil), hi...)
	Butterfly(lo, hi)
	for i := range lo {
		if ws, wd := butterflyOracle(origLo[i], origHi[i]); lo[i] != ws || hi[i] != wd {
			t.Fatalf("Butterfly distinct n=%d: at %d got (%d,%d), want (%d,%d)", n, i, lo[i], hi[i], ws, wd)
		}
	}
}

// TestButterfly_UnalignedOperands sweeps all eight element offsets into separate
// backing arrays for lo and hi, so neither slice starts at a reliable alignment
// and an aligned-load or aligned-store substitution cannot survive.
func TestButterfly_UnalignedOperands(t *testing.T) {
	const span = 320
	backingLo := genI32(span, 61)
	backingHi := genI32(span, 62)
	for _, n := range []int{4, 5, 8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			lo := backingLo[off+1 : off+1+n]
			hi := backingHi[off+1 : off+1+n]
			origLo := append([]int32(nil), lo...)
			origHi := append([]int32(nil), hi...)
			Butterfly(lo, hi)
			for i := range n {
				if ws, wd := butterflyOracle(origLo[i], origHi[i]); lo[i] != ws || hi[i] != wd {
					t.Fatalf("Butterfly unaligned n=%d off=%d: at %d got (%d,%d), want (%d,%d)", n, off, i, lo[i], hi[i], ws, wd)
				}
			}
		}
	}
}

// TestButterfly_AllocFree declares the buffers INSIDE the measured closure so only
// allocations forced by the butterfly itself are counted.
func TestButterfly_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var lo, hi [1000]int32
		Butterfly(lo[:], hi[:])
	}); n != 0 {
		t.Errorf("Butterfly forces %v caller allocations per run, want 0", n)
	}
}

// FuzzButterfly differentially fuzzes the dispatched Butterfly against the pure-Go
// reference and the int64 oracle. The raw bytes are split into two equal halves
// (lo and hi) copied into distinct backing arrays, so tail handling and the wrap
// are explored past the hand-picked seeds. It reuses addLenSeeds, whose element
// counts bracket the 8/16-lane unroll boundaries.
func FuzzButterfly(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i32sFromBits(raw)
		h := len(v) / 2
		lo := append([]int32(nil), v[:h]...)
		hi := append([]int32(nil), v[h:2*h]...)
		origLo := append([]int32(nil), lo...)
		origHi := append([]int32(nil), hi...)

		refLo := append([]int32(nil), lo...)
		refHi := append([]int32(nil), hi...)
		butterflyGo(refLo, refHi)

		Butterfly(lo, hi)
		equalI32(t, "Butterfly.lo", lo, refLo)
		equalI32(t, "Butterfly.hi", hi, refHi)
		for i := range lo {
			if ws, wd := butterflyOracle(origLo[i], origHi[i]); lo[i] != ws || hi[i] != wd {
				t.Fatalf("Butterfly oracle mismatch at %d: got (%d,%d) want (%d,%d) (len=%d)", i, lo[i], hi[i], ws, wd, len(lo))
			}
		}
	})
}
