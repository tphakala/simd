package i32

import (
	"encoding/binary"
	"math"
	"testing"
)

// Tests for FIRValidQ15, the int32 valid convolution (correlation orientation)
// with int16 Q15 taps. Each tap product is Q15-TRUNCATED before it is added and
// the accumulator wraps in int32, so the load-bearing behaviors are (1) per-
// product truncation, distinct from truncating the final sum, and (2) the
// two's-complement wrap, distinct from saturation.

// genI16 produces a deterministic spread of int16 taps from the same LCG as
// genI32, taking the high 16 bits so sign and magnitude vary at every index.
func genI16(n int, seed uint32) []int16 {
	s := make([]int16, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int16(x >> 16)
	}
	return s
}

// i16sFromBits reinterprets raw bytes as little-endian int16s, one per 2-byte
// chunk, organically reaching MinInt16/MaxInt16 where the sign extension into the
// signed multiply is most likely to be mishandled.
func i16sFromBits(raw []byte) []int16 {
	out := make([]int16, len(raw)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return out
}

// firValidQ15Oracle computes the valid convolution independently of
// firValidQ15Go on two axes. The per-tap Q15 shift is a manual floor division by
// 2^15 (not the >> operator the reference uses), and the taps are accumulated as
// an exact int64 sum truncated to int32 once at the end (not the per-add int32
// wrap the reference performs). The two are equal because wrapping int32 addition
// is addition modulo 2^32 and the exact sum of a handful of int32 products cannot
// overflow int64, so int32(exactSum) reproduces the wrapping accumulate. It pins
// the reference rather than trusting a formula shared with it.
func firValidQ15Oracle(dst, x []int32, taps []int16) {
	if len(taps) == 0 || len(x) < len(taps) {
		return
	}
	outLen := len(x) - len(taps) + 1
	n := min(len(dst), outLen)
	const q = int64(1) << 15
	for i := range n {
		var acc int64 // exact sum of the truncated int32 products
		for j := range taps {
			prod := int64(taps[j]) * int64(x[i+j]) // exact: |prod| <= 2^46
			// Floor-divide by 2^15 (truncation toward -inf), computed without >>.
			shifted := prod / q
			if prod%q != 0 && prod < 0 {
				shifted--
			}
			acc += int64(int32(shifted)) // per-product int32 truncation (2^31 -> MinInt32)
		}
		dst[i] = int32(acc) // single wrap of the exact sum == the per-add wrapping accumulate
	}
}

// TestFIRValidQ15OracleSelfCheck confirms the oracle encodes per-product
// truncation toward -inf and the wrap, so the parity tests rest on a checked
// foundation. -1 (Q15) times a small positive sample yields a negative product
// that floors below zero; a positive product just below 2^15 truncates to zero.
func TestFIRValidQ15OracleSelfCheck(t *testing.T) {
	// taps=[-1] over x=[1]: product = -1, floor(-1/2^15) = -1, not 0.
	dst := make([]int32, 1)
	firValidQ15Oracle(dst, []int32{1}, []int16{-1})
	if dst[0] != -1 {
		t.Fatalf("oracle floor: got %d, want -1 (truncation toward -inf)", dst[0])
	}
	// taps=[1] over x=[0x7FFF]: product = 32767 < 2^15, truncates to 0.
	firValidQ15Oracle(dst, []int32{0x7FFF}, []int16{1})
	if dst[0] != 0 {
		t.Fatalf("oracle truncate: got %d, want 0", dst[0])
	}
}

// TestFIRValidQ15 sweeps a grid of (len(x), len(taps)) against both the pure-Go
// reference and the independent oracle, so a fault cannot hide by agreeing with
// the reference alone. The tap counts include 5 (the combFilterConst driver), and
// the x lengths span the vector-output loop plus a scalar-output tail on both
// arches as well as short outputs that route to the Go path. MinInt32/MaxInt32
// ride the ends of x and MinInt16/MaxInt16 the ends of taps so the extremes are
// always in play.
func TestFIRValidQ15(t *testing.T) {
	tapCounts := []int{1, 2, 3, 5, 8, 16}
	xLens := []int{1, 2, 3, 4, 5, 8, 9, 11, 16, 17, 20, 24, 31, 32, 33, 40, 64, 65, 100, 128}
	for _, kl := range tapCounts {
		taps := genI16(kl, uint32(kl)*101+7)
		taps[0] = math.MinInt16
		taps[kl-1] = math.MaxInt16
		for _, xl := range xLens {
			x := genI32(xl, uint32(xl)*13+uint32(kl))
			x[0] = math.MinInt32
			x[xl-1] = math.MaxInt32

			outLen := 0
			if xl >= kl {
				outLen = xl - kl + 1
			}
			dst := make([]int32, outLen)
			ref := make([]int32, outLen)
			orc := make([]int32, outLen)
			FIRValidQ15(dst, x, taps)
			firValidQ15Go(ref, x, taps)
			firValidQ15Oracle(orc, x, taps)
			for i := range dst {
				if dst[i] != ref[i] {
					t.Fatalf("FIRValidQ15 kl=%d xl=%d: dst[%d] = %d, want %d (reference)", kl, xl, i, dst[i], ref[i])
				}
				if dst[i] != orc[i] {
					t.Fatalf("FIRValidQ15 kl=%d xl=%d: dst[%d] = %d, want %d (oracle)", kl, xl, i, dst[i], orc[i])
				}
			}
		}
	}
}

// TestFIRValidQ15_PerProductTruncation pins the go-opus MULT16_32_Q15 semantics:
// each tap product is truncated to Q15 BEFORE it is added, not the final sum. Two
// taps of 1 over samples of 0x4000 give products of 16384, each of which
// truncates to 0 (16384 >> 15 = 0), so every output is 0. A build that summed the
// full products (32768) and truncated once would get 32768 >> 15 = 1 per output.
// The sample lengths force the Go path, the NEON vector body and the AVX2 vector
// body plus scalar tails.
func TestFIRValidQ15_PerProductTruncation(t *testing.T) {
	taps := []int16{1, 1}
	for _, xl := range []int{2, 5, 12, 20} {
		x := make([]int32, xl)
		for i := range x {
			x[i] = 0x4000
		}
		outLen := xl - 1
		dst := make([]int32, outLen)
		orc := make([]int32, outLen)
		FIRValidQ15(dst, x, taps)
		firValidQ15Oracle(orc, x, taps)
		for i := range dst {
			if dst[i] != 0 {
				t.Fatalf("FIRValidQ15 per-product truncation xl=%d: dst[%d] = %d, want 0 (final-sum truncation gives 1)", xl, i, dst[i])
			}
			if orc[i] != 0 {
				t.Fatalf("oracle per-product truncation xl=%d: orc[%d] = %d, want 0", xl, i, orc[i])
			}
		}
	}
}

// TestFIRValidQ15_Wrap pins the wrapping (non-saturating) accumulate. taps of
// MinInt16 over samples of MinInt32 give a product of 2^46, whose Q15 truncation
// is 2^31, which wraps to MinInt32 per product; a saturating build would clamp to
// MaxInt32. A single such tap is used so the output is exactly the wrapped
// product. The length forces the vector body and the tail on both arches.
func TestFIRValidQ15_Wrap(t *testing.T) {
	taps := []int16{math.MinInt16}
	const xl = 20
	x := make([]int32, xl)
	for i := range x {
		x[i] = math.MinInt32
	}
	dst := make([]int32, xl) // outLen = xl - 1 + 1 = xl
	FIRValidQ15(dst, x, taps)
	orc := make([]int32, xl)
	firValidQ15Oracle(orc, x, taps)
	for i := range dst {
		if dst[i] != math.MinInt32 {
			t.Fatalf("FIRValidQ15 wrap: dst[%d] = %d, want %d (wrap, not saturate)", i, dst[i], int32(math.MinInt32))
		}
		if orc[i] != math.MinInt32 {
			t.Fatalf("oracle wrap: orc[%d] = %d, want %d", i, orc[i], int32(math.MinInt32))
		}
	}
}

// TestFIRValidQ15_Guards exercises the CRITICAL guard: empty taps or an x shorter
// than taps must produce no output (the valid-output count would otherwise
// underflow), leaving dst untouched and never panicking. len(x) == len(taps)
// produces exactly one output.
func TestFIRValidQ15_Guards(t *testing.T) {
	const sentinel = int32(0x0BADF00D)
	mkDst := func(n int) []int32 {
		d := make([]int32, n)
		for i := range d {
			d[i] = sentinel
		}
		return d
	}
	untouched := func(name string, d []int32, from int) {
		for i := from; i < len(d); i++ {
			if d[i] != sentinel {
				t.Fatalf("%s: dst[%d] = %d, want untouched sentinel", name, i, d[i])
			}
		}
	}

	// Empty taps (nil and empty slice): no output, dst fully untouched, no panic.
	d := mkDst(3)
	FIRValidQ15(d, []int32{1, 2, 3}, nil)
	untouched("nil taps", d, 0)
	FIRValidQ15(d, []int32{1, 2, 3}, []int16{})
	untouched("empty taps", d, 0)

	// x shorter than taps: no output, dst untouched, no panic.
	d = mkDst(3)
	FIRValidQ15(d, []int32{1, 2}, []int16{1, 2, 3})
	untouched("short x", d, 0)
	FIRValidQ15(d, nil, []int16{1})
	untouched("nil x", d, 0)

	// nil everything: no panic, nothing to check.
	FIRValidQ15(nil, nil, nil)

	// Empty dst with valid taps and x: exercises the n == 0 early return (len(dst)
	// clamps n to 0) in both the public wrapper and the self-guarding reference. No
	// panic, nothing written.
	FIRValidQ15([]int32{}, []int32{1, 2, 3}, []int16{1})
	firValidQ15Go([]int32{}, []int32{1, 2, 3}, []int16{1})

	// len(x) == len(taps): exactly one output; the rest of dst stays untouched.
	x := []int32{2, 3, 4}
	taps := []int16{1 << 14, 1 << 13, 1 << 12}
	d = mkDst(4)
	FIRValidQ15(d, x, taps)
	orc := make([]int32, 1)
	firValidQ15Oracle(orc, x, taps)
	if d[0] != orc[0] {
		t.Fatalf("len(x)==len(taps): dst[0] = %d, want %d (exactly one output)", d[0], orc[0])
	}
	untouched("single output", d, 1)
}

// TestFIRValidQ15_Clamp covers outLen > len(dst): only len(dst) outputs are
// written and they match the oracle over the same inputs; nothing past len(dst)
// is computed. The clamp length forces both the vector body and a scalar tail.
func TestFIRValidQ15_Clamp(t *testing.T) {
	x := genI32(40, 51)
	taps := genI16(5, 52)
	const shortN = 20 // fewer than the 36 available outputs
	short := make([]int32, shortN)
	FIRValidQ15(short, x, taps)

	fullOut := len(x) - len(taps) + 1
	ref := make([]int32, fullOut)
	firValidQ15Oracle(ref, x, taps)
	for i := range short {
		if short[i] != ref[i] {
			t.Fatalf("FIRValidQ15 clamp: dst[%d] = %d, want %d", i, short[i], ref[i])
		}
	}
}

// TestFIRValidQ15_TailUntouched plants sentinels past the output count and
// confirms the scalar-output tail stops exactly at n. outLen = 26 leaves a
// 2-output tail on both the 8-wide (AVX2) and 4-wide (NEON) bodies.
func TestFIRValidQ15_TailUntouched(t *testing.T) {
	x := genI32(30, 61)
	taps := genI16(5, 62)
	outLen := len(x) - len(taps) + 1 // 26
	dst := make([]int32, outLen+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // non-zero sentinel
	}
	FIRValidQ15(dst[:outLen], x, taps)
	for i := outLen; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("FIRValidQ15 wrote past outLen at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestFIRValidQ15_Unaligned holds x, taps and dst at mutually different element
// offsets so neither an aligned-load nor an aligned-store substitution can
// survive, across lengths that straddle a vector block and its tail on both
// arches.
func TestFIRValidQ15_Unaligned(t *testing.T) {
	const span = 400
	baseX := genI32(span, 71)
	baseTaps := genI16(60, 72)
	backing := make([]int32, span)
	for _, kl := range []int{1, 3, 5, 8} {
		for _, xl := range []int{8, 9, 16, 17, 33, 64} {
			outLen := xl - kl + 1
			for off := range 8 {
				x := baseX[off+1 : off+1+xl]
				taps := baseTaps[off+2 : off+2+kl]
				dst := backing[off+3 : off+3+outLen]
				FIRValidQ15(dst, x, taps)
				ref := make([]int32, outLen)
				firValidQ15Oracle(ref, x, taps)
				for i := range dst {
					if dst[i] != ref[i] {
						t.Fatalf("FIRValidQ15 unaligned kl=%d xl=%d off=%d: dst[%d] = %d, want %d", kl, xl, off, i, dst[i], ref[i])
					}
				}
			}
		}
	}
}

// TestFIRValidQ15_AllocFree declares the buffers INSIDE the measured closure so
// only allocations forced by the call itself are counted. 5 taps is the comb
// filter case.
func TestFIRValidQ15_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var x [1000]int32
		var taps [5]int16
		var dst [996]int32
		FIRValidQ15(dst[:], x[:], taps[:])
	}); n != 0 {
		t.Errorf("FIRValidQ15 forces %v caller allocations per run, want 0", n)
	}
}

// firSeeds seeds raw byte buffers for x and taps whose element counts bracket the
// 4/8-lane block boundaries and the guard (taps longer than x, empty taps).
func firSeeds(f *testing.F) {
	f.Helper()
	xLens := []int{0, 1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 32, 33, 40, 64}
	tapLens := []int{0, 1, 2, 3, 5, 8, 16}
	for _, xl := range xLens {
		for _, tl := range tapLens {
			xRaw := make([]byte, xl*4)
			for i := range xRaw {
				xRaw[i] = byte(i*37 + 11)
			}
			tRaw := make([]byte, tl*2)
			for i := range tRaw {
				tRaw[i] = byte(i*53 + 7)
			}
			f.Add(xRaw, tRaw)
		}
	}
}

// FuzzFIRValidQ15 differentially fuzzes the dispatched FIRValidQ15 against the
// pure-Go reference and the independent oracle over arbitrary int32 samples and
// int16 taps, so tail handling, the guard and the wrap are explored past the
// hand-picked seeds. The guard path (empty taps or short x) is checked to write
// nothing.
func FuzzFIRValidQ15(f *testing.F) {
	firSeeds(f)
	f.Fuzz(func(t *testing.T, xRaw, tapsRaw []byte) {
		x := i32sFromBits(xRaw)
		taps := i16sFromBits(tapsRaw)

		if len(taps) == 0 || len(x) < len(taps) {
			dst := []int32{7, 7, 7}
			FIRValidQ15(dst, x, taps)
			for i, v := range dst {
				if v != 7 {
					t.Fatalf("guard path wrote dst[%d] = %d (len(x)=%d len(taps)=%d)", i, v, len(x), len(taps))
				}
			}
			return
		}

		outLen := len(x) - len(taps) + 1
		got := make([]int32, outLen)
		want := make([]int32, outLen)
		orc := make([]int32, outLen)
		FIRValidQ15(got, x, taps)
		firValidQ15Go(want, x, taps)
		firValidQ15Oracle(orc, x, taps)
		equalI32(t, "FIRValidQ15/ref", got, want)
		equalI32(t, "FIRValidQ15/oracle", got, orc)
	})
}
