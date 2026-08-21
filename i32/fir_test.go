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

// Tests for FIRSymValidQ15, the int32 valid convolution with a SYMMETRIC Q15 tap
// set (a center tap plus K mirror pairs). The load-bearing behavior distinct from
// FIRValidQ15 is the mirror FOLD: each pair's two mirror samples x[c-k]+x[c+k] are
// summed with a wrapping int32 add BEFORE a SINGLE Q15 truncation, so a pair
// contributes one truncation, not two. This is what makes it bit-exact with
// libopus comb_filter_const_c; trunc((p+q)>>15) != trunc(p>>15)+trunc(q>>15) in
// general, so folding a symmetric tap set through FIRValidQ15 would diverge.

// firSymValidQ15Oracle computes the symmetric valid convolution independently of
// firSymValidQ15Go on two axes, exactly as firValidQ15Oracle pins FIRValidQ15. The
// Q15 shift is a manual floor division by 2^15 (not the >> operator the reference
// uses), and the truncated int32 products are accumulated as an exact int64 sum
// truncated to int32 once at the end (not the per-add int32 wrap the reference
// performs). The mirror fold x[c-k]+x[c+k] is a wrapping int32 add computed BEFORE
// the single truncation, matching the reference and the codec. It pins the
// reference rather than trusting a formula shared with it.
func firSymValidQ15Oracle(dst, x []int32, center int16, pairs []int16) {
	k := len(pairs)
	if len(x) < 2*k+1 {
		return
	}
	outLen := len(x) - 2*k
	n := min(len(dst), outLen)
	const q = int64(1) << 15
	floorShift := func(prod int64) int64 {
		// Floor-divide by 2^15 (truncation toward -inf), computed without >>.
		shifted := prod / q
		if prod%q != 0 && prod < 0 {
			shifted--
		}
		return shifted
	}
	for i := range n {
		c := i + k
		var acc int64 // exact sum of the truncated int32 products
		acc += int64(int32(floorShift(int64(center) * int64(x[c]))))
		for p := 1; p <= k; p++ {
			sum := x[c-p] + x[c+p] // wrapping int32 add BEFORE truncation
			acc += int64(int32(floorShift(int64(pairs[p-1]) * int64(sum))))
		}
		dst[i] = int32(acc) // single wrap of the exact sum == the per-add wrapping accumulate
	}
}

// TestFIRSymValidQ15OracleSelfCheck confirms the oracle folds each mirror pair
// BEFORE truncating and floors toward -inf, so the parity tests rest on a checked
// foundation. center=-1 over a single sample floors below zero; a pair of 0x4000
// mirror samples folds to 0x8000 and truncates to 1, whereas truncating each
// mirror separately (0x4000>>15 = 0) would give 0.
func TestFIRSymValidQ15OracleSelfCheck(t *testing.T) {
	// K=0, center=-1 over x=[1]: product = -1, floor(-1/2^15) = -1, not 0.
	dst := make([]int32, 1)
	firSymValidQ15Oracle(dst, []int32{1}, -1, nil)
	if dst[0] != -1 {
		t.Fatalf("oracle floor: got %d, want -1 (truncation toward -inf)", dst[0])
	}
	// K=1, center=0, pairs=[1] over x=[0x4000, 5, 0x4000]: fold 0x4000+0x4000 =
	// 0x8000, *1 >>15 = 1. Per-mirror truncation would give 0.
	firSymValidQ15Oracle(dst, []int32{0x4000, 5, 0x4000}, 0, []int16{1})
	if dst[0] != 1 {
		t.Fatalf("oracle fold: got %d, want 1 (fold-then-truncate)", dst[0])
	}
}

// TestFIRSymValidQ15 sweeps a grid of (len(x), K) against both the pure-Go
// reference and the independent oracle, so a fault cannot hide by agreeing with
// the reference alone. K=2 is the combFilterConst driver (a 5-tap symmetric
// window). The x lengths span the vector-output loop plus a scalar-output tail on
// both arches as well as short outputs that route to the Go path. MinInt32/
// MaxInt32 ride the ends of x and MinInt16/MaxInt16 the center and pair extremes.
func TestFIRSymValidQ15(t *testing.T) {
	pairCounts := []int{0, 1, 2, 3, 5, 8}
	xLens := []int{1, 2, 3, 4, 5, 8, 9, 11, 16, 17, 20, 24, 31, 32, 33, 40, 64, 65, 100, 128}
	for _, k := range pairCounts {
		center := int16(math.MinInt16)
		pairs := genI16(k, uint32(k)*211+13)
		if k > 0 {
			pairs[0] = math.MaxInt16
			pairs[k-1] = math.MinInt16
		}
		for _, xl := range xLens {
			x := genI32(xl, uint32(xl)*29+uint32(k))
			x[0] = math.MinInt32
			x[xl-1] = math.MaxInt32

			outLen := 0
			if xl >= 2*k+1 {
				outLen = xl - 2*k
			}
			dst := make([]int32, outLen)
			ref := make([]int32, outLen)
			orc := make([]int32, outLen)
			FIRSymValidQ15(dst, x, center, pairs)
			firSymValidQ15Go(ref, x, center, pairs)
			firSymValidQ15Oracle(orc, x, center, pairs)
			for i := range dst {
				if dst[i] != ref[i] {
					t.Fatalf("FIRSymValidQ15 k=%d xl=%d: dst[%d] = %d, want %d (reference)", k, xl, i, dst[i], ref[i])
				}
				if dst[i] != orc[i] {
					t.Fatalf("FIRSymValidQ15 k=%d xl=%d: dst[%d] = %d, want %d (oracle)", k, xl, i, dst[i], orc[i])
				}
			}
		}
	}
}

// TestFIRSymValidQ15_SingleTruncationPerPair pins the codec fold semantics: each
// mirror pair is summed and truncated ONCE, not truncated per mirror sample. With
// center 0 and one pair of 1 over mirror samples of 0x4000, the fold is 0x8000,
// truncating to 1; a build that truncated each mirror (0x4000>>15 = 0) would get
// 0. The sample lengths force the Go path, the NEON vector body and the AVX2
// vector body plus scalar tails.
func TestFIRSymValidQ15_SingleTruncationPerPair(t *testing.T) {
	center := int16(0)
	pairs := []int16{1}
	for _, xl := range []int{3, 6, 13, 21} {
		x := make([]int32, xl)
		for i := range x {
			x[i] = 0x4000
		}
		outLen := xl - 2
		dst := make([]int32, outLen)
		orc := make([]int32, outLen)
		FIRSymValidQ15(dst, x, center, pairs)
		firSymValidQ15Oracle(orc, x, center, pairs)
		for i := range dst {
			if dst[i] != 1 {
				t.Fatalf("FIRSymValidQ15 fold xl=%d: dst[%d] = %d, want 1 (per-mirror truncation gives 0)", xl, i, dst[i])
			}
			if orc[i] != 1 {
				t.Fatalf("oracle fold xl=%d: orc[%d] = %d, want 1", xl, i, orc[i])
			}
		}
	}
}

// TestFIRSymValidQ15_Wrap pins both wraps: the wrapping int32 fold of the two
// mirror samples BEFORE truncation, and the wrapping (non-saturating) accumulate.
// Two mirror samples of MaxInt32 fold to -2 in int32 (0x7FFFFFFF+0x7FFFFFFF wraps),
// not to a saturated 2^32-2, so a pair coeff of MinInt16 over them gives
// int32(int64(-32768)*int64(-2) >> 15) = 2; a build that widened the addends
// before summing, or that saturated, would diverge. The length forces the vector
// body and the tail on both arches.
func TestFIRSymValidQ15_Wrap(t *testing.T) {
	center := int16(0)
	pairs := []int16{math.MinInt16}
	const xl = 22
	x := make([]int32, xl)
	for i := range x {
		x[i] = math.MaxInt32
	}
	outLen := xl - 2 // 20
	dst := make([]int32, outLen)
	orc := make([]int32, outLen)
	FIRSymValidQ15(dst, x, center, pairs)
	firSymValidQ15Oracle(orc, x, center, pairs)
	// fold = int32(MaxInt32+MaxInt32) = -2; product = -32768 * -2 = 65536;
	// 65536 >> 15 = 2.
	const want = int32(2)
	for i := range dst {
		if dst[i] != want {
			t.Fatalf("FIRSymValidQ15 wrap: dst[%d] = %d, want %d (int32 fold before truncate)", i, dst[i], want)
		}
		if orc[i] != want {
			t.Fatalf("oracle wrap: orc[%d] = %d, want %d", i, orc[i], want)
		}
	}
}

// TestFIRSymValidQ15_Guards exercises the guard: an x shorter than the 2K+1 window
// must produce no output, leaving dst untouched and never panicking. len(x) ==
// 2K+1 produces exactly one output.
func TestFIRSymValidQ15_Guards(t *testing.T) {
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

	// x shorter than the 2K+1 window: no output, dst untouched, no panic.
	d := mkDst(3)
	FIRSymValidQ15(d, []int32{1, 2}, 100, []int16{1}) // window 3 > len 2
	untouched("short x", d, 0)
	FIRSymValidQ15(d, nil, 100, []int16{1})
	untouched("nil x", d, 0)
	FIRSymValidQ15(d, []int32{1, 2, 3, 4}, 100, []int16{1, 2}) // window 5 > len 4
	untouched("short x k=2", d, 0)

	// nil everything: no panic.
	FIRSymValidQ15(nil, nil, 0, nil)

	// Empty dst with a valid window: exercises the n == 0 early return in both the
	// wrapper and the self-guarding reference. No panic, nothing written.
	FIRSymValidQ15([]int32{}, []int32{1, 2, 3}, 100, []int16{1})
	firSymValidQ15Go([]int32{}, []int32{1, 2, 3}, 100, []int16{1})

	// len(x) == 2K+1: exactly one output; the rest of dst stays untouched.
	x := []int32{2, 3, 4}
	d = mkDst(4)
	FIRSymValidQ15(d, x, 1<<14, []int16{1 << 13})
	orc := make([]int32, 1)
	firSymValidQ15Oracle(orc, x, 1<<14, []int16{1 << 13})
	if d[0] != orc[0] {
		t.Fatalf("len(x)==2K+1: dst[0] = %d, want %d (exactly one output)", d[0], orc[0])
	}
	untouched("single output", d, 1)
}

// TestFIRSymValidQ15_K0 covers the K=0 degenerate case: with no pairs the kernel is
// a pure center scale over the whole window (outLen = len(x)), which must equal
// scaleQ15Go with k = center. The length forces the vector body and a scalar tail.
func TestFIRSymValidQ15_K0(t *testing.T) {
	const center = int16(0x2000)
	for _, xl := range []int{1, 4, 8, 11, 20, 33} {
		x := genI32(xl, uint32(xl)*7+3)
		if xl >= 2 {
			x[0] = math.MinInt32
			x[xl-1] = math.MaxInt32
		}
		got := make([]int32, xl)
		want := make([]int32, xl)
		FIRSymValidQ15(got, x, center, nil)
		scaleQ15Go(want, x, center)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("FIRSymValidQ15 K=0 xl=%d: dst[%d] = %d, want %d (ScaleQ15)", xl, i, got[i], want[i])
			}
		}
	}
}

// TestFIRSymValidQ15_Clamp covers outLen > len(dst): only len(dst) outputs are
// written and they match the oracle over the same inputs. The clamp length forces
// both the vector body and a scalar tail.
func TestFIRSymValidQ15_Clamp(t *testing.T) {
	x := genI32(40, 51)
	center := int16(0x1234)
	pairs := genI16(2, 52)
	const shortN = 20 // fewer than the 36 available outputs
	short := make([]int32, shortN)
	FIRSymValidQ15(short, x, center, pairs)

	fullOut := len(x) - 2*len(pairs)
	ref := make([]int32, fullOut)
	firSymValidQ15Oracle(ref, x, center, pairs)
	for i := range short {
		if short[i] != ref[i] {
			t.Fatalf("FIRSymValidQ15 clamp: dst[%d] = %d, want %d", i, short[i], ref[i])
		}
	}
}

// TestFIRSymValidQ15_TailUntouched plants sentinels past the output count and
// confirms the scalar-output tail stops exactly at n. outLen = 26 leaves a
// 2-output tail on both the 8-wide (AVX2) and 4-wide (NEON) bodies.
func TestFIRSymValidQ15_TailUntouched(t *testing.T) {
	x := genI32(30, 61)
	center := int16(0x0777)
	pairs := genI16(2, 62)
	outLen := len(x) - 2*len(pairs) // 26
	dst := make([]int32, outLen+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // non-zero sentinel
	}
	FIRSymValidQ15(dst[:outLen], x, center, pairs)
	for i := outLen; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("FIRSymValidQ15 wrote past outLen at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestFIRSymValidQ15_Unaligned holds x, pairs and dst at mutually different element
// offsets so neither an aligned-load nor an aligned-store substitution can survive,
// across lengths that straddle a vector block and its tail on both arches.
func TestFIRSymValidQ15_Unaligned(t *testing.T) {
	const span = 400
	baseX := genI32(span, 71)
	basePairs := genI16(60, 72)
	backing := make([]int32, span)
	for _, k := range []int{0, 1, 2, 3} {
		for _, xl := range []int{8, 9, 16, 17, 33, 64} {
			outLen := xl - 2*k
			for off := range 8 {
				x := baseX[off+1 : off+1+xl]
				pairs := basePairs[off+2 : off+2+k]
				center := basePairs[off]
				dst := backing[off+3 : off+3+outLen]
				FIRSymValidQ15(dst, x, center, pairs)
				ref := make([]int32, outLen)
				firSymValidQ15Oracle(ref, x, center, pairs)
				for i := range dst {
					if dst[i] != ref[i] {
						t.Fatalf("FIRSymValidQ15 unaligned k=%d xl=%d off=%d: dst[%d] = %d, want %d", k, xl, off, i, dst[i], ref[i])
					}
				}
			}
		}
	}
}

// TestFIRSymValidQ15_AllocFree declares the buffers INSIDE the measured closure so
// only allocations forced by the call itself are counted. K=2 is the comb filter
// case (a 5-tap symmetric window).
func TestFIRSymValidQ15_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var x [1000]int32
		var pairs [2]int16
		var dst [996]int32
		FIRSymValidQ15(dst[:], x[:], 0x4000, pairs[:])
	}); n != 0 {
		t.Errorf("FIRSymValidQ15 forces %v caller allocations per run, want 0", n)
	}
}

// FuzzFIRSymValidQ15 differentially fuzzes the dispatched FIRSymValidQ15 against
// the pure-Go reference and the independent oracle over arbitrary int32 samples,
// int16 center and int16 pairs, so the fold, the tail, the guard and the wrap are
// explored past the hand-picked seeds. The first int16 of the tap stream is the
// center; the rest are the pairs. The guard path (x shorter than the 2K+1 window)
// is checked to write nothing.
func FuzzFIRSymValidQ15(f *testing.F) {
	firSeeds(f)
	f.Fuzz(func(t *testing.T, xRaw, tapsRaw []byte) {
		x := i32sFromBits(xRaw)
		coeffs := i16sFromBits(tapsRaw)
		// Split the tap stream into a center plus pairs; an empty stream means K=0
		// with a zero center.
		var center int16
		var pairs []int16
		if len(coeffs) > 0 {
			center = coeffs[0]
			pairs = coeffs[1:]
		}
		k := len(pairs)

		if len(x) < 2*k+1 {
			dst := []int32{7, 7, 7}
			FIRSymValidQ15(dst, x, center, pairs)
			for i, v := range dst {
				if v != 7 {
					t.Fatalf("guard path wrote dst[%d] = %d (len(x)=%d K=%d)", i, v, len(x), k)
				}
			}
			return
		}

		outLen := len(x) - 2*k
		got := make([]int32, outLen)
		want := make([]int32, outLen)
		orc := make([]int32, outLen)
		FIRSymValidQ15(got, x, center, pairs)
		firSymValidQ15Go(want, x, center, pairs)
		firSymValidQ15Oracle(orc, x, center, pairs)
		equalI32(t, "FIRSymValidQ15/ref", got, want)
		equalI32(t, "FIRSymValidQ15/oracle", got, orc)
	})
}
