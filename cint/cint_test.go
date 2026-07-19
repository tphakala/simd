package cint

import (
	"encoding/binary"
	"math"
	"math/big"
	"testing"
)

// Tests for the fixed-point complex arithmetic kernels. The load-bearing
// contract is the TRUNCATING Q15 multiply S_MUL (an arithmetic right shift, no
// rounding constant) with int32 wraparound on every add and subtract, so the
// critical cases are (1) truncation vs rounding at small products and (2) the
// MinInt32 / MinInt16 wraps where a saturating build would diverge. Every SIMD
// path is validated bit-for-bit against the pure-Go reference AND an independent
// arbitrary-precision oracle, so a fault cannot hide by agreeing with the
// reference alone.

// genI32 produces a deterministic spread across the full int32 range from a cheap
// LCG, so sign and high bits are exercised at every index.
func genI32(n int, seed uint32) []int32 {
	s := make([]int32, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int32(x)
	}
	return s
}

// genI16 produces a deterministic spread across the full int16 range for the Q15
// twiddle, using the high bits of the same LCG so the low-order patterns differ
// from genI32.
func genI16(n int, seed uint32) []int16 {
	s := make([]int16, n)
	x := seed*40503 + 12345
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int16(x >> 16)
	}
	return s
}

// sMulBig computes S_MUL(x, c) in arbitrary precision, fully independent of the
// int64 arithmetic in sMul and the SIMD kernels. big.Int.Rsh is a floor
// (arithmetic) shift, matching Go's signed >>, and the final int32() drops the
// value back to a 32-bit lane exactly as a store does, so 2^31 wraps to MinInt32.
func sMulBig(x int32, c int16) int32 {
	p := new(big.Int).Mul(big.NewInt(int64(x)), big.NewInt(int64(c)))
	p.Rsh(p, sMulShift) // floor shift, matches Go's arithmetic >>
	return int32(p.Int64())
}

// mulOracle and mulConjOracle compute one complex product in arbitrary precision.
// The two S_MULs are truncated to int32 first, then combined with wrapping int32
// arithmetic (Go int32 add/sub wrap), so the oracle reproduces the kernel's exact
// two-step contract.
func mulOracle(ar, ai int32, br, bi int16) (re, im int32) {
	return sMulBig(ar, br) - sMulBig(ai, bi), sMulBig(ar, bi) + sMulBig(ai, br)
}

func mulConjOracle(ar, ai int32, br, bi int16) (re, im int32) {
	return sMulBig(ar, br) + sMulBig(ai, bi), sMulBig(ai, br) - sMulBig(ar, bi)
}

// evenLengths sweeps whole-complex-pair int32 lengths: every even length 0..80
// (0..40 complex), covering each vector-block/scalar-tail split of the 4-complex
// (8 int32) AVX2/NEON blocks, plus a spread of larger ones with non-block tails.
var evenLengths = func() []int {
	lens := make([]int, 0, 60)
	for n := 0; n <= 80; n += 2 {
		lens = append(lens, n)
	}
	return append(lens, 128, 200, 256, 258, 512, 1000, 1024, 2048)
}()

// checkMul verifies got against the Go reference and the oracle over n int32 lanes
// (n/2 complex). conj selects the conjugated combine.
func checkMul(t *testing.T, name string, conj bool, got, a []int32, tw []int16) {
	t.Helper()
	n := len(got)
	ref := make([]int32, n)
	if conj {
		mulConjGo(ref, a, tw)
	} else {
		mulGo(ref, a, tw)
	}
	for k := 0; k+1 < n; k += 2 {
		var re, im int32
		if conj {
			re, im = mulConjOracle(a[k], a[k+1], tw[k], tw[k+1])
		} else {
			re, im = mulOracle(a[k], a[k+1], tw[k], tw[k+1])
		}
		if got[k] != ref[k] || got[k+1] != ref[k+1] {
			t.Fatalf("%s n=%d at complex %d: got (%d,%d), reference (%d,%d)", name, n, k/2, got[k], got[k+1], ref[k], ref[k+1])
		}
		if got[k] != re || got[k+1] != im {
			t.Fatalf("%s n=%d at complex %d: got (%d,%d), oracle (%d,%d)", name, n, k/2, got[k], got[k+1], re, im)
		}
	}
}

// TestSMulTruncation pins the truncating contract in isolation, the single most
// important behavior: S_MUL must arithmetically shift with NO rounding constant.
// Each case names a product whose >>15 truncation differs from a +(1<<14) rounding
// build, plus the MinInt32/MinInt16 wraps a saturating build gets wrong.
func TestSMulTruncation(t *testing.T) {
	cases := []struct {
		x    int32
		c    int16
		want int32
	}{
		{32767, 1, 0},              // 32767/32768 truncates to 0 (rounding -> 1)
		{1, 16384, 0},              // 0.5 in Q15 truncates to 0 (rounding -> 1)
		{3, 16384, 1},              // 1.5 truncates to 1 (rounding -> 2)
		{-1, 16384, -1},            // -0.5 floors to -1 (rounding -> 0)
		{-3, 1, -1},                // -3/32768 floors to -1
		{32768, 1, 1},              // exactly 1
		{math.MinInt32, 1, -65536}, // -2^31 >> 15 = -2^16
		{math.MaxInt32, 1, 65535},  // (2^31-1) >> 15
		{math.MinInt32, math.MinInt16, math.MinInt32},     // 2^46 >> 15 = 2^31 -> wraps to MinInt32
		{math.MinInt32, math.MaxInt16, -2147418112},       // -2^31 * (2^15-1) >> 15
		{math.MaxInt32, math.MinInt16, math.MinInt32 + 1}, // -(2^46-2^15) >> 15 = -(2^31-1)
	}
	for _, c := range cases {
		if got := sMul(c.x, c.c); got != c.want {
			t.Errorf("sMul(%d,%d) = %d, want %d", c.x, c.c, got, c.want)
		}
		if got := sMulBig(c.x, c.c); got != c.want {
			t.Errorf("sMulBig(%d,%d) = %d, want %d (oracle disagrees)", c.x, c.c, got, c.want)
		}
	}
}

// TestMul sweeps every whole-pair length against the reference and the oracle.
// MinInt32/MaxInt32 ride the head and tail complex of a, and MinInt16/MaxInt16 the
// head and tail of tw, so the wraps run at every length at both ends.
func TestMul(t *testing.T) {
	for _, n := range evenLengths {
		a := genI32(n, 11)
		tw := genI16(n, 12)
		plantExtremes(a, tw)
		dst := make([]int32, n)
		Mul(dst, a, tw)
		checkMul(t, "Mul", false, dst, a, tw)

		dst2 := make([]int32, n)
		MulConj(dst2, a, tw)
		checkMul(t, "MulConj", true, dst2, a, tw)
	}
}

// plantExtremes seeds the head and tail complex with the wrap-driving extremes.
func plantExtremes(a []int32, tw []int16) {
	// Each slice is indexed by its own length so a shorter tw never panics.
	if len(a) >= 2 {
		a[0], a[1] = math.MinInt32, math.MaxInt32
		a[len(a)-2], a[len(a)-1] = math.MaxInt32, math.MinInt32
	}
	if len(tw) >= 2 {
		tw[0], tw[1] = math.MinInt16, math.MaxInt16
		tw[len(tw)-2], tw[len(tw)-1] = math.MaxInt16, math.MinInt16
	}
}

// TestMulValueMatrix plants a load-bearing complex pair (data and twiddle) at
// every complex position across a length that spans a vector block plus a scalar
// tail on both arches, so a lane error, an index error and a recombine the even/odd
// blend mishandles are all caught wherever they live. The AVX2 VPMULDQ even/odd
// split makes the per-position sweep essential.
func TestMulValueMatrix(t *testing.T) {
	dataPlant := [][2]int32{
		{math.MinInt32, math.MaxInt32}, {math.MaxInt32, math.MinInt32},
		{math.MinInt32, math.MinInt32}, {-1, 1}, {0x12345678, -0x0BADF00D},
		{1, -1}, {0, 0},
	}
	twPlant := [][2]int16{
		{math.MinInt16, math.MaxInt16}, {math.MaxInt16, math.MinInt16},
		{math.MinInt16, math.MinInt16}, {-1, 1}, {0x4000, -0x4000}, {1, -1},
	}
	const nc = 6 // 6 complex = 12 int32: one 8-int32 block + 2-complex tail
	const n = nc * 2
	fillerA := genI32(n, 21)
	fillerTw := genI16(n, 22)
	for _, dp := range dataPlant {
		for _, tp := range twPlant {
			for pos := range nc {
				a := append([]int32(nil), fillerA...)
				tw := append([]int16(nil), fillerTw...)
				a[2*pos], a[2*pos+1] = dp[0], dp[1]
				tw[2*pos], tw[2*pos+1] = tp[0], tp[1]
				dst := make([]int32, n)
				Mul(dst, a, tw)
				checkMul(t, "Mul", false, dst, a, tw)
				MulConj(dst, a, tw)
				checkMul(t, "MulConj", true, dst, a, tw)
			}
		}
	}
}

// TestAddSub sweeps the wrapping complex add and subtract against the reference,
// with a MinInt32+overflow head and a MaxInt32+overflow tail so both wraps run at
// every length.
func TestAddSub(t *testing.T) {
	for _, n := range evenLengths {
		a := genI32(n, 31)
		b := genI32(n, 32)
		if n >= 2 {
			a[0], b[0] = math.MinInt32, -1    // sum underflows, diff = MinInt32+1
			a[n-1], b[n-1] = math.MaxInt32, 1 // sum overflows
		}
		dst := make([]int32, n)
		ref := make([]int32, n)

		Add(dst, a, b)
		addGo(ref, a, b)
		for i := range dst {
			want := int32(int64(a[i]) + int64(b[i]))
			if dst[i] != ref[i] || dst[i] != want {
				t.Fatalf("Add n=%d at %d: got %d, ref %d, oracle %d", n, i, dst[i], ref[i], want)
			}
		}

		Sub(dst, a, b)
		subGo(ref, a, b)
		for i := range dst {
			want := int32(int64(a[i]) - int64(b[i]))
			if dst[i] != ref[i] || dst[i] != want {
				t.Fatalf("Sub n=%d at %d: got %d, ref %d, oracle %d", n, i, dst[i], ref[i], want)
			}
		}
	}
}

// TestMulByScalar sweeps the in-place Q15 scale against the reference and oracle,
// planting MinInt32 at index 0 and MaxInt32 at the last index so the wrap runs at
// every length under a MinInt16 coefficient.
func TestMulByScalar(t *testing.T) {
	ss := []int16{math.MinInt16, math.MaxInt16, 1, -1, 0x4000, 0x1234}
	for _, n := range evenLengths {
		for _, s := range ss {
			a := genI32(n, 41)
			if n >= 2 {
				a[0] = math.MinInt32
				a[n-1] = math.MaxInt32
			}
			orig := append([]int32(nil), a...)
			MulByScalar(a, s)
			for i := range a {
				want := sMulBig(orig[i], s)
				if a[i] != want {
					t.Fatalf("MulByScalar n=%d s=%d at %d: got %d, want %d (oracle)", n, s, i, a[i], want)
				}
				if ref := sMul(orig[i], s); a[i] != ref {
					t.Fatalf("MulByScalar n=%d s=%d at %d: got %d, want %d (reference)", n, s, i, a[i], ref)
				}
			}
		}
	}
}

// TestOddLengthMasking confirms every operation masks the processed length to a
// whole number of complex pairs: a trailing lone real lane in an odd-length slice
// is left untouched and, for Mul/MulConj, its non-existent imaginary partner is
// never read (the slice is exactly odd-length, so a read of index len would be out
// of bounds).
func TestOddLengthMasking(t *testing.T) {
	for _, nc := range []int{1, 2, 4, 5, 8, 9, 13} {
		n := 2*nc + 1 // odd int32 length: nc whole complex + 1 lone real
		const sentinel = int32(0x0BADBEEF)

		a := genI32(n, 51)
		tw := genI16(n, 52)

		// Mul: dst last lane is the sentinel and must survive; the first n-1 lanes
		// must match the oracle over the nc whole pairs.
		dst := genI32(n, 53)
		dst[n-1] = sentinel
		Mul(dst, a, tw)
		for k := 0; k+1 < n-1; k += 2 {
			re, im := mulOracle(a[k], a[k+1], tw[k], tw[k+1])
			if dst[k] != re || dst[k+1] != im {
				t.Fatalf("Mul odd n=%d at complex %d: got (%d,%d), want (%d,%d)", n, k/2, dst[k], dst[k+1], re, im)
			}
		}
		if dst[n-1] != sentinel {
			t.Errorf("Mul odd n=%d wrote the trailing half-complex: dst[%d]=%d", n, n-1, dst[n-1])
		}

		dst = genI32(n, 54)
		dst[n-1] = sentinel
		MulConj(dst, a, tw)
		for k := 0; k+1 < n-1; k += 2 {
			re, im := mulConjOracle(a[k], a[k+1], tw[k], tw[k+1])
			if dst[k] != re || dst[k+1] != im {
				t.Fatalf("MulConj odd n=%d at complex %d: got (%d,%d), want (%d,%d)", n, k/2, dst[k], dst[k+1], re, im)
			}
		}
		if dst[n-1] != sentinel {
			t.Errorf("MulConj odd n=%d wrote the trailing half-complex: dst[%d]=%d", n, n-1, dst[n-1])
		}

		// Add/Sub: trailing lane untouched.
		b := genI32(n, 55)
		dst = make([]int32, n)
		dst[n-1] = sentinel
		Add(dst, a, b)
		if dst[n-1] != sentinel {
			t.Errorf("Add odd n=%d wrote the trailing half-complex", n)
		}
		dst[n-1] = sentinel
		Sub(dst, a, b)
		if dst[n-1] != sentinel {
			t.Errorf("Sub odd n=%d wrote the trailing half-complex", n)
		}

		// MulByScalar in place: trailing lane unscaled.
		sc := genI32(n, 56)
		sc[n-1] = sentinel
		MulByScalar(sc, 0x1234)
		if sc[n-1] != sentinel {
			t.Errorf("MulByScalar odd n=%d scaled the trailing half-complex: got %d", n, sc[n-1])
		}
	}
}

// TestInPlaceAlias runs Mul, MulConj and MulByScalar fully in place (dst aliases a
// exactly) and confirms every lane matches the oracle computed from the saved
// originals. Each SIMD block reads its whole input block into registers before
// storing, so the exact overlay is well defined. MinInt32 rides the head so the
// wrap is exercised in place.
func TestInPlaceAlias(t *testing.T) {
	for _, n := range []int{2, 4, 6, 8, 10, 14, 16, 18, 32, 34, 64, 128, 130} {
		a := genI32(n, 61)
		tw := genI16(n, 62)
		if n >= 2 {
			a[0], a[1] = math.MinInt32, math.MaxInt32
			tw[0], tw[1] = math.MinInt16, math.MaxInt16
		}

		orig := append([]int32(nil), a...)
		Mul(a, a, tw) // dst == a
		for k := 0; k+1 < n; k += 2 {
			re, im := mulOracle(orig[k], orig[k+1], tw[k], tw[k+1])
			if a[k] != re || a[k+1] != im {
				t.Fatalf("Mul in place n=%d at complex %d: got (%d,%d), want (%d,%d)", n, k/2, a[k], a[k+1], re, im)
			}
		}

		a2 := append([]int32(nil), orig...)
		MulConj(a2, a2, tw)
		for k := 0; k+1 < n; k += 2 {
			re, im := mulConjOracle(orig[k], orig[k+1], tw[k], tw[k+1])
			if a2[k] != re || a2[k+1] != im {
				t.Fatalf("MulConj in place n=%d at complex %d: got (%d,%d), want (%d,%d)", n, k/2, a2[k], a2[k+1], re, im)
			}
		}

		a3 := append([]int32(nil), orig...)
		MulByScalar(a3, -0x4000)
		for i := range a3 {
			if want := sMulBig(orig[i], -0x4000); a3[i] != want {
				t.Fatalf("MulByScalar in place n=%d at %d: got %d, want %d", n, i, a3[i], want)
			}
		}
	}
}

// TestEmpty confirms empty inputs write nothing and short/nil operands are a no-op
// that leaves the survivor untouched.
func TestEmpty(t *testing.T) {
	Mul(nil, nil, nil)
	MulConj(nil, nil, nil)
	Add(nil, nil, nil)
	Sub(nil, nil, nil)
	MulByScalar(nil, 5)

	one := []int32{42, 43}
	tw := []int16{}
	Mul(one, one, tw) // n = 0 after clamp
	if one[0] != 42 || one[1] != 43 {
		t.Errorf("Mul with empty tw modified dst: %v", one)
	}
	MulByScalar([]int32{7}, 5) // len 1 -> masked to 0
}

// TestClamp covers mismatched lengths: n is the shortest of the applicable slices,
// masked to even, and nothing past it is touched.
func TestClamp(t *testing.T) {
	a := genI32(40, 71)
	tw := genI16(30, 72) // tw shortest -> n = 30
	dst := make([]int32, 50)
	for i := range dst {
		dst[i] = -7 // sentinel
	}
	Mul(dst, a, tw)
	checkMul(t, "Mul", false, dst[:30], a[:30], tw)
	for i := 30; i < len(dst); i++ {
		if dst[i] != -7 {
			t.Fatalf("Mul wrote past the clamp at dst[%d] = %d", i, dst[i])
		}
	}

	// dst shortest (odd) -> n = 24 (25 &^ 1).
	short := make([]int32, 25)
	Mul(short, a, tw)
	checkMul(t, "Mul", false, short[:24], a[:24], tw[:24])
	if short[24] != 0 {
		t.Fatalf("Mul wrote the masked-off lane: short[24] = %d", short[24])
	}
}

// TestUnalignedOperands sweeps element offsets so neither dst, a nor tw is reliably
// aligned; an aligned-load or aligned-store substitution cannot survive.
func TestUnalignedOperands(t *testing.T) {
	const span = 400
	baseA := genI32(span, 81)
	baseTw := genI16(span, 82)
	backing := make([]int32, span)
	for _, n := range []int{8, 10, 16, 18, 34, 66, 130} {
		for off := range 8 {
			a := baseA[off+1 : off+1+n]
			tw := baseTw[off+2 : off+2+n]
			dst := backing[off+3 : off+3+n]
			Mul(dst, a, tw)
			checkMul(t, "Mul", false, dst, a, tw)
			MulConj(dst, a, tw)
			checkMul(t, "MulConj", true, dst, a, tw)
		}
	}
}

// TestAllocFree declares the buffers INSIDE the measured closure so only
// allocations forced by the operation itself are counted. The dispatch is a direct
// CPU-flag switch precisely so //go:noescape holds and no slice header escapes.
func TestAllocFree(t *testing.T) {
	check := func(name string, f func()) {
		if got := testing.AllocsPerRun(50, f); got != 0 {
			t.Errorf("%s forces %v allocations per run, want 0", name, got)
		}
	}
	check("Mul", func() {
		var a, dst [1024]int32
		var tw [1024]int16
		Mul(dst[:], a[:], tw[:])
	})
	check("MulConj", func() {
		var a, dst [1024]int32
		var tw [1024]int16
		MulConj(dst[:], a[:], tw[:])
	})
	check("MulByScalar", func() {
		var a [1024]int32
		MulByScalar(a[:], 0x1234)
	})
	check("Add", func() {
		var a, b, dst [1024]int32
		Add(dst[:], a[:], b[:])
	})
	check("Sub", func() {
		var a, b, dst [1024]int32
		Sub(dst[:], a[:], b[:])
	})
}

// --- Differential fuzzing over data + twiddle bytes ---

// i32sFromBits reinterprets raw bytes as little-endian int32s, reaching the full
// int32 range including MinInt32/MaxInt32.
func i32sFromBits(raw []byte) []int32 {
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// i16sFromBits reinterprets raw bytes as little-endian int16s for the Q15 twiddle.
func i16sFromBits(raw []byte) []int16 {
	out := make([]int16, len(raw)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return out
}

func cintLenSeeds(f *testing.F) {
	f.Helper()
	ns := []int{0, 1, 2, 3, 4, 6, 8, 9, 10, 14, 16, 17, 18, 22, 30, 32, 33, 34, 64, 65, 66, 130}
	for _, n := range ns {
		da := make([]byte, n*4)
		db := make([]byte, n*4)
		tb := make([]byte, n*2)
		for i := range da {
			da[i] = byte(i*37 + 11)
		}
		for i := range db {
			db[i] = byte(i*29 + 3)
		}
		for i := range tb {
			tb[i] = byte(i*53 + 7)
		}
		f.Add(da, db, tb, int16(0x1234))
	}
}

// FuzzCint differentially fuzzes every operation against its pure-Go reference and,
// for the multiplies, the arbitrary-precision oracle, over arbitrary int32 data and
// int16 twiddle bytes. Tail handling and the wraps are explored far past the
// hand-picked seeds.
func FuzzCint(f *testing.F) {
	cintLenSeeds(f)
	f.Fuzz(func(t *testing.T, dataRaw, bRaw, twRaw []byte, s int16) {
		a := i32sFromBits(dataRaw)
		b := i32sFromBits(bRaw)
		tw := i16sFromBits(twRaw)
		n := min(len(a), len(b), len(tw))
		n &^= 1
		a = a[:n]
		b = b[:n]
		tw = tw[:n]

		dst := make([]int32, n)
		Mul(dst, a, tw)
		checkMul(t, "FuzzMul", false, dst, a, tw)
		MulConj(dst, a, tw)
		checkMul(t, "FuzzMulConj", true, dst, a, tw)

		ref := make([]int32, n)
		Add(dst, a, b)
		addGo(ref, a, b)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf("FuzzAdd at %d: got %d want %d (n=%d)", i, dst[i], ref[i], n)
			}
		}
		Sub(dst, a, b)
		subGo(ref, a, b)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf("FuzzSub at %d: got %d want %d (n=%d)", i, dst[i], ref[i], n)
			}
		}

		scale := append([]int32(nil), a...)
		MulByScalar(scale, s)
		for i := range scale {
			if want := sMulBig(a[i], s); scale[i] != want {
				t.Fatalf("FuzzMulByScalar at %d: got %d want %d (n=%d, s=%d)", i, scale[i], want, n, s)
			}
		}
	})
}
