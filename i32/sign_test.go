package i32

import (
	"encoding/binary"
	"math"
	"testing"
)

// Tests for NegWhereNeg, the branchless conditional negate. The predicate is
// purely the IEEE-754 sign bit of the float32 sign lane, so -0.0/-Inf/-NaN
// negate while +0.0/+Inf/+NaN do not; the load-bearing magnitude is MinInt32,
// whose negation does not fit int32 and must wrap in place, exactly as Abs does.

// signbitF32 reports whether f's IEEE-754 sign bit is set, computed straight
// from the bit pattern so it is deterministic for every input including -0.0,
// -Inf and negative NaN (math.Signbit routes through float64, whose NaN sign
// preservation is not guaranteed across architectures).
func signbitF32(f float32) bool {
	return math.Float32bits(f)>>31 != 0
}

// negWhereNegOracle is an independent reference: it branches on the sign bit and
// negates in int64, then truncates to int32 the way a 32-bit lane store does.
// Negating in int64 makes the MinInt32 wrap explicit rather than trusting the
// int32 arithmetic under test: -int64(MinInt32) is +2^31, and int32() drops it
// back to MinInt32, which is what the kernel must produce.
func negWhereNegOracle(mag int32, sign float32) int32 {
	if signbitF32(sign) {
		return int32(-int64(mag))
	}
	return mag
}

// genSigns produces a deterministic spread of float32 sign lanes from a cheap
// LCG, reinterpreting the raw 32-bit state as a float32 so sign bits, NaNs and
// infinities all appear across the indices.
func genSigns(n int, seed uint32) []float32 {
	s := make([]float32, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = math.Float32frombits(x)
	}
	return s
}

// f32sFromBits reinterprets raw bytes as little-endian float32s, reaching the
// full sign/exponent/mantissa space (including signed zeros, infinities and
// NaNs) for the differential fuzz target.
func f32sFromBits(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestNegWhereNeg sweeps every tier-3 length against both the pure-Go reference
// and the independent int64 oracle, so a fault cannot hide by agreeing with the
// reference alone. MinInt32 rides index 0 under a guaranteed-negative sign so
// the wrap is exercised at every length.
func TestNegWhereNeg(t *testing.T) {
	for _, n := range tier3Lengths {
		mag := genI32(n, 71)
		sign := genSigns(n, 72)
		if n > 0 {
			mag[0] = math.MinInt32
			sign[0] = math.Float32frombits(1 << 31) // -0.0: sign bit set, negates
		}
		dst := make([]int32, n)
		ref := make([]int32, n)
		NegWhereNeg(dst, mag, sign)
		negWhereNegGo(ref, mag, sign)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf("NegWhereNeg n=%d: dst[%d] = %d, want %d (reference)", n, i, dst[i], ref[i])
			}
			if want := negWhereNegOracle(mag[i], sign[i]); dst[i] != want {
				t.Fatalf("NegWhereNeg n=%d: dst[%d] = %d, want %d (oracle)", n, i, dst[i], want)
			}
		}
	}
}

// TestNegWhereNeg_ValueMatrix crosses the load-bearing magnitudes with the full
// sign-class matrix and plants each pair in every lane position (rotating pos
// across a length that spans a vector block plus a scalar tail on both arches),
// so a lane or index error, and a sign-class the predicate mishandles, are both
// caught.
func TestNegWhereNeg_ValueMatrix(t *testing.T) {
	mags := []int32{math.MinInt32, math.MaxInt32, 0, -1, 1, 0x12345678}
	posZero := math.Float32frombits(0)
	negZero := math.Float32frombits(1 << 31)
	posNaN := math.Float32frombits(0x7FC00000)
	negNaN := math.Float32frombits(0xFFC00000)
	signs := []float32{posZero, negZero, 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)), posNaN, negNaN}

	const n = 11 // one 8-wide AVX2 block + 3 tail; two 4-wide NEON blocks + 3 tail
	filler := genI32(n, 73)
	fillerSign := genSigns(n, 74)
	for _, m := range mags {
		for _, s := range signs {
			for pos := range n {
				mag := append([]int32(nil), filler...)
				sign := append([]float32(nil), fillerSign...)
				mag[pos] = m
				sign[pos] = s
				dst := make([]int32, n)
				NegWhereNeg(dst, mag, sign)
				for i := range dst {
					if want := negWhereNegOracle(mag[i], sign[i]); dst[i] != want {
						t.Fatalf("NegWhereNeg mag=%d sign=%#08x pos=%d: dst[%d] = %d, want %d",
							m, math.Float32bits(s), pos, i, dst[i], want)
					}
				}
			}
		}
	}
}

// TestNegWhereNeg_SignPredicate pins the exact contract in isolation: a negative
// zero negates, a positive zero does not, and MinInt32 negates back to itself.
func TestNegWhereNeg_SignPredicate(t *testing.T) {
	negZero := math.Float32frombits(1 << 31)
	posZero := math.Float32frombits(0)
	cases := []struct {
		name string
		mag  int32
		sign float32
		want int32
	}{
		{"-0.0 negates", 5, negZero, -5},
		{"+0.0 keeps", 5, posZero, 5},
		{"-1.0 negates", 5, -1, -5},
		{"+1.0 keeps", 5, 1, 5},
		{"-Inf negates", 5, float32(math.Inf(-1)), -5},
		{"+Inf keeps", 5, float32(math.Inf(1)), 5},
		{"-NaN negates", 5, math.Float32frombits(0xFFC00000), -5},
		{"+NaN keeps", 5, math.Float32frombits(0x7FC00000), 5},
		{"MinInt32 under -1.0 wraps to itself", math.MinInt32, -1, math.MinInt32},
		{"MinInt32 under -0.0 wraps to itself", math.MinInt32, negZero, math.MinInt32},
		{"MinInt32 under +0.0 unchanged", math.MinInt32, posZero, math.MinInt32},
	}
	for _, c := range cases {
		dst := make([]int32, 1)
		NegWhereNeg(dst, []int32{c.mag}, []float32{c.sign})
		if dst[0] != c.want {
			t.Errorf("%s: NegWhereNeg(%d, %#08x) = %d, want %d",
				c.name, c.mag, math.Float32bits(c.sign), dst[0], c.want)
		}
	}
}

// TestNegWhereNeg_TailUntouched plants sentinels past the clamp point at n=11
// (one 8-wide AVX2 block + 3 tail, two 4-wide NEON blocks + the same tail), so
// both vector bodies run and both scalar tails must stop exactly at n.
func TestNegWhereNeg_TailUntouched(t *testing.T) {
	const n = 11
	mag := genI32(n, 75)
	sign := genSigns(n, 76)
	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // non-zero sentinel
	}
	NegWhereNeg(dst[:n], mag, sign)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("NegWhereNeg wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestNegWhereNeg_Clamp covers mismatched lengths across all three operands and
// the empty no-op: n is the shortest of dst, mag and sign, and nothing past it
// is touched.
func TestNegWhereNeg_Clamp(t *testing.T) {
	mag := genI32(40, 77)
	sign := genSigns(40, 78)

	short := make([]int32, 25) // dst shortest: n = 25
	NegWhereNeg(short, mag, sign)
	for i := range short {
		if want := negWhereNegOracle(mag[i], sign[i]); short[i] != want {
			t.Fatalf("NegWhereNeg short dst: dst[%d] = %d, want %d", i, short[i], want)
		}
	}

	long := make([]int32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	NegWhereNeg(long, mag[:25], sign) // mag shortest: n = 25, long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("NegWhereNeg wrote past mag clamp at dst[%d] = %d", i, long[i])
		}
	}

	for i := range long {
		long[i] = -7
	}
	NegWhereNeg(long, mag, sign[:25]) // sign shortest: n = 25, long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("NegWhereNeg wrote past sign clamp at dst[%d] = %d", i, long[i])
		}
	}

	NegWhereNeg(nil, nil, nil)
	one := []int32{42}
	NegWhereNeg(one, nil, nil)
	if one[0] != 42 {
		t.Errorf("NegWhereNeg wrote on empty input: %v", one)
	}
}

// TestNegWhereNeg_Aliasing negates the magnitudes in place (dst == mag) and
// confirms every lane matches the oracle computed from the saved originals. The
// kernel reads each mag lane before its own store, so the in-place overlay is
// well defined lane by lane.
func TestNegWhereNeg_Aliasing(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 11, 16, 17, 33, 64} {
		buf := genI32(n, 79)
		if n > 0 {
			buf[0] = math.MinInt32
		}
		orig := append([]int32(nil), buf...)
		sign := genSigns(n, 80)
		NegWhereNeg(buf, buf, sign) // dst aliases mag
		for i := range buf {
			if want := negWhereNegOracle(orig[i], sign[i]); buf[i] != want {
				t.Fatalf("NegWhereNeg in-place n=%d: dst[%d] = %d, want %d", n, i, buf[i], want)
			}
		}
	}
}

// TestNegWhereNeg_UnalignedOperands sweeps all eight element offsets, holding
// dst, mag and sign at different offsets from one another so none is reliably
// aligned and an aligned-load or aligned-store substitution cannot survive.
func TestNegWhereNeg_UnalignedOperands(t *testing.T) {
	const span = 300
	baseMag := genI32(span, 81)
	baseSign := genSigns(span, 82)
	backing := make([]int32, span)
	for _, n := range []int{8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			mag := baseMag[off+1 : off+1+n]
			sign := baseSign[off+2 : off+2+n]
			dst := backing[off+3 : off+3+n]
			NegWhereNeg(dst, mag, sign)
			for i := range n {
				if want := negWhereNegOracle(mag[i], sign[i]); dst[i] != want {
					t.Fatalf("NegWhereNeg unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// TestNegWhereNeg_AllocFree declares the buffers INSIDE the measured closure so
// only allocations forced by NegWhereNeg itself are counted.
func TestNegWhereNeg_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var mag, dst [1000]int32
		var sign [1000]float32
		NegWhereNeg(dst[:], mag[:], sign[:])
	}); n != 0 {
		t.Errorf("NegWhereNeg forces %v caller allocations per run, want 0", n)
	}
}

// negWhereNegLenSeeds seeds raw byte buffers whose float32/int32 element counts
// cover 0 through ~70, hitting every remainder around the 8/16-lane unrolls plus
// a couple of larger blocks.
func negWhereNegLenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for _, n := range lens {
		mag := make([]byte, n*4)
		sign := make([]byte, n*4)
		for i := range mag {
			mag[i] = byte(i*37 + 11)
			sign[i] = byte(i*29 + 5)
		}
		f.Add(mag, sign)
	}
}

// FuzzNegWhereNeg differentially fuzzes the dispatched NegWhereNeg against the
// pure-Go reference and the independent int64 oracle over arbitrary int32
// magnitudes and arbitrary float32 sign bits, so tail handling and the sign-bit
// predicate are explored past the hand-picked seeds.
func FuzzNegWhereNeg(f *testing.F) {
	negWhereNegLenSeeds(f)
	f.Fuzz(func(t *testing.T, magRaw, signRaw []byte) {
		mag := i32sFromBits(magRaw)
		sign := f32sFromBits(signRaw)
		n := min(len(mag), len(sign))
		mag, sign = mag[:n], sign[:n]

		got := make([]int32, n)
		want := make([]int32, n)
		NegWhereNeg(got, mag, sign)
		negWhereNegGo(want, mag, sign)
		equalI32(t, "NegWhereNeg", got, want)
		for i := range got {
			if o := negWhereNegOracle(mag[i], sign[i]); got[i] != o {
				t.Fatalf("NegWhereNeg oracle mismatch at %d: got %d want %d (len=%d)", i, got[i], o, n)
			}
		}
	})
}
