package i32

import (
	"math"
	"math/big"
	"testing"
)

// Tests for GainQ31, the fused Q31 gain with an input pre-shift and a rounding
// output post-shift. Each stage wraps in int32 rather than saturating, so the
// load-bearing cases are the ScaleQ31 wrap (MinInt32 * MinInt32 -> MinInt32) plus
// the PSHR32 bias-add wrap (PSHR32(MaxInt32, 1) = -1073741824, not a saturated
// MaxInt32) and the SHL32 pre-shift wrap (a positive sample shifted past bit 31
// changes sign). The kernel result is bit-identical to the ScaleQ31 + SHL32/PSHR32
// composition in libopus/go-opus, including the bias definition (int32(1)<<s)>>1,
// which sign-extends at s == 31.

var (
	oracleMask32   = big.NewInt(0xFFFFFFFF)
	oracleTwoPow32 = new(big.Int).Lsh(big.NewInt(1), 32)
)

// trunc32Big reduces x to its signed 32-bit value as a big.Int, the wrap a 32-bit
// store performs: keep the low 32 bits (big.Int And uses two's-complement
// semantics, so this works for negative x) and subtract 2^32 when bit 31 is set.
func trunc32Big(x *big.Int) *big.Int {
	m := new(big.Int).And(x, oracleMask32)
	if m.Bit(31) == 1 {
		m.Sub(m, oracleTwoPow32)
	}
	return m
}

// gainQ31Oracle recomputes GainQ31's per-element result in arbitrary precision,
// truncating to a signed 32-bit lane at every stage boundary exactly as the SIMD
// int32 lanes and the store do. It is independent of the int64 arithmetic in
// gainQ31Go (big.Int throughout, the wrap modeled explicitly), so a fault cannot
// hide by agreeing with the reference alone. big.Int.Rsh is a floor (arithmetic)
// shift, matching Go's signed >>, so the bias >>1 and the final >> postShift both
// match the int32 arithmetic-shift semantics.
func gainQ31Oracle(a, gain int32, preShift, postShift int) int32 {
	// SHL32: wrapping left shift in int32.
	x := new(big.Int).Lsh(big.NewInt(int64(a)), uint(preShift))
	x = trunc32Big(x)

	// MULT32_32_Q31: 64-bit product arithmetically shifted right by 31.
	p := new(big.Int).Mul(x, big.NewInt(int64(gain)))
	p.Rsh(p, scaleQ31Shift)
	v := trunc32Big(p)

	// PSHR32: bias = (int32(1)<<postShift)>>1, arithmetic (sign-extends at 31);
	// then the int32-wrapping add and the arithmetic >> postShift.
	bias := new(big.Int).Lsh(big.NewInt(1), uint(postShift))
	bias = trunc32Big(bias) // (int32(1)<<postShift), wrapped to int32
	bias.Rsh(bias, 1)       // arithmetic >>1 (floor)
	w := trunc32Big(new(big.Int).Add(v, bias))
	w.Rsh(w, uint(postShift))
	return int32(w.Int64())
}

// gainShifts is the (preShift, postShift) matrix swept by the parity tests. It
// covers the identity pair (which must reduce GainQ31 to exact ScaleQ31), the
// no-rounding post-shift, the go-opus denormaliseBands shape (30, 15), and the
// [0,31] range endpoints where the bias sign-extends.
var gainShifts = []struct{ pre, post int }{
	{0, 0}, {0, 1}, {5, 2}, {9, 12}, {30, 15}, {31, 0}, {0, 31}, {31, 31},
}

// TestGainQ31 sweeps every tier-3 length against both the pure-Go reference and
// the arbitrary-precision oracle across the gain and shift matrices, so a fault
// cannot hide by agreeing with the reference alone. MinInt32 rides index 0 and
// MaxInt32 the last index so the wraps are exercised at every length and in the
// tail.
//
//nolint:dupl // The dispatched/AVX2/NEON parity sweeps are intentionally identical bar the entry point under test.
func TestGainQ31(t *testing.T) {
	gains := []int32{math.MinInt32, math.MaxInt32, 1, -1, 0x40000000, 0x2BADF00D}
	for _, n := range tier3Lengths {
		a := genI32(n, 51)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, g := range gains {
			for _, s := range gainShifts {
				dst := make([]int32, n)
				ref := make([]int32, n)
				GainQ31(dst, a, g, s.pre, s.post)
				gainQ31Go(ref, a, g, s.pre, s.post)
				for i := range dst {
					if dst[i] != ref[i] {
						t.Fatalf("GainQ31 n=%d g=%d pre=%d post=%d: dst[%d] = %d, want %d (reference)", n, g, s.pre, s.post, i, dst[i], ref[i])
					}
					if want := gainQ31Oracle(a[i], g, s.pre, s.post); dst[i] != want {
						t.Fatalf("GainQ31 n=%d g=%d pre=%d post=%d: dst[%d] = %d, want %d (oracle)", n, g, s.pre, s.post, i, dst[i], want)
					}
				}
			}
		}
	}
}

// TestGainQ31_Identity pins that pre=post=0 makes GainQ31 exactly ScaleQ31, the
// core it fuses around: with no pre-shift, a zero bias and a zero post-shift the
// two must agree lane for lane over the vector body and the tail.
func TestGainQ31_Identity(t *testing.T) {
	gains := []int32{math.MinInt32, math.MaxInt32, 0, 1, -1, 0x40000000, -0x40000000, 0x2BADF00D}
	const n = 37 // several AVX2 blocks + tail, several NEON blocks + tail
	a := genI32(n, 52)
	a[0] = math.MinInt32
	a[n-1] = math.MaxInt32
	for _, g := range gains {
		got := make([]int32, n)
		want := make([]int32, n)
		GainQ31(got, a, g, 0, 0)
		ScaleQ31(want, a, g)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("GainQ31(pre=0,post=0) g=%d: dst[%d] = %d, want %d (ScaleQ31)", g, i, got[i], want[i])
			}
		}
	}
}

// TestGainQ31_ValueMatrix crosses the load-bearing samples with the gain and shift
// matrices and plants each sample in every lane position across n=11 (one 8-wide
// AVX2 block + 3 tail; two 4-wide NEON blocks + 3 tail), so a lane error, an index
// error and a value the even/odd VPMULDQ recombine mishandles are all caught. The
// samples include the SHL32 sign-flip cases (0x40000001<<1, 0x60000000<<2) and the
// MULT32_32_Q31 wrap edges.
func TestGainQ31_ValueMatrix(t *testing.T) {
	as := []int32{math.MinInt32, math.MinInt32 + 1, math.MaxInt32, 0, -1, 1, 2, 0x40000001, 0x60000000, -0x12345678, 0x7FFFFFFE}
	gains := []int32{math.MinInt32, math.MaxInt32, 0, 1, -1, 2, 0x40000000, -0x40000000, 0x2BADF00D}
	shifts := []struct{ pre, post int }{{0, 0}, {1, 1}, {2, 0}, {0, 31}, {31, 0}, {30, 15}}
	const n = 11
	filler := genI32(n, 53)
	for _, s := range shifts {
		for _, g := range gains {
			for _, av := range as {
				for pos := range n {
					a := append([]int32(nil), filler...)
					a[pos] = av
					dst := make([]int32, n)
					GainQ31(dst, a, g, s.pre, s.post)
					for i := range dst {
						if want := gainQ31Oracle(a[i], g, s.pre, s.post); dst[i] != want {
							t.Fatalf("GainQ31 a=%d g=%d pre=%d post=%d pos=%d: dst[%d] = %d, want %d", av, g, s.pre, s.post, pos, i, dst[i], want)
						}
					}
				}
			}
		}
	}
}

// TestGainQ31_Wrap pins the extreme contracts in isolation with values computed by
// hand, independent of the oracle, at n=11 so both the vector body and the scalar
// tail run on both arches.
func TestGainQ31_Wrap(t *testing.T) {
	const n = 11
	fill := func(v int32) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = v
		}
		return s
	}
	check := func(name string, a []int32, g int32, pre, post int, want int32) {
		t.Helper()
		dst := fill(-999)
		GainQ31(dst, a, g, pre, post)
		for i := range dst {
			if dst[i] != want {
				t.Fatalf("%s: dst[%d] = %d, want %d", name, i, dst[i], want)
			}
		}
	}

	// ScaleQ31 wrap, unchanged by the identity shifts: MinInt32*MinInt32 = 2^62,
	// >>31 = 2^31 -> MinInt32. A saturating build would return MaxInt32.
	check("MinInt32*MinInt32 pre=0 post=0", fill(math.MinInt32), math.MinInt32, 0, 0, math.MinInt32)

	// PSHR32 bias-add wrap: MULT32_32_Q31(MinInt32+1, MinInt32) = MaxInt32, then
	// PSHR32(MaxInt32, 1) = (MaxInt32+1 wraps to MinInt32) >> 1 = -1073741824.
	check("PSHR32(MaxInt32,1)", fill(math.MinInt32+1), math.MinInt32, 0, 1, -1073741824)

	// SHL32 pre-shift wrap: SHL32(1, 31) = MinInt32 from a positive sample, then
	// MULT32_32_Q31(MinInt32, MaxInt32) = -(2^31-1) = MinInt32+1, PSHR32(_,0) = it.
	check("SHL32(1,31) pre-wrap", fill(1), math.MaxInt32, 31, 0, math.MinInt32+1)

	// gain 0 with the real (post=0) pairing zeroes everything.
	check("gain=0 post=0", fill(math.MinInt32), 0, 7, 0, 0)

	// go-opus PSHR32(0,31) contract, externally pinning the arithmetic bias choice.
	// bias@31 = (int32(1)<<31)>>1 = -2^30 (arithmetic, sign-extended), so with a zero
	// product (gain=0) the result is (-2^30)>>31 = -1, NOT 0. A logical bias 1<<(n-1)
	// would give +2^30 here and (+2^30)>>31 = 0, so this case distinguishes the two.
	check("PSHR32(0,31) arithmetic bias", fill(12345), 0, 0, 31, -1)
}

// TestGainQ31_TailUntouched plants sentinels past the clamp point at n=11 so both
// vector bodies run and both scalar tails must stop exactly at n.
func TestGainQ31_TailUntouched(t *testing.T) {
	const n = 11
	a := genI32(n, 54)
	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // non-zero sentinel
	}
	GainQ31(dst[:n], a, 0x0BADBEEF, 9, 12)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("GainQ31 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestGainQ31_Clamp covers mismatched dst/a lengths and the empty no-op: n is the
// shorter of dst and a, and nothing past it is touched.
func TestGainQ31_Clamp(t *testing.T) {
	a := genI32(40, 55)

	short := make([]int32, 25) // dst shortest: n = 25
	GainQ31(short, a, 0x12345678, 3, 4)
	for i := range short {
		if want := gainQ31Oracle(a[i], 0x12345678, 3, 4); short[i] != want {
			t.Fatalf("GainQ31 short dst: dst[%d] = %d, want %d", i, short[i], want)
		}
	}

	long := make([]int32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	GainQ31(long, a[:25], 0x12345678, 3, 4) // a shortest: n = 25, long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("GainQ31 wrote past a clamp at dst[%d] = %d", i, long[i])
		}
	}

	// Empty inputs are a no-op.
	GainQ31(nil, nil, 5, 2, 3)
	one := []int32{42}
	GainQ31(one, nil, 5, 2, 3)
	if one[0] != 42 {
		t.Errorf("GainQ31 wrote on empty input: %v", one)
	}
}

// TestGainQ31_ShiftRange asserts the public entry point rejects out-of-range shift
// counts (which would otherwise diverge across backends) and accepts the [0,31]
// endpoints. Validation runs before the length check, so it fires even on empty
// slices.
func TestGainQ31_ShiftRange(t *testing.T) {
	dst := make([]int32, 4)
	a := make([]int32, 4)

	bad := []struct {
		name                string
		preShift, postShift int
	}{
		{"preShift<0", -1, 0},
		{"preShift>31", 32, 0},
		{"postShift<0", 0, -1},
		{"postShift>31", 0, 32},
		{"both invalid", -5, 40},
		{"invalid on empty", 32, 0},
	}
	for _, c := range bad {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("GainQ31(pre=%d post=%d) did not panic", c.preShift, c.postShift)
				}
			}()
			in := dst
			if c.name == "invalid on empty" {
				in = nil // validation must precede the n==0 short-circuit
			}
			GainQ31(in, a, 1, c.preShift, c.postShift)
		})
	}

	// The [0,31] endpoints are valid and must not panic.
	for _, s := range []struct{ pre, post int }{{0, 0}, {0, 31}, {31, 0}, {31, 31}} {
		GainQ31(dst, a, 0x2BADF00D, s.pre, s.post)
	}
}

// TestGainQ31_Aliasing processes the samples in place (dst == a) and confirms every
// lane matches the oracle computed from the saved originals. The kernel reads each
// a lane before its own store, so the in-place overlay is well defined lane by
// lane. MinInt32 rides index 0 so the wrap is exercised in place.
func TestGainQ31_Aliasing(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 11, 16, 17, 33, 64} {
		const g = int32(-0x40000000)
		const pre, post = 6, 9
		buf := genI32(n, 56)
		if n > 0 {
			buf[0] = math.MinInt32
		}
		orig := append([]int32(nil), buf...)
		GainQ31(buf, buf, g, pre, post) // dst aliases a
		for i := range buf {
			if want := gainQ31Oracle(orig[i], g, pre, post); buf[i] != want {
				t.Fatalf("GainQ31 in-place n=%d: dst[%d] = %d, want %d", n, i, buf[i], want)
			}
		}
	}
}

// TestGainQ31_UnalignedOperands sweeps all eight element offsets, holding dst and a
// at different offsets from one another so neither is reliably aligned and an
// aligned-load or aligned-store substitution cannot survive.
func TestGainQ31_UnalignedOperands(t *testing.T) {
	const span = 300
	baseA := genI32(span, 57)
	backing := make([]int32, span)
	const g = int32(0x50000001)
	const pre, post = 11, 7
	for _, n := range []int{8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			a := baseA[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			GainQ31(dst, a, g, pre, post)
			for i := range n {
				if want := gainQ31Oracle(a[i], g, pre, post); dst[i] != want {
					t.Fatalf("GainQ31 unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// TestGainQ31_AllocFree declares the buffers INSIDE the measured closure so only
// allocations forced by the gain itself are counted.
func TestGainQ31_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a, dst [1000]int32
		GainQ31(dst[:], a[:], 0x12345678, 9, 12)
	}); n != 0 {
		t.Errorf("GainQ31 forces %v caller allocations per run, want 0", n)
	}
}

// gainLenSeeds seeds raw byte buffers whose int32 element counts cover 0 through
// ~70, hitting every remainder around the 8/16-lane unrolls plus larger blocks,
// each paired with a gain and pre/post shift seed.
func gainLenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for i, n := range lens {
		raw := make([]byte, n*4)
		for j := range raw {
			raw[j] = byte(j*37 + 11)
		}
		f.Add(raw, int32(0x12345678), uint8(i%32), uint8((i*3)%32))
	}
	f.Add(make([]byte, 44), int32(math.MinInt32), uint8(31), uint8(31))
	f.Add(make([]byte, 44), int32(0), uint8(0), uint8(0))
}

// FuzzGainQ31 differentially fuzzes the dispatched GainQ31 against the pure-Go
// reference and the arbitrary-precision oracle over arbitrary int32 samples, an
// arbitrary int32 gain and shift counts masked into the [0,31] precondition, so
// tail handling and every wrap are explored past the hand-picked seeds.
func FuzzGainQ31(f *testing.F) {
	gainLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte, g int32, pre, post uint8) {
		preShift := int(pre & 31)
		postShift := int(post & 31)
		a := i32sFromBits(raw)
		got := make([]int32, len(a))
		want := make([]int32, len(a))
		GainQ31(got, a, g, preShift, postShift)
		gainQ31Go(want, a, g, preShift, postShift)
		equalI32(t, "GainQ31", got, want)
		for i := range got {
			if o := gainQ31Oracle(a[i], g, preShift, postShift); got[i] != o {
				t.Fatalf("GainQ31 oracle mismatch at %d: got %d want %d (len=%d, g=%d, pre=%d, post=%d)", i, got[i], o, len(a), g, preShift, postShift)
			}
		}
	})
}
