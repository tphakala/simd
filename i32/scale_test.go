package i32

import (
	"math"
	"math/big"
	"testing"
)

// Tests for ScaleQ31 and ScaleQ15, the truncating fixed-point scale-by-scalar
// pair. Both wrap in int32 rather than saturating, so the load-bearing case is
// MinInt32 * MinInt32 = 2^62, whose arithmetic shift by 31 is 2^31 and wraps back
// to MinInt32: a saturating implementation would return MaxInt32 and fail here.

// scaleQ31Oracle computes the Q31 scale in arbitrary precision, fully independent
// of the int64 arithmetic in scaleQ31Go and the SIMD kernels. big.Int.Rsh is a
// floor (arithmetic) shift, matching Go's signed >>, and the final int32() drops
// the value back to a 32-bit lane exactly as a store does, so 2^31 wraps to
// MinInt32. It pins the reference rather than trusting a formula shared with it.
func scaleQ31Oracle(a, k int32) int32 {
	return scaleOracle(a, int64(k), 31)
}

// scaleQ15Oracle is the Q15 counterpart: k is a signed int16 coefficient, shift 15.
func scaleQ15Oracle(a int32, k int16) int32 {
	return scaleOracle(a, int64(k), 15)
}

func scaleOracle(a int32, k int64, shift uint) int32 {
	p := new(big.Int).Mul(big.NewInt(int64(a)), big.NewInt(k))
	p.Rsh(p, shift) // floor shift, matches Go's arithmetic >>
	return int32(p.Int64())
}

// TestScaleQ31 sweeps every tier-3 length against both the pure-Go reference and
// the arbitrary-precision oracle, so a fault cannot hide by agreeing with the
// reference alone. MinInt32 rides index 0 so that under k=MinInt32 the wrap is
// exercised at every length; MaxInt32 rides the last index to stress the tail.
func TestScaleQ31(t *testing.T) {
	ks := []int32{math.MinInt32, 1, 0x12345678}
	for _, n := range tier3Lengths {
		a := genI32(n, 21)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, k := range ks {
			dst := make([]int32, n)
			ref := make([]int32, n)
			ScaleQ31(dst, a, k)
			scaleQ31Go(ref, a, k)
			for i := range dst {
				if dst[i] != ref[i] {
					t.Fatalf("ScaleQ31 n=%d k=%d: dst[%d] = %d, want %d (reference)", n, k, i, dst[i], ref[i])
				}
				if want := scaleQ31Oracle(a[i], k); dst[i] != want {
					t.Fatalf("ScaleQ31 n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, dst[i], want)
				}
			}
		}
	}
}

// TestScaleQ15 mirrors TestScaleQ31 over the int16 coefficient range.
func TestScaleQ15(t *testing.T) {
	ks := []int16{math.MinInt16, 1, 0x1234}
	for _, n := range tier3Lengths {
		a := genI32(n, 23)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, k := range ks {
			dst := make([]int32, n)
			ref := make([]int32, n)
			ScaleQ15(dst, a, k)
			scaleQ15Go(ref, a, k)
			for i := range dst {
				if dst[i] != ref[i] {
					t.Fatalf("ScaleQ15 n=%d k=%d: dst[%d] = %d, want %d (reference)", n, k, i, dst[i], ref[i])
				}
				if want := scaleQ15Oracle(a[i], k); dst[i] != want {
					t.Fatalf("ScaleQ15 n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, dst[i], want)
				}
			}
		}
	}
}

// TestScaleQ31_ValueMatrix crosses the load-bearing samples with the full
// coefficient matrix and plants each sample in every lane position across a
// length that spans a vector block plus a scalar tail on both arches, so a lane
// error, an index error, and a value the recombine mishandles are all caught. The
// even/odd lane split of the AVX2 VPMULDQ path makes the per-position sweep
// essential: a swapped even/odd blend only shows up at specific positions.
func TestScaleQ31_ValueMatrix(t *testing.T) {
	as := []int32{math.MinInt32, math.MaxInt32, 0, -1, 1, 2, 0x12345678, -0x12345678, 0x7FFFFFFE}
	ks := []int32{math.MinInt32, math.MaxInt32, 0, 1, -1, 2, 0x40000000, -0x40000000, 0x2BADF00D}
	const n = 11 // one 8-wide AVX2 block + 3 tail; two 4-wide NEON blocks + 3 tail
	filler := genI32(n, 24)
	for _, k := range ks {
		for _, av := range as {
			for pos := range n {
				a := append([]int32(nil), filler...)
				a[pos] = av
				dst := make([]int32, n)
				ScaleQ31(dst, a, k)
				for i := range dst {
					if want := scaleQ31Oracle(a[i], k); dst[i] != want {
						t.Fatalf("ScaleQ31 a=%d k=%d pos=%d: dst[%d] = %d, want %d", av, k, pos, i, dst[i], want)
					}
				}
			}
		}
	}
}

// TestScaleQ15_ValueMatrix is the Q15 counterpart across the int16 coefficient
// space, including MinInt16/MaxInt16 which have no int32 wrap of their own but
// must sign-extend correctly into the signed 32x32 multiply.
func TestScaleQ15_ValueMatrix(t *testing.T) {
	as := []int32{math.MinInt32, math.MaxInt32, 0, -1, 1, 2, 0x12345678, -0x12345678, 0x7FFFFFFE}
	ks := []int16{math.MinInt16, math.MaxInt16, 0, 1, -1, 2, -2, 0x4000, 0x2BAD}
	const n = 11
	filler := genI32(n, 25)
	for _, k := range ks {
		for _, av := range as {
			for pos := range n {
				a := append([]int32(nil), filler...)
				a[pos] = av
				dst := make([]int32, n)
				ScaleQ15(dst, a, k)
				for i := range dst {
					if want := scaleQ15Oracle(a[i], k); dst[i] != want {
						t.Fatalf("ScaleQ15 a=%d k=%d pos=%d: dst[%d] = %d, want %d", av, k, pos, i, dst[i], want)
					}
				}
			}
		}
	}
}

// TestScale_Wrap pins the two extreme contracts in isolation: MinInt32 * MinInt32
// wraps to MinInt32 (a saturating build would return MaxInt32), and k=0 zeroes
// every lane. n=11 forces both the vector body and the scalar tail on both arches.
func TestScale_Wrap(t *testing.T) {
	const n = 11
	minInts := make([]int32, n)
	for i := range minInts {
		minInts[i] = math.MinInt32
	}

	// Q31: MinInt32 * MinInt32 = 2^62, >>31 = 2^31 -> MinInt32.
	dst := make([]int32, n)
	ScaleQ31(dst, minInts, math.MinInt32)
	for i := range dst {
		if dst[i] != math.MinInt32 {
			t.Fatalf("ScaleQ31 MinInt32*MinInt32: dst[%d] = %d, want %d (wrap, not saturate)", i, dst[i], int32(math.MinInt32))
		}
	}

	// Q31: k=0 zeroes everything.
	for i := range dst {
		dst[i] = -1
	}
	ScaleQ31(dst, minInts, 0)
	for i := range dst {
		if dst[i] != 0 {
			t.Fatalf("ScaleQ31 k=0: dst[%d] = %d, want 0", i, dst[i])
		}
	}

	// Q15: MinInt32 * MinInt16 = 2^46, >>15 = 2^31 -> MinInt32.
	ScaleQ15(dst, minInts, math.MinInt16)
	for i := range dst {
		if want := scaleQ15Oracle(math.MinInt32, math.MinInt16); dst[i] != want {
			t.Fatalf("ScaleQ15 MinInt32*MinInt16: dst[%d] = %d, want %d", i, dst[i], want)
		}
	}
	// Confirm the oracle itself says this wraps to MinInt32.
	if got := scaleQ15Oracle(math.MinInt32, math.MinInt16); got != math.MinInt32 {
		t.Fatalf("ScaleQ15 oracle MinInt32*MinInt16 = %d, want %d", got, int32(math.MinInt32))
	}

	// Q15: k=0 zeroes everything.
	for i := range dst {
		dst[i] = -1
	}
	ScaleQ15(dst, minInts, 0)
	for i := range dst {
		if dst[i] != 0 {
			t.Fatalf("ScaleQ15 k=0: dst[%d] = %d, want 0", i, dst[i])
		}
	}
}

// TestScale_TailUntouched plants sentinels past the clamp point at n=11 (one
// 8-wide AVX2 block + 3 tail, two 4-wide NEON blocks + the same tail), so both
// vector bodies run and both scalar tails must stop exactly at n.
func TestScale_TailUntouched(t *testing.T) {
	const n = 11
	a := genI32(n, 26)

	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // non-zero sentinel
	}
	ScaleQ31(dst[:n], a, 0x0BADBEEF)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("ScaleQ31 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}

	for i := range dst {
		dst[i] = math.MaxInt32
	}
	ScaleQ15(dst[:n], a, 0x2BAD)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("ScaleQ15 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestScale_Clamp covers mismatched dst/a lengths and the empty no-op: n is the
// shorter of dst and a, and nothing past it is touched.
func TestScale_Clamp(t *testing.T) {
	a := genI32(40, 27)

	short := make([]int32, 25) // dst shortest: n = 25
	ScaleQ31(short, a, 0x12345678)
	for i := range short {
		if want := scaleQ31Oracle(a[i], 0x12345678); short[i] != want {
			t.Fatalf("ScaleQ31 short dst: dst[%d] = %d, want %d", i, short[i], want)
		}
	}

	long := make([]int32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	ScaleQ31(long, a[:25], 0x12345678) // a shortest: n = 25, long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("ScaleQ31 wrote past a clamp at dst[%d] = %d", i, long[i])
		}
	}

	// Empty inputs are a no-op.
	ScaleQ31(nil, nil, 5)
	ScaleQ15(nil, nil, 5)
	one := []int32{42}
	ScaleQ31(one, nil, 5)
	ScaleQ15(one, nil, 5)
	if one[0] != 42 {
		t.Errorf("ScaleQ31/ScaleQ15 wrote on empty input: %v", one)
	}
}

// TestScale_Aliasing scales the samples in place (dst == a) and confirms every
// lane matches the oracle computed from the saved originals. The kernel reads
// each a lane before its own store, so the in-place overlay is well defined lane
// by lane. MinInt32 rides index 0 so the wrap is exercised in place.
func TestScale_Aliasing(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 11, 16, 17, 33, 64} {
		const k31 = int32(-0x40000000)
		buf := genI32(n, 28)
		if n > 0 {
			buf[0] = math.MinInt32
		}
		orig := append([]int32(nil), buf...)
		ScaleQ31(buf, buf, k31) // dst aliases a
		for i := range buf {
			if want := scaleQ31Oracle(orig[i], k31); buf[i] != want {
				t.Fatalf("ScaleQ31 in-place n=%d: dst[%d] = %d, want %d", n, i, buf[i], want)
			}
		}

		const k15 = int16(-0x4000)
		buf2 := genI32(n, 29)
		if n > 0 {
			buf2[0] = math.MinInt32
		}
		orig2 := append([]int32(nil), buf2...)
		ScaleQ15(buf2, buf2, k15)
		for i := range buf2 {
			if want := scaleQ15Oracle(orig2[i], k15); buf2[i] != want {
				t.Fatalf("ScaleQ15 in-place n=%d: dst[%d] = %d, want %d", n, i, buf2[i], want)
			}
		}
	}
}

// TestScale_UnalignedOperands sweeps all eight element offsets, holding dst and a
// at different offsets from one another so neither is reliably aligned and an
// aligned-load or aligned-store substitution cannot survive.
func TestScale_UnalignedOperands(t *testing.T) {
	const span = 300
	baseA := genI32(span, 31)
	backing := make([]int32, span)
	const k31 = int32(0x50000001)
	const k15 = int16(0x3777)
	for _, n := range []int{8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			a := baseA[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			ScaleQ31(dst, a, k31)
			for i := range n {
				if want := scaleQ31Oracle(a[i], k31); dst[i] != want {
					t.Fatalf("ScaleQ31 unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
			ScaleQ15(dst, a, k15)
			for i := range n {
				if want := scaleQ15Oracle(a[i], k15); dst[i] != want {
					t.Fatalf("ScaleQ15 unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// TestScale_AllocFree declares the buffers INSIDE the measured closure so only
// allocations forced by the scale itself are counted.
func TestScale_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a, dst [1000]int32
		ScaleQ31(dst[:], a[:], 0x12345678)
	}); n != 0 {
		t.Errorf("ScaleQ31 forces %v caller allocations per run, want 0", n)
	}
	if n := testing.AllocsPerRun(50, func() {
		var a, dst [1000]int32
		ScaleQ15(dst[:], a[:], 0x1234)
	}); n != 0 {
		t.Errorf("ScaleQ15 forces %v caller allocations per run, want 0", n)
	}
}

// scaleLenSeeds seeds raw byte buffers whose int32 element counts cover 0 through
// ~70, hitting every remainder around the 8/16-lane unrolls plus larger blocks.
func scaleLenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for _, n := range lens {
		raw := make([]byte, n*4)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw, int32(0x12345678))
	}
}

// FuzzScaleQ31 differentially fuzzes the dispatched ScaleQ31 against the pure-Go
// reference and the arbitrary-precision oracle over arbitrary int32 samples and an
// arbitrary int32 coefficient, so tail handling and the wrap are explored past the
// hand-picked seeds.
func FuzzScaleQ31(f *testing.F) {
	scaleLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte, k int32) {
		a := i32sFromBits(raw)
		got := make([]int32, len(a))
		want := make([]int32, len(a))
		ScaleQ31(got, a, k)
		scaleQ31Go(want, a, k)
		equalI32(t, "ScaleQ31", got, want)
		for i := range got {
			if o := scaleQ31Oracle(a[i], k); got[i] != o {
				t.Fatalf("ScaleQ31 oracle mismatch at %d: got %d want %d (len=%d, k=%d)", i, got[i], o, len(a), k)
			}
		}
	})
}

// FuzzScaleQ15 is the Q15 counterpart; the int16 coefficient is fuzzed as the low
// 16 bits of the provided int32.
func FuzzScaleQ15(f *testing.F) {
	scaleLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte, kRaw int32) {
		k := int16(kRaw)
		a := i32sFromBits(raw)
		got := make([]int32, len(a))
		want := make([]int32, len(a))
		ScaleQ15(got, a, k)
		scaleQ15Go(want, a, k)
		equalI32(t, "ScaleQ15", got, want)
		for i := range got {
			if o := scaleQ15Oracle(a[i], k); got[i] != o {
				t.Fatalf("ScaleQ15 oracle mismatch at %d: got %d want %d (len=%d, k=%d)", i, got[i], o, len(a), k)
			}
		}
	})
}
