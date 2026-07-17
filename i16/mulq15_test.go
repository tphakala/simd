package i16

import (
	"math"
	"testing"
)

// Tests for MulQ15, the rounding Q15 fixed-point multiply.
//
// Two properties carry this suite: the ROUNDING form (add 2^14 before the
// shift, libopus MULT16_16_P15) rather than truncation, and the narrowing
// WRAP at the one input pair whose product exceeds int16 range. Both have
// tiny discriminating inputs, pinned below at lengths that put the vector
// bodies on the hook rather than just the Go path or a scalar tail.

// mulQ15Oracle computes one rounded Q15 product in int64, independent of
// mulQ15Go: the widening, the rounding add and the shift cannot overflow
// int64, and the final int16 conversion is the same defined Go truncation, so
// it pins the reference rather than trusting it.
func mulQ15Oracle(a, b int16) int16 {
	return int16((int64(a)*int64(b) + q15Round) >> q15Shift)
}

// tier3Lengths sweeps EVERY length 0..64, then a spread of larger ones. The
// exhaustive low range covers each vector-block/scalar-tail split of the
// 8-wide NEON and 16-wide AVX2 bodies (a hand-picked list of block boundaries
// misses combinations, see dotLengths); the spread keeps non-multiples of
// both widths so the tails also run at realistic lengths.
var tier3Lengths = func() []int {
	lens := make([]int, 0, 76)
	for n := range 65 {
		lens = append(lens, n)
	}
	return append(lens, 100, 127, 128, 129, 240, 255, 256, 257, 1000, 1003, 1024)
}()

func TestMulQ15(t *testing.T) {
	for _, n := range tier3Lengths {
		a, b := genI16(n, 71), genI16(n, 72)
		dst := make([]int16, n)
		ref := make([]int16, n)
		MulQ15(dst, a, b)
		mulQ15Go(ref, a, b)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf("MulQ15 n=%d: dst[%d] = %d, want %d (reference)", n, i, dst[i], ref[i])
			}
			if want := mulQ15Oracle(a[i], b[i]); dst[i] != want {
				t.Fatalf("MulQ15 n=%d: dst[%d] = %d, want %d (int64 oracle)", n, i, dst[i], want)
			}
		}
	}
}

// TestMulQ15_Rounding pins the rounding form with the smallest discriminating
// inputs, at n=19 so the vector bodies (one 16-wide AVX2 block, two 8-wide
// NEON blocks) compute most positions and the scalar tails the rest.
// 1 * 16384 = 2^14 rounds up to 1 where a truncating shift gives 0, and
// -1 * 16384 = -2^14 rounds up to 0 where a truncating (floor) shift gives -1.
func TestMulQ15_Rounding(t *testing.T) {
	const n = 19
	cases := []struct {
		a, b, want int16
	}{
		{1, 16384, 1},
		{-1, 16384, 0},
		{16384, 16384, 8192},
		{-16384, 16384, -8192},
		{32767, 32767, 32766},
	}
	for _, c := range cases {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i], b[i] = c.a, c.b
		}
		dst := make([]int16, n)
		MulQ15(dst, a, b)
		for i := range dst {
			if dst[i] != c.want {
				t.Errorf("MulQ15(%d, %d): dst[%d] = %d, want %d", c.a, c.b, i, dst[i], c.want)
			}
		}
	}
}

// TestMulQ15_MinInt16 pins the narrowing wrap at the one product outside
// int16 range: (-32768)^2 rounds to +32768 and must wrap to -32768. A
// saturating implementation (NEON SQRDMULH) returns 32767 here, which is why
// that instruction is banned from the kernels. The sweep walks the pair
// through every lane position and into the scalar tails.
func TestMulQ15_MinInt16(t *testing.T) {
	// n=1 is below every dispatch threshold, pinning the reference itself.
	one := []int16{math.MinInt16}
	first := make([]int16, 1)
	MulQ15(first, one, one)
	if first[0] != math.MinInt16 {
		t.Errorf("MulQ15(MinInt16, MinInt16) = %d, want %d (saturation would give %d)",
			first[0], math.MinInt16, math.MaxInt16)
	}
	for n := 1; n <= 64; n++ {
		a := make([]int16, n)
		for i := range a {
			a[i] = math.MinInt16
		}
		got := make([]int16, n)
		MulQ15(got, a, a)
		for i := range got {
			if got[i] != math.MinInt16 {
				t.Fatalf("MulQ15 all-MinInt16 n=%d: dst[%d] = %d, want %d", n, i, got[i], math.MinInt16)
			}
		}
	}
}

// TestMulQ15_MinInt16Exhaustive multiplies -32768 by every int16 value, so
// the full b axis of the extreme row is pinned against the oracle. It is
// cheap (one 65536-element pass) and deterministic, and unlike the
// all-MinInt16 sweep it is not sign-symmetric, so a lane or index error is
// visible, not just a miscount.
func TestMulQ15_MinInt16Exhaustive(t *testing.T) {
	const total = 1 << 16
	a := make([]int16, total)
	b := make([]int16, total)
	for i := range b {
		a[i] = math.MinInt16
		b[i] = int16(i - 32768)
	}
	dst := make([]int16, total)
	ref := make([]int16, total)
	MulQ15(dst, a, b)
	mulQ15Go(ref, a, b)
	for i := range dst {
		if dst[i] != ref[i] {
			t.Fatalf("MulQ15(-32768, %d) = %d, want %d (reference)", b[i], dst[i], ref[i])
		}
		if want := mulQ15Oracle(a[i], b[i]); dst[i] != want {
			t.Fatalf("MulQ15(-32768, %d) = %d, want %d (oracle)", b[i], dst[i], want)
		}
	}
}

// TestMulQ15_Clamp covers mismatched operand lengths in every order: n is
// min(len(dst), len(a), len(b)) and the trailing dst elements stay untouched.
func TestMulQ15_Clamp(t *testing.T) {
	a := genI16(40, 73)
	b := genI16(40, 74)
	for _, tc := range []struct{ nd, na, nb int }{
		{40, 40, 25}, {40, 25, 40}, {25, 40, 40},
		{40, 25, 19}, {19, 40, 25}, {25, 19, 40},
	} {
		dst := make([]int16, 48)
		for i := range dst {
			dst[i] = math.MaxInt16 // sentinel
		}
		MulQ15(dst[:tc.nd], a[:tc.na], b[:tc.nb])
		n := min(tc.nd, tc.na, tc.nb)
		for i := range n {
			if want := mulQ15Oracle(a[i], b[i]); dst[i] != want {
				t.Fatalf("MulQ15(%d,%d,%d): dst[%d] = %d, want %d", tc.nd, tc.na, tc.nb, i, dst[i], want)
			}
		}
		for i := n; i < len(dst); i++ {
			if dst[i] != math.MaxInt16 {
				t.Fatalf("MulQ15(%d,%d,%d) wrote past clamp at dst[%d] = %d", tc.nd, tc.na, tc.nb, i, dst[i])
			}
		}
	}
}

// TestMulQ15_Empty: no panics and no writes on empty or nil inputs.
func TestMulQ15_Empty(t *testing.T) {
	MulQ15(nil, nil, nil)
	dst := []int16{99}
	MulQ15(dst, nil, []int16{1})
	if dst[0] != 99 {
		t.Errorf("MulQ15 wrote on empty input: %v", dst)
	}
}

// TestMulQ15_UnalignedOperands sweeps all eight element offsets, holding the
// three operands at three different offsets from each other, so no operand is
// reliably aligned and an aligned-load or aligned-store substitution
// (VMOVDQU -> VMOVDQA) cannot survive the suite: a 16-byte-aligned access
// needs an element offset that is a multiple of 8, which only one of the eight
// iterations supplies. Every other test builds operands with make, which is
// aligned.
func TestMulQ15_UnalignedOperands(t *testing.T) {
	const span = 300
	base, other := genI16(span, 75), genI16(span, 76)
	backing := make([]int16, span)
	for _, n := range []int{16, 17, 19, 25, 33, 64, 100, 240} {
		for off := range 8 {
			a := base[off : off+n]
			b := other[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			MulQ15(dst, a, b)
			for i := range n {
				if want := mulQ15Oracle(a[i], b[i]); dst[i] != want {
					t.Fatalf("MulQ15 unaligned n=%d off=%d: dst[%d] = %d, want %d", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// TestMulQ15_AllocFree pins the zero-allocation contract from the CALLER's
// side. The buffers are declared INSIDE the measured closure deliberately:
// hoisting them out measures only MulQ15's own allocations and passes even
// when MulQ15 leaks its parameters, forcing every caller to heap-allocate.
func TestMulQ15_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a, b, dst [240]int16
		MulQ15(dst[:], a[:], b[:])
	}); n != 0 {
		t.Errorf("MulQ15 forces %v caller allocations per run, want 0", n)
	}
}
