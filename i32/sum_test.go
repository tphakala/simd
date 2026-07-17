package i32

import (
	"math"
	"testing"
)

// Tests for Sum, the wrapping int32 total.
//
// Wrapping accumulation is the documented contract and the reason the kernels
// may reassociate at all, so overflow is the main event here, not an edge
// case: a saturating implementation would pass a small-value test and fail
// these.

// genI32 produces a deterministic spread across the full int32 range from a
// cheap LCG, so sign and high bits are exercised at every index.
func genI32(n int, seed uint32) []int32 {
	s := make([]int32, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int32(x)
	}
	return s
}

// sumOracle sums in int64 and truncates to int32, independent of sumGo:
// wrapping int32 addition is addition modulo 2^32, and an int64 sum of a few
// thousand int32 values cannot itself overflow, so truncating the exact sum
// must equal the wrapping sum. It pins the reference rather than trusting it.
func sumOracle(a []int32) int32 {
	var s int64
	for _, v := range a {
		s += int64(v)
	}
	return int32(s) // deliberate truncation: this is the wrapping contract
}

// tier3Lengths sweeps EVERY length 0..64, then a spread of larger ones,
// covering each vector-block/scalar-tail split of the 4-wide NEON and 8-wide
// AVX2 bodies, with non-multiples of both widths in the spread.
var tier3Lengths = func() []int {
	lens := make([]int, 0, 76)
	for n := range 65 {
		lens = append(lens, n)
	}
	return append(lens, 100, 127, 128, 129, 240, 255, 256, 257, 1000, 1003, 1024)
}()

func TestSum(t *testing.T) {
	if got := Sum(nil); got != 0 {
		t.Errorf("Sum(nil) = %d, want 0", got)
	}
	if got := Sum([]int32{}); got != 0 {
		t.Errorf("Sum([]) = %d, want 0", got)
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 8)
		got := Sum(a)
		if want := sumGo(a); got != want {
			t.Errorf("Sum n=%d: got %d, want %d (reference)", n, got, want)
		}
		if want := sumOracle(a); got != want {
			t.Errorf("Sum n=%d: got %d, want %d (int64 oracle)", n, got, want)
		}
	}
}

// TestSum_Wraparound drives the accumulator through many wraps with same-sign
// values, so an implementation that clamped anywhere along the chain diverges
// well before the end.
func TestSum_Wraparound(t *testing.T) {
	for _, n := range []int{4, 8, 11, 25, 240, 1003, 5000} {
		a := make([]int32, n)
		for i := range a {
			a[i] = math.MaxInt32 - int32(i%3) // wraps roughly every 2 elements
		}
		if got, want := Sum(a), sumOracle(a); got != want {
			t.Errorf("Sum wraparound n=%d: got %d, want %d", n, got, want)
		}
	}
	// All-MinInt32 operands are sign-symmetric under wrap (each element
	// contributes 2^31 modulo 2^32 regardless of sign), so this sweep can only
	// catch a miscounted element, never a sign, lane, or index error; the
	// genI32 sweeps above are what catch those.
	for n := 1; n <= 64; n++ {
		a := make([]int32, n)
		for i := range a {
			a[i] = math.MinInt32
		}
		if got, want := Sum(a), sumOracle(a); got != want {
			t.Errorf("Sum all-MinInt32 n=%d: got %d, want %d", n, got, want)
		}
	}
}

// sumOverReadPoison fills the memory past an operand in the over-read tests.
//
// The value has to be chosen with care, and the obvious extreme is wrong.
// MinInt32 is the one non-zero int32 of additive order 2: 2*MinInt32 = -2^32,
// which is 0 modulo 2^32. Reading an even number of MinInt32 elements past the
// operand therefore adds exactly nothing to a wrapping sum, and every kernel
// block here is an even width (4 on NEON, 8 on AVX2), so a whole-block
// over-read, the realistic defect, would stay invisible. This is not
// hypothetical: a sumNEON that reads one block past the operand passes this
// test under a MinInt32 poison and fails it under the poison below.
//
// 0x5555_5555 is odd, so gcd(x, 2^32) = 1 and its additive order is the full
// 2^32. No over-read length can cancel.
const sumOverReadPoison = 0x5555_5555

// TestSum_NoOverRead hands Sum a prefix of a longer allocation whose every
// element past the prefix is poisoned. Past a standalone slice lies zeroed
// memory and +0 is the identity element of this reduction, so an over-reading
// kernel would pass every other test in this file; the poison is what makes
// one visible. See sumOverReadPoison for why the value is not an extreme.
func TestSum_NoOverRead(t *testing.T) {
	backing := make([]int32, 64+8)
	for i := range backing {
		backing[i] = sumOverReadPoison
	}
	for _, n := range []int{1, 3, 4, 5, 7, 8, 9, 11, 15, 16, 17, 25, 32, 33, 64} {
		a := backing[:n]
		for i := range a {
			a[i] = int32(i - 5)
		}
		want := sumOracle(a)
		if got := Sum(a); got != want {
			t.Fatalf("Sum n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = sumOverReadPoison // restore the poison for the next length
		}
	}
}

// TestSum_AllocFree declares the buffer INSIDE the measured closure: hoisting
// it out measures only Sum's own allocations and passes even when Sum leaks
// its parameter, forcing every caller to heap-allocate.
func TestSum_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a [1000]int32
		_ = Sum(a[:])
	}); n != 0 {
		t.Errorf("Sum forces %v caller allocations per run, want 0", n)
	}
}
