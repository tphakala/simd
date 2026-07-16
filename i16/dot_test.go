package i16

import (
	"math"
	"testing"
)

// Tests for DotProduct / DotProductUnsafe.
//
// The property under test is bit-exactness with the scalar reference for every
// input, including inputs that overflow the int32 accumulator. Wrapping
// accumulation is the documented contract and the reason the kernels may
// reassociate at all, so overflow is not an edge case here, it is the main
// event: a kernel that saturated (or that a compiler "helpfully" reassociated
// under saturation) would pass a small-value test and fail these.

// genI16 produces a deterministic spread across the full int16 range from a
// cheap LCG, so sign and high bits are exercised at every index.
func genI16(n int, seed uint32) []int16 {
	s := make([]int16, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int16(x >> 9)
	}
	return s
}

// dotOracle sums in int64 and truncates to int32. This is independent of
// dotGo: wrapping int32 addition is addition modulo 2^32, and an int64 sum of
// at most a few thousand products (each |p| <= 2^30) cannot itself overflow, so
// truncating the exact sum must equal the wrapping sum. It pins the reference
// rather than trusting it.
func dotOracle(a, b []int16) int32 {
	var s int64
	for i := range min(len(a), len(b)) {
		s += int64(a[i]) * int64(b[i])
	}
	return int32(s) //nolint:gosec // deliberate truncation: this is the wrapping contract
}

// dotLengths sweeps EVERY length from 0 to 64, then a spread of larger ones.
//
// The exhaustive low range is deliberate rather than lazy. The arm64 kernel
// runs an 8-wide block when n mod 16 >= 8 and follows it with an (n mod 8)
// scalar tail, so reaching that block with a short tail needs n mod 16 in
// {10, 11, 13, 14}. A hand-picked list of block boundaries and their immediate
// neighbours never produces those combinations. The all-MinInt16 sweep in
// TestDotProduct_MinInt16 does visit every length, but its operands are
// sign-symmetric ((-32768)*(-32768) == 32768*32768), so it can only catch a
// miscounted element, never a sign, lane, or index error. This list is what the
// non-uniform tests sweep, so it has to be the exhaustive one.
var dotLengths = func() []int {
	lens := make([]int, 0, 75)
	for n := range 65 {
		lens = append(lens, n)
	}
	return append(lens, 100, 127, 128, 129, 240, 255, 256, 257, 300, 1024)
}()

func TestDotProduct(t *testing.T) {
	if got := DotProduct(nil, nil); got != 0 {
		t.Errorf("DotProduct(nil, nil) = %d, want 0", got)
	}
	if got := DotProduct([]int16{1, 2}, nil); got != 0 {
		t.Errorf("DotProduct(a, nil) = %d, want 0", got)
	}

	// A hand-computed case that exceeds int16 but fits int32, pinning the
	// widening independently of the reference. n=2 is below every dispatch
	// threshold, so this exercises dotGo only; the kernels' widening is covered
	// by the dotLengths sweeps below and by the direct kernel tests.
	if got, want := DotProduct([]int16{1000, -2000}, []int16{3000, 4000}), int32(1000*3000+-2000*4000); got != want {
		t.Errorf("DotProduct([1000,-2000],[3000,4000]) = %d, want %d", got, want)
	}

	for _, n := range dotLengths {
		a, b := genI16(n, 8), genI16(n, 9)
		got := DotProduct(a, b)
		if want := dotGo(a, b); got != want {
			t.Errorf("DotProduct n=%d: got %d, want %d (reference)", n, got, want)
		}
		if want := dotOracle(a, b); got != want {
			t.Errorf("DotProduct n=%d: got %d, want %d (int64 oracle)", n, got, want)
		}
	}
}

// TestDotProduct_Clamp covers mismatched operand lengths. This matters beyond
// arithmetic: the kernels read len(a) and len(b) and clamp in assembly, so a
// regression here is an out-of-bounds read, not a wrong number.
func TestDotProduct_Clamp(t *testing.T) {
	if got, want := DotProduct([]int16{1, 2, 3}, []int16{4, 5}), int32(1*4+2*5); got != want {
		t.Errorf("DotProduct(len 3, len 2) = %d, want %d", got, want)
	}
	if got, want := DotProduct([]int16{4, 5}, []int16{1, 2, 3}), int32(4*1+5*2); got != want {
		t.Errorf("DotProduct(len 2, len 3) = %d, want %d", got, want)
	}
	// Long/short pairings across block boundaries: the long side must never be
	// read past the short side's length.
	for _, n := range dotLengths {
		if n == 0 {
			continue
		}
		long := genI16(n+37, 11)
		short := genI16(n, 12)
		if got, want := DotProduct(long, short), dotOracle(long, short); got != want {
			t.Errorf("DotProduct clamp n=%d: got %d, want %d", n, got, want)
		}
		if got, want := DotProduct(short, long), dotOracle(short, long); got != want {
			t.Errorf("DotProduct clamp (swapped) n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestDotProduct_MinInt16 pins the one input that can overflow int32 inside a
// single PMADDWD: a pair of (-32768 * -32768) sums to 2^31 and must wrap to
// MinInt32 rather than saturate to MaxInt32. The length sweep walks the wrap
// through every lane position and into the scalar tail.
//
// The three headline cases below are n=1, 2 and 4, all under every dispatch
// threshold, so they pin the reference's semantics; the sweep and the direct
// kernel tests are what reach PMADDWD and SMLAL. Note also that all-MinInt16
// operands are sign-symmetric, so this test cannot catch a sign-extension or
// lane error: that is TestDotProduct_ExtremeMix and the dotLengths sweeps.
func TestDotProduct_MinInt16(t *testing.T) {
	if got, want := DotProduct([]int16{math.MinInt16}, []int16{math.MinInt16}), int32(1<<30); got != want {
		t.Errorf("DotProduct(MinInt16, MinInt16) = %d, want %d", got, want)
	}
	// Exactly two such products: 2^31 wraps to MinInt32. A saturating
	// accumulator would return MaxInt32 here.
	two := []int16{math.MinInt16, math.MinInt16}
	if got, want := DotProduct(two, two), int32(math.MinInt32); got != want {
		t.Errorf("DotProduct(2x MinInt16) = %d, want %d (saturation would give %d)", got, want, int32(math.MaxInt32))
	}
	// Four such products: 2^32 wraps to exactly 0.
	four := []int16{math.MinInt16, math.MinInt16, math.MinInt16, math.MinInt16}
	if got := DotProduct(four, four); got != 0 {
		t.Errorf("DotProduct(4x MinInt16) = %d, want 0", got)
	}
	for n := 1; n <= 64; n++ {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i], b[i] = math.MinInt16, math.MinInt16
		}
		got := DotProduct(a, b)
		if want := dotGo(a, b); got != want {
			t.Errorf("DotProduct all-MinInt16 n=%d: got %d, want %d (reference)", n, got, want)
		}
		if want := dotOracle(a, b); got != want {
			t.Errorf("DotProduct all-MinInt16 n=%d: got %d, want %d (int64 oracle)", n, got, want)
		}
	}
}

// TestDotProduct_Wraparound drives the accumulator through many wraps with
// same-sign products, so an implementation that clamped anywhere along the
// chain diverges well before the end.
func TestDotProduct_Wraparound(t *testing.T) {
	for _, n := range []int{8, 16, 17, 240, 1000, 5000} {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i], b[i] = 30000, 30000 // 9e8 per product: wraps roughly every 3
		}
		got := DotProduct(a, b)
		if want := dotOracle(a, b); got != want {
			t.Errorf("DotProduct wraparound n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestDotProduct_ExtremeMix mixes the type extremes with ordinary values so the
// wrap lands at irregular offsets rather than aligning with the block size.
func TestDotProduct_ExtremeMix(t *testing.T) {
	for _, n := range dotLengths {
		a, b := genI16(n, 21), genI16(n, 22)
		for i := range a {
			switch i % 5 {
			case 0:
				a[i] = math.MinInt16
			case 1:
				b[i] = math.MinInt16
			case 2:
				a[i], b[i] = math.MaxInt16, math.MinInt16
			}
		}
		if got, want := DotProduct(a, b), dotOracle(a, b); got != want {
			t.Errorf("DotProduct extreme mix n=%d: got %d, want %d", n, got, want)
		}
	}
}

func TestDotProductUnsafe(t *testing.T) {
	for _, n := range dotLengths {
		if n == 0 {
			continue // Unsafe requires non-empty operands.
		}
		a, b := genI16(n, 31), genI16(n, 32)
		if got, want := DotProductUnsafe(a, b), dotGo(a, b); got != want {
			t.Errorf("DotProductUnsafe n=%d: got %d, want %d", n, got, want)
		}
	}
	// Documented to tolerate mismatched lengths by clamping internally.
	if got, want := DotProductUnsafe([]int16{1, 2, 3}, []int16{4, 5}), int32(1*4+2*5); got != want {
		t.Errorf("DotProductUnsafe mismatched = %d, want %d", got, want)
	}
}

// TestDotProduct_UnalignedOperands calls DotProduct on offset subslices, which
// is the shape the motivating use case actually has: sliding correlation
// evaluates DotProduct(x[lag:], h) at every lag, so an operand is rarely
// 16-byte aligned. Every other test here builds operands with make, which is
// aligned, so an aligned-load regression (MOVOU -> MOVOA, VMOVDQU -> VMOVDQA)
// would pass the whole suite and then fault in production. The odd offsets
// matter most: they leave the pointer 2-byte aligned.
func TestDotProduct_UnalignedOperands(t *testing.T) {
	const span = 400
	base, other := genI16(span, 81), genI16(span, 82)
	for _, n := range dotLengths {
		if n > 300 {
			continue // keep off+n inside span
		}
		for off := range 8 {
			a := base[off : off+n]
			b := other[off+1 : off+1+n]
			if got, want := DotProduct(a, b), dotOracle(a, b); got != want {
				t.Errorf("DotProduct unaligned n=%d off=%d: got %d, want %d", n, off, got, want)
			}
		}
	}
}

func TestDotProduct_AllocFree(t *testing.T) {
	a, b := genI16(1024, 41), genI16(1024, 42)
	if n := testing.AllocsPerRun(100, func() { _ = DotProduct(a, b) }); n != 0 {
		t.Errorf("DotProduct allocates %v times per run, want 0", n)
	}
	if n := testing.AllocsPerRun(100, func() { _ = DotProductUnsafe(a, b) }); n != 0 {
		t.Errorf("DotProductUnsafe allocates %v times per run, want 0", n)
	}
}
