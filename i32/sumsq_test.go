package i32

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// Tests for SumSqShiftedQ31, the wrapping int32 sum of pre-shifted Q31 squares
// (libopus compute_band_energies). Two contracts carry the weight: the shift is a
// wrapping int32 SHL32 applied BEFORE the widen, so the square is of the truncated
// value; and the accumulator wraps in int32, so overflow is the main event, not an
// edge case (a saturating build would pass small-value tests and fail these).

// sumSqShiftedQ31Oracle recomputes the reduction in arbitrary precision,
// truncating to a signed 32-bit lane at every stage boundary exactly as the SIMD
// int32 lanes do: SHL32 wraps to int32, the >>31 of the square wraps to int32, and
// the accumulate wraps to int32 each step. It is independent of the int64
// arithmetic in sumSqShiftedQ31Go (big.Int throughout, every wrap modeled), so a
// fault cannot hide by agreeing with the reference alone. big.Int.Rsh is a floor
// (arithmetic) shift, matching Go's signed >>; the square is non-negative so the
// distinction does not even arise for the >>31.
func sumSqShiftedQ31Oracle(a []int32, shift int) int32 {
	sum := big.NewInt(0)
	for _, v := range a {
		x := trunc32Big(new(big.Int).Lsh(big.NewInt(int64(v)), uint(shift)))     // SHL32, wraps to int32
		p := trunc32Big(new(big.Int).Rsh(new(big.Int).Mul(x, x), scaleQ31Shift)) // (x*x) >> 31, wraps to int32
		sum = trunc32Big(new(big.Int).Add(sum, p))                               // wrapping int32 accumulate
	}
	return int32(sum.Int64())
}

// sumSqShiftedQ31Naive is the issue's verbatim definition (v << shift is Go's
// signed left shift). For shift in [0,31] Go's signed << wraps identically to the
// int32(uint32(v)<<s) form in the reference, so this pins that the masking in
// sumSqShiftedQ31Go did not change the arithmetic.
func sumSqShiftedQ31Naive(a []int32, shift int) int32 {
	var sum int32
	for _, v := range a {
		x := v << shift
		sum += int32(int64(x) * int64(x) >> 31)
	}
	return sum
}

// sumSqWidenThenShiftWrong is the WRONG ordering SumSqShiftedQ31 must NOT
// implement: shift the sample in full (int64) precision, so it does not wrap to
// int32 before the square. The discrimination test below pins that the correct
// result diverges from this for shift-induced overflow, so a widen-then-shift
// kernel could never pass silently.
func sumSqWidenThenShiftWrong(a []int32, shift int) int32 {
	sum := big.NewInt(0)
	for _, v := range a {
		x := new(big.Int).Lsh(big.NewInt(int64(v)), uint(shift)) // NO int32 truncation
		p := trunc32Big(new(big.Int).Rsh(new(big.Int).Mul(x, x), scaleQ31Shift))
		sum = trunc32Big(new(big.Int).Add(sum, p))
	}
	return int32(sum.Int64())
}

// sumSqShifts is the shift matrix swept by the parity tests: block-straddling small
// counts, the mid-range, and the [0,31] endpoints where the SHL32 saturates a
// positive sample past the sign bit.
var sumSqShifts = []int{0, 1, 2, 5, 15, 16, 30, 31}

// TestSumSqShiftedQ31Oracle confirms the oracle itself encodes the shift-before-
// widen wrap and the accumulate wrap, so the parity tests rest on a checked
// foundation.
func TestSumSqShiftedQ31Oracle(t *testing.T) {
	cases := []struct {
		a     []int32
		shift int
		want  int32
	}{
		{[]int32{math.MinInt32}, 0, math.MinInt32},    // (2^62)>>31 = 2^31 -> MinInt32
		{[]int32{1}, 31, math.MinInt32},               // 1<<31 = MinInt32, then squared
		{[]int32{math.MinInt32, math.MinInt32}, 0, 0}, // 2*MinInt32 wraps to 0
		{[]int32{math.MaxInt32}, 0, 2147483646},       // (2^31-1)^2 >> 31 = 2^31-2
	}
	for _, c := range cases {
		if got := sumSqShiftedQ31Oracle(c.a, c.shift); got != c.want {
			t.Fatalf("oracle(%v, %d) = %d, want %d", c.a, c.shift, got, c.want)
		}
	}
}

// TestSumSqShiftedQ31 sweeps every tier-3 length across the shift matrix against
// the pure-Go reference, the arbitrary-precision oracle, and the issue's naive
// form, so a fault cannot hide by agreeing with the reference alone. MinInt32 rides
// index 0 and MaxInt32 the last index so the wraps are exercised at every length
// and the scalar tail must be folded in.
func TestSumSqShiftedQ31(t *testing.T) {
	for _, n := range tier3Lengths {
		a := genI32(n, 73)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, s := range sumSqShifts {
			got := SumSqShiftedQ31(a, s)
			if want := sumSqShiftedQ31Go(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 n=%d shift=%d = %d, want %d (reference)", n, s, got, want)
			}
			if want := sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 n=%d shift=%d = %d, want %d (oracle)", n, s, got, want)
			}
			if want := sumSqShiftedQ31Naive(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 n=%d shift=%d = %d, want %d (naive)", n, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31_Random crosses random int32 (organically hitting the full
// range) with the hand-picked specials, at every length that straddles a vector
// block and its scalar tail on both arches, over all 32 shifts.
func TestSumSqShiftedQ31_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	specials := []int32{math.MinInt32, math.MinInt32 + 1, math.MaxInt32, 0, -1, 1, 0x40000000, 0x00018000, 0x40000001}
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000} {
		for trial := range 12 {
			a := make([]int32, n)
			for i := range a {
				if rng.Intn(3) == 0 {
					a[i] = specials[rng.Intn(len(specials))]
				} else {
					a[i] = int32(rng.Uint32())
				}
			}
			s := rng.Intn(32)
			got := SumSqShiftedQ31(a, s)
			if want := sumSqShiftedQ31Go(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 n=%d trial=%d shift=%d = %d, want %d (reference)", n, trial, s, got, want)
			}
			if want := sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 n=%d trial=%d shift=%d = %d, want %d (oracle)", n, trial, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31_Cases pins the hand-computed contracts in isolation, each
// value derived by hand independently of the oracle, spanning the Go path (n below
// the SIMD thresholds) and, at n=11, one 8-wide AVX2 block + tail and two 4-wide
// NEON blocks + tail with the value planted in every lane.
func TestSumSqShiftedQ31_Cases(t *testing.T) {
	cases := []struct {
		name  string
		a     []int32
		shift int
		want  int32
	}{
		{"empty", []int32{}, 0, 0},
		{"nil", nil, 3, 0},
		{"one-shift0", []int32{1}, 0, 0},                                 // 1>>31 = 0
		{"one-shift31", []int32{1}, 31, math.MinInt32},                   // 1<<31 = MinInt32 squared
		{"neg-one-shift0", []int32{-1}, 0, 0},                            // 1>>31 = 0
		{"neg-one-shift31", []int32{-1}, 31, math.MinInt32},              // uint32(-1)<<31 = MinInt32
		{"half-shift0", []int32{0x40000000}, 0, 0x20000000},              // (2^30)^2>>31 = 2^29
		{"half-shift1", []int32{0x40000000}, 1, math.MinInt32},           // 2^30<<1 = MinInt32, squared
		{"two-shift0", []int32{2}, 0, 0},                                 // 4>>31 = 0
		{"max-shift0", []int32{math.MaxInt32}, 0, 2147483646},            // (2^31-1)^2>>31
		{"two-max-shift0", []int32{math.MaxInt32, math.MaxInt32}, 0, -4}, // 2*(2^31-2) wraps
	}
	for _, c := range cases {
		if got := SumSqShiftedQ31(c.a, c.shift); got != c.want {
			t.Errorf("SumSqShiftedQ31(%s) = %d, want %d", c.name, got, c.want)
		}
		if got := sumSqShiftedQ31Oracle(c.a, c.shift); got != c.want {
			t.Errorf("oracle(%s) = %d, want %d", c.name, got, c.want)
		}
	}
}

// TestSumSqShiftedQ31_ShiftBeforeWiden is the correctness crux: the shift is a
// wrapping int32 SHL32 applied before the widening square, so a shifted sample that
// overflows int32 wraps and the square is of the wrapped value. It runs at lengths
// that force the Go path, the vector body and the tail, asserts the dispatched
// result matches the correct oracle, AND asserts that oracle diverges from the
// widen-then-shift ordering for these inputs, so a kernel that shifted after
// widening could not pass silently.
func TestSumSqShiftedQ31_ShiftBeforeWiden(t *testing.T) {
	type in struct {
		v     int32
		shift int
	}
	// Inputs where v<<shift overflows int32, so the shift-before-widen wrap matters.
	// The dispatched result must match the correct oracle at Go-path, vector-body and
	// tail lengths.
	inputs := []in{
		{0x00018000, 16}, // truncates to MinInt32, term = MinInt32
		{0x00010000, 16}, // truncates to 0, term = 0
		{0x00018001, 16},
		{0x40000001, 1}, // large positive shifted past the sign bit -> negative int32
		{0x12345678, 8},
		{0x00007FFF, 17},
		{3, 30}, // 3<<30 wraps to a negative int32
	}
	for _, in := range inputs {
		for _, n := range []int{1, 4, 8, 11} {
			a := make([]int32, n)
			for i := range a {
				a[i] = in.v
			}
			got := SumSqShiftedQ31(a, in.shift)
			if want := sumSqShiftedQ31Oracle(a, in.shift); got != want {
				t.Fatalf("SumSqShiftedQ31(v=%#x shift=%d n=%d) = %d, want %d (oracle)", in.v, in.shift, n, got, want)
			}
		}
	}
	// The whole point of shifting before the widen: for these inputs the correct
	// result diverges from the widen-then-shift ordering, so a kernel that widened
	// first could not have passed the parity above. Confirm each is genuinely
	// discriminating and that SumSqShiftedQ31 follows the correct ordering.
	discriminating := []in{{0x00018001, 16}, {0x40000001, 1}, {0x12345678, 8}, {0x00007FFF, 17}}
	for _, in := range discriminating {
		a := []int32{in.v}
		correct := sumSqShiftedQ31Oracle(a, in.shift)
		wrong := sumSqWidenThenShiftWrong(a, in.shift)
		if correct == wrong {
			t.Fatalf("case v=%#x shift=%d is not discriminating (correct==wrong==%d)", in.v, in.shift, correct)
		}
		if got := SumSqShiftedQ31(a, in.shift); got != correct {
			t.Fatalf("SumSqShiftedQ31(v=%#x shift=%d) = %d, want %d (shift-before-widen); widen-then-shift would give %d", in.v, in.shift, got, correct, wrong)
		}
	}
}

// TestSumSqShiftedQ31_Wraparound drives the accumulator through many wraps, so a
// build that clamped anywhere along the chain diverges. All-MinInt32-derived terms
// alternate MinInt32/0 by count parity; MaxInt32 terms accumulate an odd stride.
func TestSumSqShiftedQ31_Wraparound(t *testing.T) {
	// k copies of a term equal to MinInt32 (1 shifted to the sign bit, then squared):
	// the sum alternates MinInt32 (odd k) and 0 (even k).
	for k := 1; k <= 9; k++ {
		a := make([]int32, k)
		for i := range a {
			a[i] = 1
		}
		want := int32(0)
		if k%2 == 1 {
			want = math.MinInt32
		}
		if got := SumSqShiftedQ31(a, 31); got != want {
			t.Fatalf("SumSqShiftedQ31 k=%d x [1] shift=31 = %d, want %d", k, got, want)
		}
		if o := sumSqShiftedQ31Oracle(a, 31); o != want {
			t.Fatalf("oracle k=%d = %d, want %d", k, o, want)
		}
	}
	// k copies of MaxInt32 at shift 0: each term is 2^31-2, accumulating with wrap.
	for _, n := range []int{2, 3, 8, 11, 25, 1003} {
		a := make([]int32, n)
		for i := range a {
			a[i] = math.MaxInt32
		}
		if got, want := SumSqShiftedQ31(a, 0), sumSqShiftedQ31Oracle(a, 0); got != want {
			t.Fatalf("SumSqShiftedQ31 all-MaxInt32 n=%d = %d, want %d", n, got, want)
		}
	}
}

// TestSumSqShiftedQ31_PlantedTerm plants a value with a distinctive nonzero term at
// every position over an all-zero body (whose term is 0), so a kernel that drops a
// vector lane or skips the scalar tail loses the planted term where it lives and is
// caught. n=11 forces one AVX2 block + tail and two NEON blocks + tail.
func TestSumSqShiftedQ31_PlantedTerm(t *testing.T) {
	const n = 11
	const planted = int32(0x2BADF00D)
	for _, s := range []int{0, 1, 7} {
		want := sumSqShiftedQ31Oracle([]int32{planted}, s) // one term over a zero body
		for pos := range n {
			a := make([]int32, n)
			a[pos] = planted
			if got := SumSqShiftedQ31(a, s); got != want {
				t.Fatalf("SumSqShiftedQ31 planted pos=%d shift=%d = %d, want %d", pos, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31_Unaligned sweeps every element offset so neither an
// aligned-load substitution nor an off-by-one block boundary can survive. The
// driving extremes ride the ends so the head block and the scalar tail both stay
// load-bearing.
func TestSumSqShiftedQ31_Unaligned(t *testing.T) {
	const span = 300
	backing := genI32(span, 33)
	for _, n := range []int{4, 5, 7, 8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			a := backing[off+1 : off+1+n]
			first, last := a[0], a[n-1]
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
			for _, s := range []int{0, 3, 16} {
				got := SumSqShiftedQ31(a, s)
				if want := sumSqShiftedQ31Oracle(a, s); got != want {
					t.Fatalf("SumSqShiftedQ31 unaligned n=%d off=%d shift=%d = %d, want %d", n, off, s, got, want)
				}
			}
			a[0], a[n-1] = first, last
		}
	}
}

// TestSumSqShiftedQ31_ShiftRange asserts the public entry point rejects out-of-range
// shift counts (which would diverge across backends) and accepts the [0,31]
// endpoints. Validation runs before the length check, so it fires even on a nil
// slice.
func TestSumSqShiftedQ31_ShiftRange(t *testing.T) {
	a := make([]int32, 8)
	bad := []struct {
		name  string
		shift int
		nilIn bool
	}{
		{"shift<0", -1, false},
		{"shift>31", 32, false},
		{"shift=40", 40, false},
		{"invalid on empty", 32, true},
	}
	for _, c := range bad {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("SumSqShiftedQ31(shift=%d) did not panic", c.shift)
				}
			}()
			in := a
			if c.nilIn {
				in = nil // validation must precede the n==0 short-circuit
			}
			_ = SumSqShiftedQ31(in, c.shift)
		})
	}
	// The [0,31] endpoints are valid and must not panic.
	for _, s := range []int{0, 31} {
		_ = SumSqShiftedQ31(a, s)
	}
}

// TestSumSqShiftedQ31_AllocFree declares the buffer INSIDE the measured closure so
// only allocations forced by SumSqShiftedQ31 itself are counted.
func TestSumSqShiftedQ31_AllocFree(t *testing.T) {
	if got := testing.AllocsPerRun(50, func() {
		var a [1000]int32
		_ = SumSqShiftedQ31(a[:], 5)
	}); got != 0 {
		t.Errorf("SumSqShiftedQ31 forces %v caller allocations per run, want 0", got)
	}
}

// sumSqLenSeeds seeds raw byte buffers whose int32 element counts bracket the
// 4/8-lane block boundaries, each paired with a shift, plus adversarial all-same
// buffers that force the accumulator wrap.
func sumSqLenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for i, n := range lens {
		raw := make([]byte, n*4)
		for j := range raw {
			raw[j] = byte(j*37 + 11)
		}
		f.Add(raw, uint8(i%32))
	}
	// Adversarial: all-MinInt32 (shift 0), all-MaxInt32 (shift 0), all-1 (shift 31,
	// each term MinInt32), all-0x00018000 (shift 16, each term MinInt32), at counts
	// that straddle both block widths.
	fill := func(n int, v uint32) []byte {
		raw := make([]byte, n*4)
		for i := range n {
			raw[i*4] = byte(v)
			raw[i*4+1] = byte(v >> 8)
			raw[i*4+2] = byte(v >> 16)
			raw[i*4+3] = byte(v >> 24)
		}
		return raw
	}
	for _, n := range []int{8, 9, 16, 17, 44} {
		f.Add(fill(n, 0x80000000), uint8(0))
		f.Add(fill(n, 0x7FFFFFFF), uint8(0))
		f.Add(fill(n, 0x00000001), uint8(31))
		f.Add(fill(n, 0x00018000), uint8(16))
	}
}

// FuzzSumSqShiftedQ31 differentially fuzzes the dispatched SumSqShiftedQ31 against
// the pure-Go reference and the arbitrary-precision oracle over arbitrary int32
// samples and a shift masked into [0,31], so tail handling and every wrap are
// explored past the hand-picked seeds.
func FuzzSumSqShiftedQ31(f *testing.F) {
	sumSqLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte, shift uint8) {
		s := int(shift & 31)
		a := i32sFromBits(raw)
		got := SumSqShiftedQ31(a, s)
		if want := sumSqShiftedQ31Go(a, s); got != want {
			t.Fatalf("SumSqShiftedQ31 = %d, want %d (reference, len=%d, shift=%d)", got, want, len(a), s)
		}
		if want := sumSqShiftedQ31Oracle(a, s); got != want {
			t.Fatalf("SumSqShiftedQ31 = %d, want %d (oracle, len=%d, shift=%d)", got, want, len(a), s)
		}
	})
}
