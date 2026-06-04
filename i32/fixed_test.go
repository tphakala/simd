package i32

import (
	"math"
	"math/rand"
	"testing"
)

// Tests for the fixed-predictor encode differences Diff1..Diff4.
//
// DiffK(dst, src) writes the order-K forward finite difference: the first K
// entries are the verbatim warm-up (dst[i]=src[i] for i<K) and dst[n] for n>=K
// is the residual the FLAC fixed predictor of order K produces. The residual
// coefficients are the signed binomials (-1)^j * C(K,j).

// diffResidualRepeated computes the order-K residual by applying the first
// difference K times (holding index 0 at the boundary). This is an algorithm
// independent of the binomial-coefficient form the package uses, so the two
// agreeing at n>=K is a meaningful cross-check rather than a restatement.
func diffResidualRepeated(src []int32, order int) []int32 {
	cur := append([]int32(nil), src...)
	for range order {
		next := make([]int32, len(cur))
		for i := range cur {
			if i == 0 {
				next[i] = cur[i]
				continue
			}
			next[i] = cur[i] - cur[i-1]
		}
		cur = next
	}
	return cur
}

func TestDiffOrders(t *testing.T) {
	diffs := []func(dst, src []int32){Diff1, Diff2, Diff3, Diff4}
	full := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4}
	// Two lengths so both the dispatched SIMD path (len >= block threshold) and
	// the pure-Go fallback (len < threshold) are exercised through the public API.
	for _, src := range [][]int32{full, full[:6]} {
		for order := 1; order <= 4; order++ {
			t.Run([]string{"", "Diff1", "Diff2", "Diff3", "Diff4"}[order], func(t *testing.T) {
				dst := make([]int32, len(src))
				diffs[order-1](dst, src)
				want := diffResidualRepeated(src, order)
				for i := 0; i < order; i++ {
					if dst[i] != src[i] {
						t.Errorf("len %d order %d warm-up dst[%d] = %d, want src %d", len(src), order, i, dst[i], src[i])
					}
				}
				for n := order; n < len(src); n++ {
					if dst[n] != want[n] {
						t.Errorf("len %d order %d residual dst[%d] = %d, want %d", len(src), order, n, dst[n], want[n])
					}
				}
			})
		}
	}
}

// TestDiff1Simple checks the order-1 difference against a hand-computed result.
func TestDiff1Simple(t *testing.T) {
	src := []int32{10, 13, 13, 8, 20}
	dst := make([]int32, len(src))
	Diff1(dst, src)
	want := []int32{10, 3, 0, -5, 12} // dst[0]=src[0] warm-up, then s[n]-s[n-1]
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("Diff1[%d] = %d, want %d", i, dst[i], w)
		}
	}
}

// TestDiffWraps verifies int32 wraparound matches the SIMD kernels at the type
// extremes (the residual can exceed the source range).
func TestDiffWraps(t *testing.T) {
	src := []int32{math.MinInt32, math.MaxInt32, math.MinInt32, math.MaxInt32, math.MinInt32, 0}
	dst := make([]int32, len(src))
	Diff1(dst, src)
	// dst[1] = MaxInt32 - MinInt32 = -1 (wraps)
	if dst[1] != -1 {
		t.Errorf("Diff1 wrap dst[1] = %d, want -1", dst[1])
	}
}

func TestDiff_Clamp(t *testing.T) {
	src := []int32{1, 2, 4, 7, 11}
	dst := make([]int32, 100)
	Diff1(dst, src)
	want := []int32{1, 1, 2, 3, 4}
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("Diff1 clamp dst[%d] = %d, want %d", i, dst[i], w)
		}
	}
	for i := len(want); i < len(dst); i++ {
		if dst[i] != 0 {
			t.Errorf("Diff1 wrote past clamp at dst[%d] = %d, want untouched 0", i, dst[i])
		}
	}
}

// TestDiff_ShortInput checks inputs shorter than the order: with fewer than K
// samples there is no residual, only warm-up, and nothing should panic.
func TestDiff_ShortInput(t *testing.T) {
	src := []int32{42, 7}
	dst := make([]int32, len(src))
	Diff4(dst, src) // order 4 but only 2 samples
	if dst[0] != 42 || dst[1] != 7 {
		t.Errorf("Diff4 short input = %v, want [42 7] warm-up", dst)
	}
}

func TestDiff_Empty(t *testing.T) {
	Diff1(nil, nil)
	Diff2([]int32{}, []int32{})
	Diff3(nil, nil)
	Diff4(nil, nil)
	dst := []int32{99}
	Diff1(dst, nil)
	if dst[0] != 99 {
		t.Errorf("Diff1 wrote on empty input: %v", dst)
	}
}

func TestDiff_AllocFree(t *testing.T) {
	src := make([]int32, 1024)
	dst := make([]int32, 1024)
	diffs := map[string]func(dst, src []int32){"Diff1": Diff1, "Diff2": Diff2, "Diff3": Diff3, "Diff4": Diff4}
	for name, fn := range diffs {
		if got := testing.AllocsPerRun(100, func() { fn(dst, src) }); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", name, got)
		}
	}
}

// Tests for FixedAbsSums, the five fixed-predictor residual abs-sums.
//
// sums[order] = Σ_{i>=order} |e_order[i]| where e_order is the order-th forward
// finite difference of src (orders 0..4), computed in int64 and excluding the
// first order warm-up samples.

// fixedAbsSumsOracle is an independent implementation that derives each order's
// finite difference by repeated first-differencing in int64 (the algorithm
// diffResidualRepeated uses, widened), then sums |diff| over the non-warm-up
// range [order, n). Written separately from fixedAbsSumsGo so the two agreeing
// is a real cross-check rather than a restatement.
func fixedAbsSumsOracle(src []int32) [5]uint64 {
	cur := make([]int64, len(src))
	for i, v := range src {
		cur[i] = int64(v)
	}
	abs := func(v int64) uint64 {
		if v < 0 {
			return uint64(-v)
		}
		return uint64(v)
	}
	var sums [5]uint64
	for order := 0; order <= 4; order++ {
		for i := order; i < len(cur); i++ {
			sums[order] += abs(cur[i])
		}
		if order < 4 {
			next := make([]int64, len(cur))
			for i := len(cur) - 1; i >= 1; i-- {
				next[i] = cur[i] - cur[i-1] // next[0] is warm-up, never read for order+1
			}
			cur = next
		}
	}
	return sums
}

// TestFixedAbsSumsSmall checks a hand-computed case including the warm-up
// exclusions for orders that have no full window yet.
func TestFixedAbsSumsSmall(t *testing.T) {
	src := []int32{3, 7, 2}
	// order0: 3+7+2 = 12
	// order1 (i>=1): |7-3| + |2-7| = 4 + 5 = 9
	// order2 (i>=2): |(2-7)-(7-3)| = |-9| = 9
	// order3,4: no terms (n<4, n<5) -> 0
	var got [5]uint64
	FixedAbsSums(src, &got)
	want := [5]uint64{12, 9, 9, 0, 0}
	if got != want {
		t.Fatalf("FixedAbsSums small = %v, want %v", got, want)
	}
}

func TestFixedAbsSumsMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 1024, 1025} {
		src := make([]int32, n)
		for i := range src {
			src[i] = int32(rng.Uint32())
		}
		var got [5]uint64
		FixedAbsSums(src, &got)
		want := fixedAbsSumsOracle(src)
		if got != want {
			t.Fatalf("n=%d FixedAbsSums = %v, want %v", n, got, want)
		}
	}
}

// TestFixedAbsSumsExtremes drives the int64 width: MinInt32/MaxInt32 differences
// reach ~2^35, so an int32-wrapping cascade would diverge from the reference.
func TestFixedAbsSumsExtremes(t *testing.T) {
	src := []int32{math.MinInt32, math.MaxInt32, math.MinInt32, math.MaxInt32, math.MinInt32, math.MaxInt32, 0, -1, 1, math.MinInt32, math.MaxInt32, 0}
	var got [5]uint64
	FixedAbsSums(src, &got)
	want := fixedAbsSumsOracle(src)
	if got != want {
		t.Fatalf("FixedAbsSums extremes = %v, want %v", got, want)
	}
}

// TestFixedAbsSumsOverwrites confirms sums is fully written, not accumulated into.
func TestFixedAbsSumsOverwrites(t *testing.T) {
	src := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	got := [5]uint64{999, 999, 999, 999, 999}
	FixedAbsSums(src, &got)
	want := fixedAbsSumsOracle(src)
	if got != want {
		t.Fatalf("FixedAbsSums did not overwrite: %v, want %v", got, want)
	}
}

func TestFixedAbsSumsEmpty(t *testing.T) {
	got := [5]uint64{7, 7, 7, 7, 7}
	FixedAbsSums(nil, &got)
	if got != ([5]uint64{}) {
		t.Errorf("FixedAbsSums(nil) = %v, want all zero", got)
	}
}

func TestFixedAbsSumsAllocFree(t *testing.T) {
	src := make([]int32, 1024)
	for i := range src {
		src[i] = int32(i*7 - 3)
	}
	var sums [5]uint64
	if got := testing.AllocsPerRun(100, func() { FixedAbsSums(src, &sums) }); got != 0 {
		t.Errorf("FixedAbsSums allocated %v times per run, want 0", got)
	}
}
