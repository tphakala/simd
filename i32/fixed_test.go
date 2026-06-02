package i32

import (
	"math"
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
	src := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4}
	for order := 1; order <= 4; order++ {
		t.Run([]string{"", "Diff1", "Diff2", "Diff3", "Diff4"}[order], func(t *testing.T) {
			dst := make([]int32, len(src))
			diffs[order-1](dst, src)
			want := diffResidualRepeated(src, order)
			for i := 0; i < order; i++ {
				if dst[i] != src[i] {
					t.Errorf("order %d warm-up dst[%d] = %d, want src %d", order, i, dst[i], src[i])
				}
			}
			for n := order; n < len(src); n++ {
				if dst[n] != want[n] {
					t.Errorf("order %d residual dst[%d] = %d, want %d", order, n, dst[n], want[n])
				}
			}
		})
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
