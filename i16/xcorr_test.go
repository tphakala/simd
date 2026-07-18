package i16

import (
	"math"
	"testing"
)

// Tests for XCorr.
//
// The defining property is that XCorr is DotProduct evaluated at every lag:
// dst[k] must equal DotProduct(x, y[k:k+len(x))) exactly, for every k. That is
// the invariant most of these tests assert, because it is stronger than
// checking against a second hand-written loop and it ties the two primitives
// together: a kernel that drifted from the dot product would be caught even if
// its own reference drifted with it.
//
// The lag-blocking is where the bugs live. The SIMD path evaluates 4 lags per
// call and finishes the remainder with the dot kernel, so lag counts that are
// not multiples of 4 exercise the seam, and each lag reads y at a different
// offset (only one of which can be even-aligned).

// xcorrOracle computes each lag with the independent int64 oracle rather than
// with dotGo, so a fault in the dot reference cannot hide by being consistent
// with itself.
//
// It pins the per-lag ARITHMETIC only. It calls the production xcorrLags, so a
// lag-count bug would propagate into the expectation rather than being caught.
// The lag count is pinned separately by the tests that recompute it inline
// (TestXCorr_MatchesDotProductAtEveryLag, TestXCorr_ClampLeavesTailUntouched).
func xcorrOracle(dst []int32, x, y []int16) []int32 {
	out := make([]int32, len(dst))
	copy(out, dst)
	for k := range xcorrLags(dst, x, y) {
		out[k] = dotOracle(x, y[k:])
	}
	return out
}

func equalI32(t *testing.T, what string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", what, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: dst[%d] = %d, want %d (len=%d)", what, i, got[i], want[i], len(got))
		}
	}
}

// TestXCorr_MatchesDotProductAtEveryLag is the core property test. The lag
// counts deliberately straddle the 4-lag block boundary in both directions, and
// the x lengths straddle the 8- and 16-wide kernel bodies plus their tails.
func TestXCorr_MatchesDotProductAtEveryLag(t *testing.T) {
	// 12/13/14 run BOTH the 8- and 4-wide AVX2 blocks (n%16 in 12..15) with a
	// 0/1/2-element scalar tail; 20/21/22 run the 4-wide block only (n%8 in 4..6)
	// plus a 16-wide iteration. Together with 7/15/23/31 (residue 7) they cover
	// every 4-wide-block residue combination (#150).
	for _, xn := range []int{1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 31, 32, 33, 64, 240} {
		for _, lags := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 61} {
			x := genI16(xn, 91)
			y := genI16(xn+lags-1, 92)
			dst := make([]int32, lags)
			XCorr(dst, x, y)
			for k := range lags {
				if got, want := dst[k], DotProduct(x, y[k:k+xn]); got != want {
					t.Fatalf("XCorr(xn=%d, lags=%d): dst[%d] = %d, want DotProduct = %d", xn, lags, k, got, want)
				}
			}
		}
	}
}

// TestXCorr_ParityWithReference checks the dispatched path against both the Go
// reference and the independent int64 oracle across the same grid.
func TestXCorr_ParityWithReference(t *testing.T) {
	for _, xn := range []int{1, 8, 9, 16, 17, 33, 64, 240} {
		for _, lags := range []int{1, 3, 4, 5, 8, 11, 16, 61} {
			x := genI16(xn, 93)
			y := genI16(xn+lags+5, 94)
			dst := make([]int32, lags)
			ref := make([]int32, lags)
			XCorr(dst, x, y)
			xcorrGo(ref, x, y)
			equalI32(t, "XCorr vs reference", dst, ref)
			equalI32(t, "XCorr vs oracle", dst, xcorrOracle(make([]int32, lags), x, y))
		}
	}
}

// TestXCorr_ClampLeavesTailUntouched pins the documented contract: only lags
// with a full window are computed, and dst beyond that keeps its prior value.
//
// Note this case has m=3, below xcorrLagBlock, so it exercises the remainder
// path only and never calls the 4-lag kernel. TestXCorr_KernelHonoursDstBounds
// is the one that pins the kernel's own dst writes.
func TestXCorr_ClampLeavesTailUntouched(t *testing.T) {
	const sentinel = int32(-999111)
	x := genI16(16, 95)
	// y holds exactly 3 full windows worth of lags (16+2 elements -> lags 0..2).
	y := genI16(18, 96)
	dst := make([]int32, 10)
	for i := range dst {
		dst[i] = sentinel
	}
	XCorr(dst, x, y)

	wantLags := len(y) - len(x) + 1 // 3
	for k := range wantLags {
		if got, want := dst[k], DotProduct(x, y[k:k+len(x)]); got != want {
			t.Errorf("dst[%d] = %d, want %d", k, got, want)
		}
	}
	for k := wantLags; k < len(dst); k++ {
		if dst[k] != sentinel {
			t.Errorf("dst[%d] = %d, want the sentinel %d left untouched (lag has no full window)", k, dst[k], sentinel)
		}
	}
}

// TestXCorr_Degenerate covers the inputs that compute nothing. Each must leave
// dst entirely untouched rather than zeroing or panicking.
func TestXCorr_Degenerate(t *testing.T) {
	const sentinel = int32(4242)
	cases := []struct {
		name string
		x, y []int16
	}{
		{"empty x", nil, genI16(8, 97)},
		{"empty y", genI16(8, 97), nil},
		{"y shorter than x", genI16(8, 97), genI16(7, 98)},
		{"both empty", nil, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dst := []int32{sentinel, sentinel, sentinel}
			XCorr(dst, c.x, c.y)
			for i, v := range dst {
				if v != sentinel {
					t.Errorf("dst[%d] = %d, want untouched sentinel %d", i, v, sentinel)
				}
			}
		})
	}
	// Empty dst must not panic.
	XCorr(nil, genI16(8, 97), genI16(16, 98))
	XCorr([]int32{}, genI16(8, 97), genI16(16, 98))
}

// TestXCorr_ExactWindow is the boundary the clamp math turns on: y is exactly
// len(x) long, so lag 0 is the only one with a full window.
func TestXCorr_ExactWindow(t *testing.T) {
	const sentinel = int32(7)
	x := genI16(16, 99)
	y := genI16(16, 100)
	dst := []int32{sentinel, sentinel, sentinel, sentinel, sentinel}
	XCorr(dst, x, y)
	if got, want := dst[0], DotProduct(x, y); got != want {
		t.Errorf("dst[0] = %d, want %d", got, want)
	}
	for k := 1; k < len(dst); k++ {
		if dst[k] != sentinel {
			t.Errorf("dst[%d] = %d, want untouched: only lag 0 has a full window", k, dst[k])
		}
	}
}

// TestXCorr_MinInt16 drives the int32 accumulator past its range at every lag.
// Note all-MinInt16 operands are sign-symmetric, so this catches a miscounted
// element but NOT a sign or lane error; TestXCorr_MatchesDotProductAtEveryLag
// with genI16 data is what covers those.
//
// The lengths must keep reaching every kernel path that reorders the sum, since
// wrapping is the property that makes reordering legal at all. 24 and 25 are
// here for that: XCorr routes len(x) >= 16 to AVX2, so the residues mod 16 must
// include some >= 8 or the amd64 8-wide blocks never run under forced overflow.
// {8,9,16,17,33} alone left AVX2 seeing only residues {0,1,1}. Both blocks sum
// their remainder BEFORE the 16-wide body, which is the ordering that needs the
// wrapping: xcorr4AVX2's, and dotAVX2's, which the lags that are not a multiple
// of xcorrLagBlock reach through dotI16.
func TestXCorr_MinInt16(t *testing.T) {
	for _, xn := range []int{8, 9, 16, 17, 24, 25, 33} {
		for _, lags := range []int{1, 4, 5, 8, 9} {
			x := make([]int16, xn)
			y := make([]int16, xn+lags-1)
			for i := range x {
				x[i] = math.MinInt16
			}
			for i := range y {
				y[i] = math.MinInt16
			}
			dst := make([]int32, lags)
			XCorr(dst, x, y)
			equalI32(t, "XCorr all-MinInt16", dst, xcorrOracle(make([]int32, lags), x, y))
		}
	}
}

// TestXCorr_UnalignedOperands is not an edge case for this op, it is the normal
// shape: lag k reads y at element offset k, so at most one lag in four can be
// 16-byte aligned and the odd lags are only 2-byte aligned. An aligned-load
// regression in the kernels would fault here.
func TestXCorr_UnalignedOperands(t *testing.T) {
	base := genI16(600, 101)
	for _, xn := range []int{8, 16, 17, 32} {
		for off := range 8 {
			x := base[off : off+xn]
			y := base[off+3 : off+3+xn+20]
			dst := make([]int32, 20)
			XCorr(dst, x, y)
			for k := range 20 {
				if got, want := dst[k], DotProduct(x, y[k:k+xn]); got != want {
					t.Fatalf("XCorr unaligned xn=%d off=%d: dst[%d] = %d, want %d", xn, off, k, got, want)
				}
			}
		}
	}
}

// TestXCorrGo_DegenerateDirect exercises the reference and its lag count with
// unguarded inputs, which the public API never delivers: XCorr's own guard
// rejects them first, so xcorrLags' len(x)==0 / len(y)<len(x) branch is
// unreachable through XCorr and went untested (codecov flagged it at 66.7%).
//
// The branch is not dead code. xcorrGo and xcorrLags are called DIRECTLY by
// the benchmarks, by xcorrOracle, and by the non-SIMD dispatch path, none of
// which re-check. This pins that contract rather than leaving it to the guard
// two layers up.
func TestXCorrGo_DegenerateDirect(t *testing.T) {
	const sentinel = int32(-5150)
	cases := []struct {
		name string
		x, y []int16
	}{
		{"empty x", nil, genI16(8, 211)},
		{"empty y", genI16(8, 211), nil},
		{"y shorter than x", genI16(8, 211), genI16(7, 212)},
		{"both empty", nil, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dst := []int32{sentinel, sentinel, sentinel}
			if got := xcorrLags(dst, c.x, c.y); got != 0 {
				t.Errorf("xcorrLags = %d, want 0", got)
			}
			xcorrGo(dst, c.x, c.y) // must not panic
			for i, v := range dst {
				if v != sentinel {
					t.Errorf("xcorrGo wrote dst[%d] = %d, want untouched sentinel %d", i, v, sentinel)
				}
			}
		})
	}
}

// TestXCorr_KernelHonoursDstBounds forces the 4-lag kernel to run (m=8, two
// full blocks, no remainder) while giving dst room past m, so a kernel that
// wrote a fifth word per block would corrupt dst[8:].
//
// The sentinel tests above cannot catch that: each has m < xcorrLagBlock, so
// the block loop never executes and only the scalar remainder path runs. Every
// other test that does reach the kernel passes dst with len(dst) == m exactly,
// which leaves no room for an over-write to land in. This is the only test
// where a kernel dst over-write is observable.
func TestXCorr_KernelHonoursDstBounds(t *testing.T) {
	const sentinel = int32(-31337)
	for _, xn := range []int{8, 16, 17, 32} {
		x := genI16(xn, 205)
		y := genI16(xn+7, 206) // m = 8: exactly two kernel blocks
		dst := make([]int32, 12)
		for i := range dst {
			dst[i] = sentinel
		}
		XCorr(dst, x, y)
		for k := range 8 {
			if got, want := dst[k], dotOracle(x, y[k:]); got != want {
				t.Errorf("xn=%d: dst[%d] = %d, want %d", xn, k, got, want)
			}
		}
		for k := 8; k < len(dst); k++ {
			if dst[k] != sentinel {
				t.Errorf("xn=%d: dst[%d] = %d, want untouched sentinel %d (kernel wrote past its block)", xn, k, dst[k], sentinel)
			}
		}
	}
}

// TestXCorr_AllocFree pins the zero-allocation contract from the CALLER's side.
// The buffers are declared INSIDE the measured closure deliberately: hoisting
// them out (the obvious way to write this) measures only XCorr's own
// allocations and passes even when XCorr leaks its parameters, forcing every
// caller to heap-allocate. That regression is exactly what a shared
// higher-order dispatcher reintroduces, since escape analysis cannot see
// through an indirect call.
func TestXCorr_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var xa [240]int16
		var ya [600]int16
		var da [64]int32
		XCorr(da[:], xa[:], ya[:])
	}); n != 0 {
		t.Errorf("XCorr forces %v caller allocations per run, want 0", n)
	}
}
