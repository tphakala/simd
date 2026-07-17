//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestInterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)

		gotNEON := make([]int32, n*2)
		gotGo := make([]int32, n*2)
		interleave2NEON(gotNEON, a, b)
		interleave2Go(gotGo, a, b)

		for i := range gotGo {
			if gotNEON[i] != gotGo[i] {
				t.Fatalf("n=%d: interleave2NEON[%d] = %d, want %d (Go)", n, i, gotNEON[i], gotGo[i])
			}
		}
	}
}

func TestDeinterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		src := make([]int32, n*2)
		for i := range src {
			src[i] = int32(i) ^ math.MinInt32
		}

		aNEON := make([]int32, n)
		bNEON := make([]int32, n)
		aGo := make([]int32, n)
		bGo := make([]int32, n)
		deinterleave2NEON(aNEON, bNEON, src)
		deinterleave2Go(aGo, bGo, src)

		for i := range aGo {
			if aNEON[i] != aGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON a[%d] = %d, want %d (Go)", n, i, aNEON[i], aGo[i])
			}
			if bNEON[i] != bGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON b[%d] = %d, want %d (Go)", n, i, bNEON[i], bGo[i])
			}
		}
	}
}

// TestInterleave2NEON_NoOverwrite guards the scalar tail: the kernel must not
// write past n*2 output elements when n is not a multiple of the 4-lane block.
func TestInterleave2NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 17
	a := make([]int32, n)
	b := make([]int32, n)
	fillPattern(a, b)
	dst := make([]int32, n*2+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	interleave2NEON(dst[:n*2], a, b)
	for i := n * 2; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("interleave2NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

func TestAddSubNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)
		for _, tc := range []struct {
			name string
			simd func(dst, a, b []int32)
			ref  func(dst, a, b []int32)
		}{
			{"add", addNEON, addGo},
			{"sub", subNEON, subGo},
		} {
			gotNEON := make([]int32, n)
			gotGo := make([]int32, n)
			tc.simd(gotNEON, a, b)
			tc.ref(gotGo, a, b)
			for i := range gotGo {
				if gotNEON[i] != gotGo[i] {
					t.Fatalf("n=%d: %sNEON[%d] = %d, want %d (Go)", n, tc.name, i, gotNEON[i], gotGo[i])
				}
			}
		}
	}
}

// TestNEONKernels_AllocFree asserts each NEON kernel runs allocation-free, the
// repo's zero-allocation contract enforced directly at the kernel boundary
// (the public-API alloc tests cover the dispatch, these cover the kernels).
func TestNEONKernels_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	a := make([]int32, n)
	b := make([]int32, n)
	dst := make([]int32, n)
	checks := []struct {
		name string
		fn   func()
	}{
		{"addNEON", func() { addNEON(dst, a, b) }},
		{"subNEON", func() { subNEON(dst, a, b) }},
		{"minMaxNEON", func() { _, _ = minMaxNEON(a) }},
		{"sumNEON", func() { _ = sumNEON(a) }},
		{"absNEON", func() { absNEON(dst, a) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestSumNEON_ParityWithGo drives the kernel directly, over lengths the
// dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself. The forced
// overflow leg wraps the accumulator in the lanes and in the tail.
func TestSumNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 51)
		if got, want := sumNEON(a), sumGo(a); got != want {
			t.Errorf("sumNEON n=%d: got %d, want %d", n, got, want)
		}
	}
	for n := 1; n <= 16; n++ {
		a := make([]int32, n)
		for i := range a {
			a[i] = math.MaxInt32 - int32(i%2)
		}
		if got, want := sumNEON(a), sumGo(a); got != want {
			t.Errorf("sumNEON overflow n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestSumNEON_NoOverRead is the kernel-direct over-read check: the operand is
// a prefix of a longer allocation whose every element past the prefix is
// poisoned. Zeroed past-slice memory is the identity element of this
// reduction, so only the poison makes an over-read observable. The poison is
// deliberately not an extreme; see sumOverReadPoison for why MinInt32 would
// hide a whole-block over-read here.
func TestSumNEON_NoOverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	backing := make([]int32, 64+8)
	for i := range backing {
		backing[i] = sumOverReadPoison
	}
	for _, n := range []int{1, 3, 4, 5, 7, 8, 9, 11, 16, 17, 33, 64} {
		a := backing[:n]
		for i := range a {
			a[i] = int32(i - 5)
		}
		if got, want := sumNEON(a), sumGo(a); got != want {
			t.Fatalf("sumNEON n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = sumOverReadPoison
		}
	}
}

// TestAbsNEON_ParityWithGo drives the kernel directly across the full sweep;
// the wrap input MinInt32 rides at index 0 of every non-empty case.
func TestAbsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 52)
		if n > 0 {
			a[0] = math.MinInt32
		}
		got := make([]int32, n)
		want := make([]int32, n)
		absNEON(got, a)
		absGo(want, a)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("absNEON n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestTier3Dispatch_ReachesNEON pins the dispatch state Sum and Abs depend
// on. It has to be a white-box check: the kernels are bit-identical to the Go
// references by design, so a dispatcher that silently routed every call to Go
// would pass every parity test in this package. It must not call
// t.Parallel(): it reads package-level dispatch state.
//
// Scope, so the next reader does not over-trust it: this pins the INPUTS the
// dispatcher reads (the feature flag, and thresholds low enough to vectorize),
// not that the dispatcher consults them. It kills a mis-wired flag and an
// out-of-range threshold; it does not kill a dispatch branch deleted outright,
// which leaves the kernel dead while every test here still passes. Closing
// that needs the call to be observable, e.g. a counter behind a build tag.
func TestTier3Dispatch_ReachesNEON(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	if !hasNEON {
		t.Fatal("hasNEON is false though cpu.ARM64.NEON is true: Sum and Abs silently run the Go reference on every call")
	}
	if minNEONSum > 8 || minNEONAbs > 8 {
		t.Fatalf("tier-3 NEON thresholds exceed two vector blocks (Sum %d, Abs %d): the ops would not vectorize at the lengths they were written for",
			minNEONSum, minNEONAbs)
	}
}

func TestMinMaxNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	// A tame in-range body keeps the planted extremes the unique min/max, so a
	// kernel that drops a vector lane or skips the scalar tail is caught: one
	// variant plants the extremes in a mid-block lane and in the tail, the other
	// swaps them, covering both a dropped lane and a dropped tail on both reduces.
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		for _, swap := range []bool{false, true} {
			res := make([]int32, n)
			for i := range res {
				res[i] = int32(i%13) - 6
			}
			mid, tail := int32(math.MinInt32), int32(math.MaxInt32)
			if swap {
				mid, tail = tail, mid
			}
			res[n/2] = mid
			res[n-1] = tail
			gotMin, gotMax := minMaxNEON(res)
			wantMin, wantMax := minMaxGo(res)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxNEON = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}
