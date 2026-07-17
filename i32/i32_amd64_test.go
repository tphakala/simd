//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestInterleave2AVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)

		gotAVX := make([]int32, n*2)
		gotGo := make([]int32, n*2)
		interleave2AVX(gotAVX, a, b)
		interleave2Go(gotGo, a, b)

		for i := range gotGo {
			if gotAVX[i] != gotGo[i] {
				t.Fatalf("n=%d: interleave2AVX[%d] = %d, want %d (Go)", n, i, gotAVX[i], gotGo[i])
			}
		}
	}
}

func TestDeinterleave2AVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	for _, n := range paritySizes {
		src := make([]int32, n*2)
		for i := range src {
			src[i] = int32(i) ^ math.MinInt32
		}

		aAVX := make([]int32, n)
		bAVX := make([]int32, n)
		aGo := make([]int32, n)
		bGo := make([]int32, n)
		deinterleave2AVX(aAVX, bAVX, src)
		deinterleave2Go(aGo, bGo, src)

		for i := range aGo {
			if aAVX[i] != aGo[i] {
				t.Fatalf("n=%d: deinterleave2AVX a[%d] = %d, want %d (Go)", n, i, aAVX[i], aGo[i])
			}
			if bAVX[i] != bGo[i] {
				t.Fatalf("n=%d: deinterleave2AVX b[%d] = %d, want %d (Go)", n, i, bAVX[i], bGo[i])
			}
		}
	}
}

// TestInterleave2AVX_NoOverwrite guards the scalar tail: the kernel must not
// write past n*2 output elements even when n is not a multiple of the block.
func TestInterleave2AVX_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	const n = 17
	a := make([]int32, n)
	b := make([]int32, n)
	fillPattern(a, b)
	dst := make([]int32, n*2+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	interleave2AVX(dst[:n*2], a, b)
	for i := n * 2; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("interleave2AVX wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

func TestAddSubAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
			{"add", addAVX2, addGo},
			{"sub", subAVX2, subGo},
		} {
			gotAVX := make([]int32, n)
			gotGo := make([]int32, n)
			tc.simd(gotAVX, a, b)
			tc.ref(gotGo, a, b)
			for i := range gotGo {
				if gotAVX[i] != gotGo[i] {
					t.Fatalf("n=%d: %sAVX2[%d] = %d, want %d (Go)", n, tc.name, i, gotAVX[i], gotGo[i])
				}
			}
		}
	}
}

// TestAVX2Kernels_AllocFree asserts each AVX2 kernel runs allocation-free, the
// repo's zero-allocation contract enforced directly at the kernel boundary
// (the public-API alloc tests cover the dispatch, these cover the kernels).
func TestAVX2Kernels_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	a := make([]int32, n)
	b := make([]int32, n)
	dst := make([]int32, n)
	checks := []struct {
		name string
		fn   func()
	}{
		{"addAVX2", func() { addAVX2(dst, a, b) }},
		{"subAVX2", func() { subAVX2(dst, a, b) }},
		{"minMaxAVX2", func() { _, _ = minMaxAVX2(a) }},
		{"sumAVX2", func() { _ = sumAVX2(a) }},
		{"absAVX2", func() { absAVX2(dst, a) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestSumAVX2_ParityWithGo drives the kernel directly, over lengths the
// dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself. The forced
// overflow leg wraps the accumulator in the lanes and in the tail.
func TestSumAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 61)
		if got, want := sumAVX2(a), sumGo(a); got != want {
			t.Errorf("sumAVX2 n=%d: got %d, want %d", n, got, want)
		}
	}
	for n := 1; n <= 24; n++ {
		a := make([]int32, n)
		for i := range a {
			a[i] = math.MaxInt32 - int32(i%2)
		}
		if got, want := sumAVX2(a), sumGo(a); got != want {
			t.Errorf("sumAVX2 overflow n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestSumAVX2_NoOverRead is the kernel-direct over-read check: the operand is
// a prefix of a longer allocation whose every element past the prefix is
// poisoned. Zeroed past-slice memory is the identity element of this
// reduction, so only the poison makes an over-read observable. The poison is
// deliberately not an extreme; see sumOverReadPoison for why MinInt32 would
// hide a whole-block over-read here.
func TestSumAVX2_NoOverRead(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	backing := make([]int32, 64+8)
	for i := range backing {
		backing[i] = sumOverReadPoison
	}
	for _, n := range []int{1, 3, 7, 8, 9, 11, 16, 17, 33, 64} {
		a := backing[:n]
		for i := range a {
			a[i] = int32(i - 5)
		}
		if got, want := sumAVX2(a), sumGo(a); got != want {
			t.Fatalf("sumAVX2 n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = sumOverReadPoison
		}
	}
}

// TestAbsAVX2_ParityWithGo drives the kernel directly across the full sweep;
// the wrap input MinInt32 rides at index 0 of every non-empty case.
func TestAbsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 62)
		if n > 0 {
			a[0] = math.MinInt32
		}
		got := make([]int32, n)
		want := make([]int32, n)
		absAVX2(got, a)
		absGo(want, a)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("absAVX2 n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestTier3Dispatch_ReachesSIMD pins the dispatch state Sum and Abs depend
// on. It has to be a white-box check: the kernels are bit-identical to the Go
// references by design, so a dispatcher that silently routed every call to Go
// would pass every parity test in this package. Below AVX2 the Go reference
// is the intended path. It must not call t.Parallel(): it reads
// package-level dispatch state.
//
// Scope, so the next reader does not over-trust it: this pins the INPUTS the
// dispatcher reads (the feature flag, and thresholds low enough to vectorize),
// not that the dispatcher consults them. It kills a mis-wired flag and an
// out-of-range threshold; it does not kill a dispatch branch deleted outright,
// which leaves the kernel dead while every test here still passes. Closing
// that needs the call to be observable, e.g. a counter behind a build tag.
func TestTier3Dispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2Sum > 16 || minAVX2Abs > 16 {
		t.Fatalf("tier-3 AVX2 thresholds exceed two vector blocks (Sum %d, Abs %d): the ops would not vectorize at the lengths they were written for",
			minAVX2Sum, minAVX2Abs)
	}
}

func TestMinMaxAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	// A tame in-range body keeps the planted extremes the unique min/max, so a
	// kernel that drops a vector lane or skips the scalar tail is caught: one
	// variant plants the extremes in a mid-block lane and in the tail, the other
	// swaps them, covering both a dropped lane and a dropped tail on both reduces.
	for _, n := range paritySizes {
		if n < minAVXElements {
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
			gotMin, gotMax := minMaxAVX2(res)
			wantMin, wantMax := minMaxGo(res)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxAVX2 = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}
