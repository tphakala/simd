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
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
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
