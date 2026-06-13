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
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
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
