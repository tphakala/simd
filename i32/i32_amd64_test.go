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

// fillDiffSrc fills src with values that exercise the sign bit and force the
// residual to wrap int32 at the extremes, so a kernel that handled overflow
// differently than Go would be caught.
func fillDiffSrc(src []int32) {
	for i := range src {
		src[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(src) > 1 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
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
			name   string
			simd   func(dst, a, b []int32)
			ref    func(dst, a, b []int32)
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

func TestMidSideAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		left := make([]int32, n)
		right := make([]int32, n)
		fillPattern(left, right)

		midAVX := make([]int32, n)
		sideAVX := make([]int32, n)
		midGo := make([]int32, n)
		sideGo := make([]int32, n)
		midSideEncodeAVX2(midAVX, sideAVX, left, right)
		midSideEncodeGo(midGo, sideGo, left, right)
		for i := range midGo {
			if midAVX[i] != midGo[i] || sideAVX[i] != sideGo[i] {
				t.Fatalf("n=%d: midSideEncodeAVX2[%d] = (%d,%d), want (%d,%d)", n, i, midAVX[i], sideAVX[i], midGo[i], sideGo[i])
			}
		}

		lAVX := make([]int32, n)
		rAVX := make([]int32, n)
		lGo := make([]int32, n)
		rGo := make([]int32, n)
		midSideDecodeAVX2(lAVX, rAVX, midAVX, sideAVX)
		midSideDecodeGo(lGo, rGo, midGo, sideGo)
		for i := range lGo {
			if lAVX[i] != lGo[i] || rAVX[i] != rGo[i] {
				t.Fatalf("n=%d: midSideDecodeAVX2[%d] = (%d,%d), want (%d,%d)", n, i, lAVX[i], rAVX[i], lGo[i], rGo[i])
			}
		}
	}
}

func TestDiffAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	kernels := []struct {
		name string
		simd func(dst, src []int32)
		ref  func(dst, src []int32)
	}{
		{"diff1", diff1AVX2, diff1Go},
		{"diff2", diff2AVX2, diff2Go},
		{"diff3", diff3AVX2, diff3Go},
		{"diff4", diff4AVX2, diff4Go},
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // the kernel's warm-up reads assume len >= order; the
			// dispatch only calls it for len >= minAVXElements, routing
			// shorter inputs (and len < order) to the Go path.
		}
		src := make([]int32, n)
		fillDiffSrc(src)
		for _, k := range kernels {
			gotAVX := make([]int32, n)
			gotGo := make([]int32, n)
			k.simd(gotAVX, src)
			k.ref(gotGo, src)
			for i := range gotGo {
				if gotAVX[i] != gotGo[i] {
					t.Fatalf("n=%d: %sAVX2[%d] = %d, want %d (Go)", n, k.name, i, gotAVX[i], gotGo[i])
				}
			}
		}
	}
}

// TestDiff1AVX2_NoOverwrite guards the scalar tail: the kernel must write exactly
// n elements and not run past the end when n is not a multiple of the block.
func TestDiff1AVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 17
	src := make([]int32, n)
	fillDiffSrc(src)
	dst := make([]int32, n+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	diff1AVX2(dst[:n], src)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("diff1AVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestMidSideEncodeAVX2_NoOverwrite guards both output tails.
func TestMidSideEncodeAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 17
	left := make([]int32, n)
	right := make([]int32, n)
	fillPattern(left, right)
	mid := make([]int32, n+4)
	side := make([]int32, n+4)
	for i := range mid {
		mid[i] = math.MaxInt32
		side[i] = math.MaxInt32
	}
	midSideEncodeAVX2(mid[:n], side[:n], left, right)
	for i := n; i < len(mid); i++ {
		if mid[i] != math.MaxInt32 {
			t.Errorf("midSideEncodeAVX2 wrote past mid end at [%d] = %d", i, mid[i])
		}
		if side[i] != math.MaxInt32 {
			t.Errorf("midSideEncodeAVX2 wrote past side end at [%d] = %d", i, side[i])
		}
	}
}
