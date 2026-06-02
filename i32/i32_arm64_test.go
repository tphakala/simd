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

// fillDiffSrc fills src with values that exercise the sign bit and force the
// residual to wrap int32 at the extremes.
func fillDiffSrc(src []int32) {
	for i := range src {
		src[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(src) > 1 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
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

func TestMidSideNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		left := make([]int32, n)
		right := make([]int32, n)
		fillPattern(left, right)

		midNEON := make([]int32, n)
		sideNEON := make([]int32, n)
		midGo := make([]int32, n)
		sideGo := make([]int32, n)
		midSideEncodeNEON(midNEON, sideNEON, left, right)
		midSideEncodeGo(midGo, sideGo, left, right)
		for i := range midGo {
			if midNEON[i] != midGo[i] || sideNEON[i] != sideGo[i] {
				t.Fatalf("n=%d: midSideEncodeNEON[%d] = (%d,%d), want (%d,%d)", n, i, midNEON[i], sideNEON[i], midGo[i], sideGo[i])
			}
		}

		lNEON := make([]int32, n)
		rNEON := make([]int32, n)
		lGo := make([]int32, n)
		rGo := make([]int32, n)
		midSideDecodeNEON(lNEON, rNEON, midNEON, sideNEON)
		midSideDecodeGo(lGo, rGo, midGo, sideGo)
		for i := range lGo {
			if lNEON[i] != lGo[i] || rNEON[i] != rGo[i] {
				t.Fatalf("n=%d: midSideDecodeNEON[%d] = (%d,%d), want (%d,%d)", n, i, lNEON[i], rNEON[i], lGo[i], rGo[i])
			}
		}
	}
}

func TestDiffNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	kernels := []struct {
		name string
		simd func(dst, src []int32)
		ref  func(dst, src []int32)
	}{
		{"diff1", diff1NEON, diff1Go},
		{"diff2", diff2NEON, diff2Go},
		{"diff3", diff3NEON, diff3Go},
		{"diff4", diff4NEON, diff4Go},
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // the kernel's warm-up reads assume len >= order; the
			// dispatch only calls it for len >= minNEONElements, routing
			// shorter inputs to the Go path.
		}
		src := make([]int32, n)
		fillDiffSrc(src)
		for _, k := range kernels {
			gotNEON := make([]int32, n)
			gotGo := make([]int32, n)
			k.simd(gotNEON, src)
			k.ref(gotGo, src)
			for i := range gotGo {
				if gotNEON[i] != gotGo[i] {
					t.Fatalf("n=%d: %sNEON[%d] = %d, want %d (Go)", n, k.name, i, gotNEON[i], gotGo[i])
				}
			}
		}
	}
}

// TestDiff1NEON_NoOverwrite guards the scalar tail.
func TestDiff1NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 17
	src := make([]int32, n)
	fillDiffSrc(src)
	dst := make([]int32, n+4)
	for i := range dst {
		dst[i] = math.MaxInt32
	}
	diff1NEON(dst[:n], src)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("diff1NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}
