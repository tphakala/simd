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
