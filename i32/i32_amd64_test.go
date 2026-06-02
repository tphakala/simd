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
