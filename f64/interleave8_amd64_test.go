//go:build amd64

package f64

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// The N=8 AMD64 kernels transpose 4 f64 frames per block as two stacked 4x4
// transposes (streams 0-3 fill each output frame's low YMM, streams 4-7 the
// high YMM). These tests drive the kernels directly with block-aligned frame
// counts so a misplaced lane (wrong VUNPCKxPD or VPERM2F128 selector, or a
// swapped store offset) is caught against the literal scalar oracle. chanVal
// makes every (channel, frame) value unique, so no transpose bug can alias.

func TestInterleave8AVX_MatchesOracle(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available on this CPU")
	}
	const nc = 8
	for _, n := range kernelFrameCounts() {
		srcs := make([][]float64, nc)
		for c := range srcs {
			srcs[c] = make([]float64, n)
			for i := range srcs[c] {
				srcs[c][i] = chanVal(c, i)
			}
		}
		got := make([]float64, n*nc)
		interleave8AVX(got, srcs, n)

		want := make([]float64, n*nc)
		interleaveNRef(want, srcs, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d: interleave8AVX[%d] = %v, want %v", n, i, got[i], want[i])
			}
		}
	}
}

func TestDeinterleave8AVX_MatchesOracle(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available on this CPU")
	}
	const nc = 8
	for _, n := range kernelFrameCounts() {
		src := make([]float64, n*nc)
		for i := range src {
			src[i] = float64(i + 1)
		}
		dsts := make([][]float64, nc)
		for c := range dsts {
			dsts[c] = make([]float64, n)
		}
		deinterleave8AVX(dsts, src, n)

		want := make([][]float64, nc)
		for c := range want {
			want[c] = make([]float64, n)
		}
		deinterleaveNRef(want, src, n)
		for c := range dsts {
			for i := range dsts[c] {
				if dsts[c][i] != want[c][i] {
					t.Fatalf("n=%d: deinterleave8AVX dst[%d][%d] = %v, want %v",
						n, c, i, dsts[c][i], want[c][i])
				}
			}
		}
	}
}
