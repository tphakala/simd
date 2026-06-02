//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// The N=3 AMD64 gather/blend kernels process whole 8-frame blocks; the dispatch
// reslices the tail to Go. These tests drive the kernels directly with
// block-aligned frame counts so a misplaced lane (wrong VPERMPS index or
// VPBLENDD mask) is caught against the literal scalar oracle. chanVal makes
// every (channel, frame) value unique, so no transpose bug can alias.

func interleave3KernelFrameCounts() []int { return []int{8, 16, 24, 64, 256, 1024} }

func TestInterleave3AVX_MatchesOracle(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	const nc = 3
	for _, n := range interleave3KernelFrameCounts() {
		srcs := make([][]float32, nc)
		for c := range srcs {
			srcs[c] = make([]float32, n)
			for i := range srcs[c] {
				srcs[c][i] = chanVal(c, i)
			}
		}
		got := make([]float32, n*nc)
		interleave3AVX(got, srcs[0], srcs[1], srcs[2], n)

		want := make([]float32, n*nc)
		interleaveNRef(want, srcs, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d: interleave3AVX[%d] = %v, want %v", n, i, got[i], want[i])
			}
		}
	}
}

func TestDeinterleave3AVX_MatchesOracle(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	const nc = 3
	for _, n := range interleave3KernelFrameCounts() {
		src := make([]float32, n*nc)
		for i := range src {
			src[i] = float32(i + 1)
		}
		dsts := make([][]float32, nc)
		for c := range dsts {
			dsts[c] = make([]float32, n)
		}
		deinterleave3AVX(dsts[0], dsts[1], dsts[2], src, n)

		want := make([][]float32, nc)
		for c := range want {
			want[c] = make([]float32, n)
		}
		deinterleaveNRef(want, src, n)
		for c := range dsts {
			for i := range dsts[c] {
				if dsts[c][i] != want[c][i] {
					t.Fatalf("n=%d: deinterleave3AVX dst[%d][%d] = %v, want %v",
						n, c, i, dsts[c][i], want[c][i])
				}
			}
		}
	}
}
