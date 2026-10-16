//go:build amd64 || arm64

package f32

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// biquadBlocksFunc adapts biquadCascadeBlocks to biquadFunc with the given lane
// count and the performance floor off (minWork 0). Only the kernel precondition
// then decides between kernel and Go, so the shortest legal inputs, n = 2*(b-1)
// and n = 2*(b-1)+1 (which takes the odd step), reach the kernel. It cuts both
// slices to the shorter one, as BiquadCascade does before dispatch.
func biquadBlocksFunc(lanes int) biquadFunc {
	return func(dst, src, coeffs, state []float32) {
		n := min(len(dst), len(src))
		if n == 0 {
			return
		}
		biquadCascadeBlocks(dst[:n], src[:n], coeffs, state, lanes, 0)
	}
}

// biquadNBlocksFunc adapts biquadCascadeNBlocks to biquadNFunc with the given
// lane count and floors. Lanes 0 sends everything to the Go path; floors of 0
// reach the pair kernel with the shortest legal inputs.
func biquadNBlocksFunc(lanes, minWork, pairMinWork int) biquadNFunc {
	return func(dst, src [][]float32, coeffs []float32, state [][]float32) {
		biquadCascadeNBlocks(dst, src, coeffs, state, lanes, minWork, pairMinWork)
	}
}

// TestBiquadCascadeNGo runs the multichannel suite with every channel on the Go
// path, so the driver's pairing and tails are covered when dispatch lands on a
// kernel, and a compiler-fused multiply-add in the fallback (arm64) is caught.
func TestBiquadCascadeNGo(t *testing.T) {
	runBiquadNSuite(t, biquadNBlocksFunc(0, 0, 0))
}

// TestBiquadCascadeNPairDirect runs the multichannel suite with both floors
// off at every lane count the host can execute, so every pair of channels of a
// 1- or 2-section cascade that has a sample reaches the pair kernel at its
// shortest legal length. Lanes 4 and 8 differ only in the kernels the leftover
// channels and longer cascades use. biquadCascadeNBlocks takes the lane count as
// an argument, so on arm64 this stays covered when SIMD_DISABLE sends the public
// path to Go; on amd64 biquadRunnableLanes reads the cpu flags and the test
// skips when SIMD_DISABLE clears AVX.
func TestBiquadCascadeNPairDirect(t *testing.T) {
	lanesList := biquadRunnableLanes()
	if len(lanesList) == 0 {
		t.Skip("no biquad kernel can run on this CPU")
	}
	for _, lanes := range lanesList {
		t.Run(fmt.Sprintf("lanes=%d", lanes), func(t *testing.T) {
			runBiquadNSuite(t, biquadNBlocksFunc(lanes, 0, 0))
		})
	}
}

// BenchmarkBiquadCascadeShort sweeps short chunks with the kernel forced on
// (minWork 0) against the Go path, to find the n*sections where the kernel pays;
// biquadMinWork is set from this sweep. Combinations below the kernel
// precondition n >= 2*(sections-1) would run Go on both sides and are skipped.
func BenchmarkBiquadCascadeShort(b *testing.B) {
	lanes := biquadKernelLanes()
	if lanes == 0 {
		b.Skip("no biquad kernel on this CPU")
	}
	r := rand.New(rand.NewPCG(15, 16))
	for _, sections := range []int{1, 2, 4, 8} {
		coeffs := biquadRBJ(sections, r)
		for _, n := range []int{4, 8, 16, 32, 64, 128} {
			if n < biquadSkew*(sections-1) {
				continue
			}
			src := biquadNoise(n, r)
			dst := make([]float32, n)
			state := make([]float32, biquadStateLen*sections)
			variants := []struct {
				name string
				fn   func()
			}{
				{"Kernel", func() { biquadCascadeBlocks(dst, src, coeffs, state, lanes, 0) }},
				{"Go", func() { biquadCascadeGo(dst, src, coeffs, state) }},
			}
			for _, v := range variants {
				b.Run(fmt.Sprintf("N=%d/n=%d/%s", sections, n, v.name), func(b *testing.B) {
					for b.Loop() {
						v.fn()
					}
					b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/sample")
				})
			}
		}
	}
}

// BenchmarkBiquadCascadePairShort sweeps short two-channel chunks with the pair
// kernel forced on (pair floor 0) against two production BiquadCascade calls,
// the baseline the pair path has to beat; biquadPairMinWork is set from this
// sweep. Mono2 includes the mono path's own Go fallback below biquadMinWork.
func BenchmarkBiquadCascadePairShort(b *testing.B) {
	lanes := biquadKernelLanes()
	if lanes == 0 {
		b.Skip("no biquad kernel on this CPU")
	}
	r := rand.New(rand.NewPCG(45, 46))
	for _, sections := range []int{1, 2} {
		coeffs := biquadRBJ(sections, r)
		for _, m := range []int{2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256} {
			if !biquadKernelOK(m, sections, 0) {
				continue
			}
			src := [][]float32{biquadNoise(m, r), biquadNoise(m, r)}
			dst := [][]float32{make([]float32, m), make([]float32, m)}
			state := [][]float32{make([]float32, biquadStateLen*sections), make([]float32, biquadStateLen*sections)}
			variants := []struct {
				name string
				fn   func()
			}{
				{"Pair", func() { biquadCascadeNBlocks(dst, src, coeffs, state, lanes, biquadMinWork, 0) }},
				{"Mono2", func() {
					BiquadCascade(dst[0], src[0], coeffs, state[0])
					BiquadCascade(dst[1], src[1], coeffs, state[1])
				}},
			}
			for _, v := range variants {
				b.Run(fmt.Sprintf("sec=%d/m=%d/impl=%s", sections, m, v.name), func(b *testing.B) {
					for b.Loop() {
						v.fn()
					}
					b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(2*m), "ns/chsample")
				})
			}
		}
	}
}
