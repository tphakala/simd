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
