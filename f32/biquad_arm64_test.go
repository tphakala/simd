//go:build arm64

package f32

import (
	"fmt"
	"testing"
)

// biquadKernelLanes is the widest pass the host runs on a kernel, or 0.
func biquadKernelLanes() int {
	if hasNEON {
		return biquadMaxLanes
	}
	return 0
}

// TestBiquadCascadeNEONDirect runs the suite against the NEON kernels through
// biquadCascadeBlocks, which does not consult hasNEON, so it stays covered when
// SIMD_DISABLE sends the public path to Go. Lanes 8 uses the 4-section kernel for
// blocks of up to 4 sections and the 8-section kernel above that; lanes 4 runs
// every cascade in passes of 4 sections on the 4-section kernel.
func TestBiquadCascadeNEONDirect(t *testing.T) {
	for _, lanes := range []int{biquadMaxLanes, biquadNarrowLanes} {
		t.Run(fmt.Sprintf("lanes=%d", lanes), func(t *testing.T) {
			runBiquadSuite(t, biquadBlocksFunc(lanes))
		})
	}
}
