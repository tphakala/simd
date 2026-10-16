//go:build arm64

package f32

import (
	"fmt"
	"testing"

	"github.com/tphakala/simd/cpu"
)

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

// biquadRunnableLanes lists the lane counts the host can execute. biquadCascadeBlocks
// does not consult hasNEON, so both are always runnable on arm64.
func biquadRunnableLanes() []int {
	return []int{biquadMaxLanes, biquadNarrowLanes}
}

// TestBiquadKernelLanesMatchesCPU_arm64 pins biquadKernelLanes to CPU detection,
// the part TestBiquadCascadeRoutesToKernel takes on trust. It is white-box and
// must not call t.Parallel().
func TestBiquadKernelLanesMatchesCPU_arm64(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection",
			hasNEON, cpu.ARM64.NEON)
	}
	want := 0
	if cpu.ARM64.NEON {
		want = biquadMaxLanes
	}
	if got := biquadKernelLanes(); got != want {
		t.Fatalf("biquadKernelLanes() = %d, want %d", got, want)
	}
}

// TestBiquadPairSel_arm64 pins the TBL byte-index words that move the last
// section's two lanes (channels 0 and 1) to lanes 0 and 1 of the output.
func TestBiquadPairSel_arm64(t *testing.T) {
	for sections, want := range map[int][2]uint32{
		1: {0x03020100, 0x07060504},
		2: {0x0B0A0908, 0x0F0E0D0C},
	} {
		if s0, s1 := biquadPairSel(sections); s0 != want[0] || s1 != want[1] {
			t.Errorf("biquadPairSel(%d) = (%#x, %#x), want (%#x, %#x)", sections, s0, s1, want[0], want[1])
		}
	}
}
