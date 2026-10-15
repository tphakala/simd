//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// biquadKernelLanes is the widest pass the host runs on a kernel, or 0.
func biquadKernelLanes() int {
	switch {
	case cpu.X86.AVX2:
		return biquadMaxLanes
	case cpu.X86.AVX:
		return biquadNarrowLanes
	}
	return 0
}

// TestBiquadCascadeAVXDirect runs the suite against the 4-section AVX kernel only:
// lanes 4 makes every cascade, including N = 5..8, run in passes of 4 sections,
// the path an AVX-only host takes. It skips when cpu.X86.AVX is false, which is
// also the case when SIMD_DISABLE clears the flag, so it never executes a VEX
// instruction on a CPU without it.
func TestBiquadCascadeAVXDirect(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	runBiquadSuite(t, biquadBlocksFunc(biquadNarrowLanes))
}

// TestBiquadCascadeAVX2Direct runs the suite with lanes 8, so blocks of up to 4
// sections use the AVX kernel and larger blocks the AVX2 kernel. It skips when
// cpu.X86.AVX2 is false, including when SIMD_DISABLE clears the flag.
func TestBiquadCascadeAVX2Direct(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	runBiquadSuite(t, biquadBlocksFunc(biquadMaxLanes))
}
