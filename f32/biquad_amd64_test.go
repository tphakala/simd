//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

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

// biquadRunnableLanes lists the lane counts the host can execute: 8 needs AVX2
// and 4 needs AVX. It reads the cpu flags, so when SIMD_DISABLE clears them the
// direct tests built on it skip and cover nothing on that run.
func biquadRunnableLanes() []int {
	var out []int
	if cpu.X86.AVX2 {
		out = append(out, biquadMaxLanes)
	}
	if cpu.X86.AVX {
		out = append(out, biquadNarrowLanes)
	}
	return out
}

// TestBiquadKernelLanesMatchesCPU_amd64 pins biquadKernelLanes to the cpu flags,
// the part TestBiquadCascadeRoutesToKernel takes on trust: a lane count that is
// too low or zero would send every call down a slower path and still produce the
// right output. It is white-box and must not call t.Parallel().
func TestBiquadKernelLanesMatchesCPU_amd64(t *testing.T) {
	want := 0
	switch {
	case cpu.X86.AVX2:
		want = biquadMaxLanes
	case cpu.X86.AVX:
		want = biquadNarrowLanes
	}
	if got := biquadKernelLanes(); got != want {
		t.Fatalf("biquadKernelLanes() = %d, want %d (AVX2=%v AVX=%v)",
			got, want, cpu.X86.AVX2, cpu.X86.AVX)
	}
}

// TestBiquadPairSel_amd64 pins the VPERMILPS control words: the pair of lanes
// that hold the last section of channels 0 and 1.
func TestBiquadPairSel_amd64(t *testing.T) {
	for sections, want := range map[int][2]uint32{1: {0, 1}, 2: {2, 3}} {
		if s0, s1 := biquadPairSel(sections); s0 != want[0] || s1 != want[1] {
			t.Errorf("biquadPairSel(%d) = (%d, %d), want (%d, %d)", sections, s0, s1, want[0], want[1])
		}
	}
}
