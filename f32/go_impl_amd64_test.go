//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Tests for init functions to ensure they properly configure function pointers
// These tests are AMD64-specific because they test x86 SIMD initialization paths.

// tierInits enumerates every init-time tier assigner in this package together
// with the CPU support it needs. initAVX was missing from this file entirely
// (#204): the AVX kernels were reachable only through the package's own init(),
// so nothing here ever bound them, and they were never checked against a
// sibling tier.
var tierInits = []struct {
	name      string
	init      func()
	supported func() bool
	// minSIMD is the value the tier must leave in minSIMDElements, or 0 when the
	// tier does not set it.
	minSIMD int
}{
	{"Go", initGo, func() bool { return true }, 0},
	{"SSE", initSSE, func() bool { return cpu.X86.SSE2 }, 0},
	{"AVX", initAVX, func() bool { return cpu.X86.AVX && cpu.X86.FMA }, 0},
	{"AVX512", initAVX512, func() bool { return cpu.X86.AVX512F && cpu.X86.AVX512VL }, minAVX512Elements},
}

// TestInitTiers binds each tier in turn and checks its dotProduct against the
// pure-Go reference over lengths that reach the vector body, not just its scalar
// tail. The previous form used one length-4 input, which is below both
// minAVXElements and minAVX512Elements, so it exercised the tail of whichever
// kernel it bound and could not have told two tiers apart.
func TestInitTiers(t *testing.T) {
	// Lengths straddling the block widths: 4 stays under every vector threshold,
	// 8 and 16 are whole AVX blocks, 17 and 33 leave a remainder, 64 covers
	// several AVX-512 blocks.
	lengths := []int{4, 8, 16, 17, 33, 64}

	// init* reassigns convolveValidMaxAbsImpl too; restore it so the fused kernel
	// stays consistent with dotProductImpl for the exact-equality convolution tests.
	savedDotProduct := dotProductImpl
	savedConvolveValidMaxAbs := convolveValidMaxAbsImpl
	savedMinSIMD := minSIMDElements
	t.Cleanup(func() {
		dotProductImpl = savedDotProduct
		convolveValidMaxAbsImpl = savedConvolveValidMaxAbs
		minSIMDElements = savedMinSIMD
	})

	for _, tier := range tierInits {
		t.Run(tier.name, func(t *testing.T) {
			if !tier.supported() {
				t.Skipf("%s not supported on this CPU", tier.name)
			}
			tier.init()

			if tier.minSIMD != 0 && minSIMDElements != tier.minSIMD {
				t.Errorf("init%s left minSIMDElements = %d, want %d",
					tier.name, minSIMDElements, tier.minSIMD)
			}

			for _, n := range lengths {
				a := make([]float32, n)
				b := make([]float32, n)
				for i := range n {
					// Asymmetric values on purpose: a ramp against its own
					// reverse yields the same total when lanes are dropped or
					// reordered, so it cannot detect a missing block.
					a[i] = float32(i)*0.5 + 1
					b[i] = float32(n-i) * 0.25
				}
				got := dotProductImpl(a, b)
				want := dotProductGo(a, b)
				// closeFloat32 scales with the magnitude, which matters because
				// a dot product grows with n. Measured on this input family, the
				// tiers agree exactly up to n = 256 and diverge by 2688 in
				// absolute terms at n = 4096, all of it summation order: a fixed
				// absolute tolerance would read that as a kernel bug the moment
				// anyone extended the length list.
				if !closeFloat32(got, want) {
					t.Errorf("n=%d: init%s dotProduct = %v, Go reference = %v (diff %v)",
						n, tier.name, got, want, float64(got-want))
				}
			}
		})
	}
}
