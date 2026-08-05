//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestGainQ31AVX2_ParityWithGo drives the kernel directly across the full tier-3
// sweep, over lengths the dispatcher would never route to it, so a threshold
// change cannot quietly reduce this to a test of the Go reference against itself.
// MinInt32 rides index 0 and MaxInt32 the last index so the SHL32/MULT32_32_Q31/
// PSHR32 wraps are exercised at every length; the even/odd VPMULDQ recombine is
// checked against the oracle at every position by the value-matrix test in
// gain_test.go.
//
//nolint:dupl // The dispatched/AVX2/NEON parity sweeps are intentionally identical bar the kernel under test.
func TestGainQ31AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	gains := []int32{math.MinInt32, math.MaxInt32, 1, -1, 0x40000000, 0x12345678}
	for _, n := range tier3Lengths {
		a := genI32(n, 61)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, g := range gains {
			for _, s := range gainShifts {
				got := make([]int32, n)
				want := make([]int32, n)
				gainQ31AVX2(got, a, g, s.pre, s.post)
				gainQ31Go(want, a, g, s.pre, s.post)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("gainQ31AVX2 n=%d g=%d pre=%d post=%d: dst[%d] = %d, want %d", n, g, s.pre, s.post, i, got[i], want[i])
					}
					if o := gainQ31Oracle(a[i], g, s.pre, s.post); got[i] != o {
						t.Fatalf("gainQ31AVX2 n=%d g=%d pre=%d post=%d: dst[%d] = %d, want %d (oracle)", n, g, s.pre, s.post, i, got[i], o)
					}
				}
			}
		}
	}
}

// TestGainQ31AVX2_NoOverwrite guards the scalar tail: the kernel may not write past
// n when n is not a multiple of the 8-lane block.
func TestGainQ31AVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 11
	a := genI32(n, 62)
	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	gainQ31AVX2(dst[:n], a, 0x0BADBEEF, 9, 12)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("gainQ31AVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestGainQ31AVX2_AllocFree asserts the kernel runs allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestGainQ31AVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	a := make([]int32, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { gainQ31AVX2(dst, a, 0x12345678, 9, 12) }); got != 0 {
		t.Errorf("gainQ31AVX2 allocated %v times per run, want 0", got)
	}
}

// TestGainQ31Dispatch_ReachesSIMD pins the dispatch state GainQ31 depends on. It is
// a white-box check: the AVX2 kernel is bit-identical to the Go reference by
// design, so a dispatcher that silently routed every call to Go would pass every
// parity test. It must not call t.Parallel(): it reads package-level dispatch
// state.
func TestGainQ31Dispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2GainQ31 > 16 {
		t.Fatalf("minAVX2GainQ31 = %d exceeds two vector blocks: GainQ31 would not vectorize at the lengths it was written for", minAVX2GainQ31)
	}
}
