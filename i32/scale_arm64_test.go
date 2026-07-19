//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestScaleQ31NEON_ParityWithGo drives the kernel directly across the full tier-3
// sweep, over lengths the dispatcher would never route to it, so a threshold
// change cannot quietly reduce this to a test of the Go reference against itself.
// MinInt32 rides index 0 under k=MinInt32 so the wrap is exercised at every
// length; the SMULL/XTN lane order is checked against the oracle at every position
// via the value-matrix test in scale_test.go.
//
//nolint:dupl // The Q31/Q15 parity sweep is intentionally identical bar the kernel, ref and coefficient type.
func TestScaleQ31NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	ks := []int32{math.MinInt32, math.MaxInt32, 1, -1, 0x40000000, 0x12345678}
	for _, n := range tier3Lengths {
		a := genI32(n, 41)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, k := range ks {
			got := make([]int32, n)
			want := make([]int32, n)
			scaleQ31NEON(got, a, k)
			scaleQ31Go(want, a, k)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("scaleQ31NEON n=%d k=%d: dst[%d] = %d, want %d", n, k, i, got[i], want[i])
				}
				if o := scaleQ31Oracle(a[i], k); got[i] != o {
					t.Fatalf("scaleQ31NEON n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, got[i], o)
				}
			}
		}
	}
}

// TestScaleQ15NEON_ParityWithGo is the Q15 counterpart over the int16 range.
//
//nolint:dupl // The Q31/Q15 parity sweep is intentionally identical bar the kernel, ref and coefficient type.
func TestScaleQ15NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	ks := []int16{math.MinInt16, math.MaxInt16, 1, -1, 0x4000, 0x1234}
	for _, n := range tier3Lengths {
		a := genI32(n, 42)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, k := range ks {
			got := make([]int32, n)
			want := make([]int32, n)
			scaleQ15NEON(got, a, k)
			scaleQ15Go(want, a, k)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("scaleQ15NEON n=%d k=%d: dst[%d] = %d, want %d", n, k, i, got[i], want[i])
				}
				if o := scaleQ15Oracle(a[i], k); got[i] != o {
					t.Fatalf("scaleQ15NEON n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, got[i], o)
				}
			}
		}
	}
}

// TestScaleNEON_NoOverwrite guards the scalar tails: neither kernel may write past
// n when n is not a multiple of the 4-lane block.
func TestScaleNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 11
	a := genI32(n, 43)
	dst := make([]int32, n+8)

	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	scaleQ31NEON(dst[:n], a, 0x0BADBEEF)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("scaleQ31NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}

	for i := range dst {
		dst[i] = math.MaxInt32
	}
	scaleQ15NEON(dst[:n], a, 0x2BAD)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("scaleQ15NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestScaleNEON_AllocFree asserts the kernels run allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestScaleNEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	a := make([]int32, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { scaleQ31NEON(dst, a, 0x12345678) }); got != 0 {
		t.Errorf("scaleQ31NEON allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { scaleQ15NEON(dst, a, 0x1234) }); got != 0 {
		t.Errorf("scaleQ15NEON allocated %v times per run, want 0", got)
	}
}

// TestScaleDispatch_ReachesNEON pins the dispatch state ScaleQ31/ScaleQ15 depend
// on. It is a white-box check: the NEON kernels are bit-identical to the Go
// reference by design, so a dispatcher that silently routed every call to Go would
// pass every parity test. It must not call t.Parallel(): it reads package-level
// dispatch state.
func TestScaleDispatch_ReachesNEON(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	if minNEONScaleQ31 > 8 {
		t.Fatalf("minNEONScaleQ31 = %d exceeds two vector blocks: ScaleQ31 would not vectorize at the lengths it was written for", minNEONScaleQ31)
	}
	if minNEONScaleQ15 > 8 {
		t.Fatalf("minNEONScaleQ15 = %d exceeds two vector blocks: ScaleQ15 would not vectorize at the lengths it was written for", minNEONScaleQ15)
	}
}
