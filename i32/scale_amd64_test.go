//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestScaleQ31AVX2_ParityWithGo drives the kernel directly across the full tier-3
// sweep, over lengths the dispatcher would never route to it, so a threshold
// change cannot quietly reduce this to a test of the Go reference against itself.
// MinInt32 rides index 0 under k=MinInt32 so the wrap is exercised at every
// length, and the VPMULDQ even/odd recombine is checked against the oracle at
// every position via the value-matrix test in scale_test.go.
//
//nolint:dupl // The Q31/Q15 parity sweep is intentionally identical bar the kernel, ref and coefficient type.
func TestScaleQ31AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
			scaleQ31AVX2(got, a, k)
			scaleQ31Go(want, a, k)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("scaleQ31AVX2 n=%d k=%d: dst[%d] = %d, want %d", n, k, i, got[i], want[i])
				}
				if o := scaleQ31Oracle(a[i], k); got[i] != o {
					t.Fatalf("scaleQ31AVX2 n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, got[i], o)
				}
			}
		}
	}
}

// TestScaleQ15AVX2_ParityWithGo is the Q15 counterpart over the int16 range.
//
//nolint:dupl // The Q31/Q15 parity sweep is intentionally identical bar the kernel, ref and coefficient type.
func TestScaleQ15AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
			scaleQ15AVX2(got, a, k)
			scaleQ15Go(want, a, k)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("scaleQ15AVX2 n=%d k=%d: dst[%d] = %d, want %d", n, k, i, got[i], want[i])
				}
				if o := scaleQ15Oracle(a[i], k); got[i] != o {
					t.Fatalf("scaleQ15AVX2 n=%d k=%d: dst[%d] = %d, want %d (oracle)", n, k, i, got[i], o)
				}
			}
		}
	}
}

// TestScaleAVX2_NoOverwrite guards the scalar tails: neither kernel may write past
// n when n is not a multiple of the 8-lane block.
func TestScaleAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 11
	a := genI32(n, 43)
	dst := make([]int32, n+8)

	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	scaleQ31AVX2(dst[:n], a, 0x0BADBEEF)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("scaleQ31AVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}

	for i := range dst {
		dst[i] = math.MaxInt32
	}
	scaleQ15AVX2(dst[:n], a, 0x2BAD)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("scaleQ15AVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestScaleAVX2_AllocFree asserts the kernels run allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestScaleAVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	a := make([]int32, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { scaleQ31AVX2(dst, a, 0x12345678) }); got != 0 {
		t.Errorf("scaleQ31AVX2 allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { scaleQ15AVX2(dst, a, 0x1234) }); got != 0 {
		t.Errorf("scaleQ15AVX2 allocated %v times per run, want 0", got)
	}
}

// TestScaleDispatch_ReachesSIMD pins the dispatch state ScaleQ31/ScaleQ15 depend
// on. It is a white-box check: the AVX2 kernels are bit-identical to the Go
// reference by design, so a dispatcher that silently routed every call to Go would
// pass every parity test. It must not call t.Parallel(): it reads package-level
// dispatch state.
func TestScaleDispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2ScaleQ31 > 16 {
		t.Fatalf("minAVX2ScaleQ31 = %d exceeds two vector blocks: ScaleQ31 would not vectorize at the lengths it was written for", minAVX2ScaleQ31)
	}
	if minAVX2ScaleQ15 > 16 {
		t.Fatalf("minAVX2ScaleQ15 = %d exceeds two vector blocks: ScaleQ15 would not vectorize at the lengths it was written for", minAVX2ScaleQ15)
	}
}
