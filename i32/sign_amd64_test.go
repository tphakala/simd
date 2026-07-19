//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestNegWhereNegAVX2_ParityWithGo drives the kernel directly across the full
// tier-3 sweep, over lengths the dispatcher would never route to it, so a
// threshold change cannot quietly reduce this to a test of the Go reference
// against itself. MinInt32 rides index 0 under a -0.0 sign so the wrap is
// exercised at every length.
func TestNegWhereNegAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		mag := genI32(n, 91)
		sign := genSigns(n, 92)
		if n > 0 {
			mag[0] = math.MinInt32
			sign[0] = math.Float32frombits(1 << 31) // -0.0
		}
		got := make([]int32, n)
		want := make([]int32, n)
		negWhereNegAVX2(got, mag, sign)
		negWhereNegGo(want, mag, sign)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("negWhereNegAVX2 n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestNegWhereNegAVX2_NoOverwrite guards the scalar tail: the kernel must not
// write past n even when n is not a multiple of the 8-lane block.
func TestNegWhereNegAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 11
	mag := genI32(n, 93)
	sign := genSigns(n, 94)
	dst := make([]int32, n+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	negWhereNegAVX2(dst[:n], mag, sign)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("negWhereNegAVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestNegWhereNegAVX2_AllocFree asserts the kernel runs allocation-free, the
// repo's zero-allocation contract enforced at the kernel boundary.
func TestNegWhereNegAVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	mag := make([]int32, n)
	sign := make([]float32, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { negWhereNegAVX2(dst, mag, sign) }); got != 0 {
		t.Errorf("negWhereNegAVX2 allocated %v times per run, want 0", got)
	}
}

// TestNegWhereNegDispatch_ReachesSIMD pins the dispatch state NegWhereNeg
// depends on. It is a white-box check: the AVX2 kernel is bit-identical to the
// Go reference by design, so a dispatcher that silently routed every call to Go
// would pass every parity test in this package. It must not call t.Parallel():
// it reads package-level dispatch state.
func TestNegWhereNegDispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2NegWhereNeg > 16 {
		t.Fatalf("minAVX2NegWhereNeg = %d exceeds two vector blocks: NegWhereNeg would not vectorize at the lengths it was written for", minAVX2NegWhereNeg)
	}
}
