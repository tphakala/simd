//go:build arm64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestAbsPow34NEON_ParityWithGo drives the hand-encoded NEON kernel directly over
// every prefix length, catching a wrong WORD, a mishandled 4-element remainder, a
// dropped scalar tail, or a lane error that SIMD-vs-SIMD agreement would miss.
// Each result must be bit-identical to absPow34Go, including the +Inf/0
// saturation lanes and NaN (compared NaN-equal).
func TestAbsPow34NEON_ParityWithGo(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	checkAbsPow34Kernel(t, "absPow34NEON", absPow34NEON)
}

// TestAbsPow34NEON_AllocFree asserts the kernel runs allocation-free at the
// kernel boundary.
func TestAbsPow34NEON_AllocFree(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	const n = 1024
	src := make([]float32, n)
	dst := make([]float32, n)
	if got := testing.AllocsPerRun(100, func() { absPow34NEON(dst, src) }); got != 0 {
		t.Errorf("absPow34NEON allocated %v times per run, want 0", got)
	}
}

// TestAbsPow34Dispatch_arm64 pins the dispatch inputs absPow34_32 reads: hasNEON
// must reflect CPU detection so a mis-wired flag cannot silently strand every
// call on the Go path. It is white-box (reads package-level state) and must not
// call t.Parallel(). The NEON length threshold matches sqrt32's (>= 4), verified
// by driving one below-threshold and one at-threshold length through the public
// AbsPow34 and confirming both match the reference.
func TestAbsPow34Dispatch_arm64(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	for _, n := range []int{3, 4} { // below the >=4 NEON threshold and at it
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i)*2.5 - 1
		}
		got := make([]float32, n)
		want := make([]float32, n)
		AbsPow34(got, src)
		absPow34Go(want, src)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("AbsPow34 n=%d: dst[%d] = %v, want %v", n, i, got[i], want[i])
			}
		}
	}
}
