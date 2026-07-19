//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestFIRValidQ15NEON_ParityWithGo drives the kernel directly across output
// lengths that force the 4-wide vector body plus every scalar-output-tail
// remainder, over the combFilterConst tap count (5) and neighbors. MinInt16/
// MaxInt16 ride the ends of taps and MinInt32/MaxInt32 the ends of x, so the
// sign-extension into SMULL and the wrap are exercised at every length.
func TestFIRValidQ15NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	tapCounts := []int{1, 2, 3, 5, 8, 16}
	outLens := []int{4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 23, 24, 31, 32, 33, 40}
	for _, kl := range tapCounts {
		taps := genI16(kl, uint32(kl)*17+3)
		taps[0] = math.MinInt16
		taps[kl-1] = math.MaxInt16
		for _, outLen := range outLens {
			xl := outLen + kl - 1
			x := genI32(xl, uint32(xl)*7+uint32(kl))
			x[0] = math.MinInt32
			x[xl-1] = math.MaxInt32
			got := make([]int32, outLen)
			want := make([]int32, outLen)
			firValidQ15NEON(got, x, taps)
			firValidQ15Go(want, x, taps)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("firValidQ15NEON kl=%d outLen=%d: dst[%d] = %d, want %d", kl, outLen, i, got[i], want[i])
				}
			}
		}
	}
}

// TestFIRValidQ15NEON_OverRead catches a kernel that reads x past len(x). The
// in-range body is tame (samples in -2..2), while the slack past len(x) is
// poisoned with 0x55555555, a large non-zero value that is NOT an additive
// identity (unlike a zero or an even count of MinInt32, whose over-read would be
// invisible in an additive accumulate). x is backing[:xl] over a backing with 16
// int32 of still-allocated slack, so a kernel that loads a stray window block or
// runs the tap loop one element too far lands in the poison and its result flips
// away from the reference over x[:xl]; a correct kernel stops at len(x)-1 and
// stays equal. The taps are 0x4000 (0.5 in Q15) so every poisoned lane makes a
// large, visible contribution.
func TestFIRValidQ15NEON_OverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const poison = int32(0x55555555)
	for _, kl := range []int{1, 2, 5, 8} {
		taps := make([]int16, kl)
		for i := range taps {
			taps[i] = 0x4000
		}
		for _, outLen := range []int{4, 5, 6, 7, 8, 9, 11, 13, 17, 23} {
			xl := outLen + kl - 1
			const slack = 16 // >= one widest output block worth of window
			backing := make([]int32, xl+slack)
			for i := range backing {
				if i < xl {
					backing[i] = int32(i%5 - 2) // tame body: -2..2
				} else {
					backing[i] = poison
				}
			}
			x := backing[:xl]
			got := make([]int32, outLen)
			want := make([]int32, outLen)
			firValidQ15NEON(got, x, taps)
			firValidQ15Go(want, x, taps)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("firValidQ15NEON kl=%d outLen=%d: dst[%d] = %d, want %d: kernel read x past len into poison", kl, outLen, i, got[i], want[i])
				}
			}
		}
	}
}

// TestFIRValidQ15NEON_NoOverwrite guards the scalar-output tail: the kernel may
// not write past n when n is not a multiple of the 4-output block.
func TestFIRValidQ15NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const outLen = 11
	taps := genI16(5, 81)
	x := genI32(outLen+len(taps)-1, 82)
	dst := make([]int32, outLen+8)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	firValidQ15NEON(dst[:outLen], x, taps)
	for i := outLen; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("firValidQ15NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestFIRValidQ15NEON_AllocFree asserts the kernel runs allocation-free, the
// repo's zero-allocation contract enforced at the kernel boundary.
func TestFIRValidQ15NEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	x := make([]int32, 1024)
	taps := make([]int16, 5)
	dst := make([]int32, 1020)
	if got := testing.AllocsPerRun(100, func() { firValidQ15NEON(dst, x, taps) }); got != 0 {
		t.Errorf("firValidQ15NEON allocated %v times per run, want 0", got)
	}
}

// TestFIRValidQ15Dispatch_ReachesNEON pins the dispatch state FIRValidQ15 depends
// on. It is a white-box check: the NEON kernel is bit-identical to the Go
// reference by design, so a dispatcher that silently routed every call to Go would
// pass every parity test. It must not call t.Parallel(): it reads package-level
// dispatch state.
func TestFIRValidQ15Dispatch_ReachesNEON(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	if minNEONFIR > 8 {
		t.Fatalf("minNEONFIR = %d exceeds two vector blocks: FIRValidQ15 would not vectorize at the lengths it was written for", minNEONFIR)
	}
}
