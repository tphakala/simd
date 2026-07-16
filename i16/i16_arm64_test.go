//go:build arm64

package i16

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestInterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		a := make([]int16, n)
		b := make([]int16, n)
		fillPattern(a, b)

		gotNEON := make([]int16, n*2)
		gotGo := make([]int16, n*2)
		interleave2NEON(gotNEON, a, b)
		interleave2Go(gotGo, a, b)

		for i := range gotGo {
			if gotNEON[i] != gotGo[i] {
				t.Fatalf("n=%d: interleave2NEON[%d] = %d, want %d (Go)", n, i, gotNEON[i], gotGo[i])
			}
		}
	}
}

func TestDeinterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		src := make([]int16, n*2)
		for i := range src {
			src[i] = int16(i) ^ math.MinInt16
		}

		aNEON := make([]int16, n)
		bNEON := make([]int16, n)
		aGo := make([]int16, n)
		bGo := make([]int16, n)
		deinterleave2NEON(aNEON, bNEON, src)
		deinterleave2Go(aGo, bGo, src)

		for i := range aGo {
			if aNEON[i] != aGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON a[%d] = %d, want %d (Go)", n, i, aNEON[i], aGo[i])
			}
			if bNEON[i] != bGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON b[%d] = %d, want %d (Go)", n, i, bNEON[i], bGo[i])
			}
		}
	}
}

// TestInterleave2NEON_NoOverwrite guards the scalar tail: the kernel must not
// write past n*2 output elements when n is not a multiple of the 8-lane block.
func TestInterleave2NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 13
	a := make([]int16, n)
	b := make([]int16, n)
	fillPattern(a, b)
	dst := make([]int16, n*2+8)
	for i := range dst {
		dst[i] = math.MaxInt16 // sentinel
	}
	interleave2NEON(dst[:n*2], a, b)
	for i := n * 2; i < len(dst); i++ {
		if dst[i] != math.MaxInt16 {
			t.Errorf("interleave2NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestDeinterleave2NEON_NoOverwrite guards both output buffers' scalar tails.
func TestDeinterleave2NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 13
	src := make([]int16, n*2)
	for i := range src {
		src[i] = int16(i) ^ math.MinInt16
	}
	a := make([]int16, n+8)
	b := make([]int16, n+8)
	for i := range a {
		a[i] = math.MaxInt16
		b[i] = math.MaxInt16
	}
	deinterleave2NEON(a[:n], b[:n], src)
	for i := n; i < len(a); i++ {
		if a[i] != math.MaxInt16 {
			t.Errorf("deinterleave2NEON wrote past end of a at [%d] = %d", i, a[i])
		}
		if b[i] != math.MaxInt16 {
			t.Errorf("deinterleave2NEON wrote past end of b at [%d] = %d", i, b[i])
		}
	}
}

// TestDotNEON_ParityWithGo exercises the kernel directly rather than through
// DotProduct, so a dispatch threshold change can never quietly turn this into a
// test of the Go reference against itself.
func TestDotNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range dotLengths {
		a, b := genI16(n, 51), genI16(n, 52)
		if got, want := dotNEON(a, b), dotGo(a, b); got != want {
			t.Errorf("dotNEON n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestDotNEON_MinInt16 pins SMLAL's wrapping behaviour at the one overflowing
// input, directly at the kernel.
func TestDotNEON_MinInt16(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for n := 1; n <= 64; n++ {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i], b[i] = math.MinInt16, math.MinInt16
		}
		if got, want := dotNEON(a, b), dotOracle(a, b); got != want {
			t.Errorf("dotNEON all-MinInt16 n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestDotNEON_Clamp verifies the in-assembly min(len(a), len(b)): the kernel
// must not read the longer operand past the shorter one's length.
func TestDotNEON_Clamp(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range dotLengths {
		if n == 0 {
			continue
		}
		long, short := genI16(n+37, 53), genI16(n, 54)
		if got, want := dotNEON(long, short), dotOracle(long, short); got != want {
			t.Errorf("dotNEON clamp n=%d: got %d, want %d", n, got, want)
		}
		if got, want := dotNEON(short, long), dotOracle(short, long); got != want {
			t.Errorf("dotNEON clamp (swapped) n=%d: got %d, want %d", n, got, want)
		}
	}
}

// TestDotDispatch_ReachesNEON asserts that DotProduct actually routes to the
// NEON kernel on NEON hardware.
//
// This has to be a white-box check on the dispatch state, because no black-box
// test can catch the failure. dotNEON is bit-identical to dotGo by design, so a
// dispatcher that silently sent every call to the Go reference would satisfy
// every parity assertion in this package while the SIMD path sat dead. The
// parity tests above call dotNEON directly (deliberately, so a threshold change
// cannot degrade them into dotGo-vs-dotGo), which means they cannot notice
// either. Nothing else checks that the two ever meet.
//
// It must not call t.Parallel(): it reads package-level dispatch state.
func TestDotDispatch_ReachesNEON(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	if !hasNEON {
		t.Fatal("hasNEON is false though cpu.ARM64.NEON is true: DotProduct silently runs the Go reference on every call")
	}
	// The threshold is documented as one vector block. Anything much larger
	// would leave the codec-length calls this primitive exists for running
	// scalar, which no other test would report.
	if minNEONDot > 2*minNEONElements {
		t.Fatalf("minNEONDot = %d exceeds two vector blocks (%d): DotProduct would not vectorize at the short lengths it was written for",
			minNEONDot, 2*minNEONElements)
	}
}
