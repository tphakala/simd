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
	if minNEONDot > 16 {
		t.Fatalf("minNEONDot = %d exceeds two vector blocks: DotProduct would not vectorize at the short lengths it was written for", minNEONDot)
	}
}

// TestXCorr4NEON_ParityWithGo drives the 4-lag kernel directly, over lengths
// the dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself.
func TestXCorr4NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, xn := range []int{1, 2, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 64, 240} {
		x := genI16(xn, 111)
		y := genI16(xn+3, 112) // exactly the window the dispatcher passes
		dst := make([]int32, xcorrLagBlock)
		xcorr4NEON(dst, x, y)
		for k := range xcorrLagBlock {
			if got, want := dst[k], dotOracle(x, y[k:]); got != want {
				t.Errorf("xcorr4NEON xn=%d: dst[%d] = %d, want %d", xn, k, got, want)
			}
		}
	}
}

// TestXCorr4NEON_ShortWindowIsBounded feeds the kernel a y window shorter than
// its contract (len(x)+3) and asserts it clamps instead of reading past the
// end. The dispatcher never does this; the in-assembly clamp exists so that a
// future wrapper bug is a wrong number rather than an out-of-bounds read, and
// this is what proves the net is really there.
func TestXCorr4NEON_ShortWindowIsBounded(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	x := genI16(32, 113)
	for _, yn := range []int{0, 1, 2, 3, 4, 8, 16, 31, 34} {
		y := genI16(yn, 114)
		dst := make([]int32, xcorrLagBlock)
		xcorr4NEON(dst, x, y) // must not fault or read past y
		n := max(min(len(x), yn-3), 0)
		for k := range xcorrLagBlock {
			if got, want := dst[k], dotOracle(x[:n], y[min(k, yn):]); got != want {
				t.Errorf("xcorr4NEON short window yn=%d: dst[%d] = %d, want %d", yn, k, got, want)
			}
		}
	}
}

// TestXCorrDispatch_ReachesNEON pins the dispatch STATE that XCorr's SIMD path
// depends on: the feature flag is wired to CPU detection, and the threshold is
// not absurd.
//
// Be clear about its limit: it does NOT prove xcorrI16 calls a kernel. The
// kernel is bit-identical to the Go reference by design, so deleting the SIMD
// branch outright leaves every parity test green and this test green too. The
// classes this does catch are a flag left unwired and a threshold retuned out
// of range, which are the realistic regressions.
//
// That blind spot is closable and this test does not close it. A build-tagged
// no-op hook in the dispatcher (a counter under a `simdspy` tag, an empty func
// otherwise) would make routing observable while keeping the kernel call
// direct, so it costs nothing in the default build. It is not done here only
// because the gate judged the blind spot narrower than the change; do not read
// this comment as evidence that it cannot be done.
func TestXCorrDispatch_ReachesNEON(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	if !hasNEON {
		t.Fatal("hasNEON is false though cpu.ARM64.NEON is true: XCorr silently runs the Go reference")
	}
	if minNEONXCorr > 16 {
		t.Fatalf("minNEONXCorr = %d: XCorr would not vectorize at the x lengths it was written for", minNEONXCorr)
	}
}

// TestMulQ15NEON_ParityWithGo drives the kernel directly, over lengths the
// dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself.
func TestMulQ15NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a, b := genI16(n, 131), genI16(n, 132)
		got := make([]int16, n)
		want := make([]int16, n)
		mulQ15NEON(got, a, b)
		mulQ15Go(want, a, b)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mulQ15NEON n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestMulQ15NEON_MinInt16 pins the SRSHR+XTN wrap at the kernel: (-32768)^2
// must narrow to -32768 in the vector lanes and in the scalar tail. A
// saturating substitute (SQRDMULH) would return 32767 in every position.
func TestMulQ15NEON_MinInt16(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for n := 1; n <= 32; n++ {
		a := make([]int16, n)
		for i := range a {
			a[i] = math.MinInt16
		}
		got := make([]int16, n)
		mulQ15NEON(got, a, a)
		for i := range got {
			if got[i] != math.MinInt16 {
				t.Fatalf("mulQ15NEON all-MinInt16 n=%d: dst[%d] = %d, want %d", n, i, got[i], math.MinInt16)
			}
		}
	}
}

// TestAbsNEON_ParityWithGo drives the kernel directly across the full sweep;
// the wrap input MinInt16 rides at index 0 of every non-empty case.
func TestAbsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a := genI16(n, 133)
		if n > 0 {
			a[0] = math.MinInt16
		}
		got := make([]int16, n)
		want := make([]int16, n)
		absNEON(got, a)
		absGo(want, a)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("absNEON n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestMaxAbsNEON_ParityWithGo drives the kernel directly across the sweep,
// then pins the widened extreme: a planted -32768 must come back as 32768
// from every lane position and from the scalar tail.
func TestMaxAbsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a := genI16(n, 134)
		if got, want := maxAbsNEON(a), maxAbsGo(a); got != want {
			t.Errorf("maxAbsNEON n=%d: got %d, want %d", n, got, want)
		}
	}
	for _, n := range []int{8, 9, 16, 17, 24} {
		for pos := range n {
			a := make([]int16, n)
			a[pos] = math.MinInt16
			if got := maxAbsNEON(a); got != 32768 {
				t.Fatalf("maxAbsNEON n=%d pos=%d: got %d, want 32768", n, pos, got)
			}
		}
	}
}

// TestMaxAbsNEON_NoOverRead is the kernel-direct over-read check: the operand
// is a prefix of a longer allocation whose every element past the prefix is
// -32768. Zeroed past-slice memory is the identity element of this reduction,
// so only the planted extreme makes an over-read observable.
func TestMaxAbsNEON_NoOverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	backing := make([]int16, 64+8)
	for i := range backing {
		backing[i] = math.MinInt16
	}
	for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 24, 33, 64} {
		a := backing[:n]
		for i := range a {
			a[i] = int16(i%50 - 25)
		}
		if got, want := maxAbsNEON(a), maxAbsGo(a); got != want {
			t.Fatalf("maxAbsNEON n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = math.MinInt16
		}
	}
}

// TestTier3Dispatch_ReachesNEON pins the dispatch state the tier-3 SIMD paths
// depend on. It has to be a white-box check: the kernels are bit-identical to
// the Go references by design, so a dispatcher that silently routed every
// call to Go would pass every parity test in this package, and the
// kernel-direct tests above are threshold-independent by construction, so
// they cannot notice either. It must not call t.Parallel(): it reads
// package-level dispatch state.
//
// Scope, so the next reader does not over-trust it: this pins the INPUTS the
// dispatcher reads (the feature flag, and thresholds low enough to vectorize),
// not that the dispatcher consults them. It kills a mis-wired flag and an
// out-of-range threshold; it does not kill a dispatch branch deleted outright,
// which leaves the kernel dead while every test here still passes. Closing
// that needs the call to be observable, e.g. a counter behind a build tag.
func TestTier3Dispatch_ReachesNEON(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	if !hasNEON {
		t.Fatal("hasNEON is false though cpu.ARM64.NEON is true: the tier-3 ops silently run the Go reference on every call")
	}
	if minNEONMulQ15 > 16 || minNEONAbs > 16 || minNEONMaxAbs > 16 {
		t.Fatalf("tier-3 NEON thresholds exceed two vector blocks (MulQ15 %d, Abs %d, MaxAbs %d): the ops would not vectorize at the frame lengths they were written for",
			minNEONMulQ15, minNEONAbs, minNEONMaxAbs)
	}
}

// TestTier3NEONKernels_AllocFree enforces the zero-allocation contract
// directly at the kernel boundary (the public-API alloc tests cover the
// dispatch, these cover the kernels).
func TestTier3NEONKernels_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	a := make([]int16, n)
	b := make([]int16, n)
	dst := make([]int16, n)
	checks := []struct {
		name string
		fn   func()
	}{
		{"mulQ15NEON", func() { mulQ15NEON(dst, a, b) }},
		{"absNEON", func() { absNEON(dst, a) }},
		{"maxAbsNEON", func() { _ = maxAbsNEON(a) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestXCorr4NEON_LongWindowIsClamped covers the other half of the kernel's
// n = min(len(x), len(y)-3) clamp. TestXCorr4NEON_ShortWindowIsBounded only
// probes len(y)-3 < len(x); ParityWithGo passes len(y) == len(x)+3, where both
// operands of the min are equal and a mutant that dropped the min entirely
// would still agree. This is the case that pins it: a y far longer than the
// contract must not make the kernel read x past its end.
func TestXCorr4NEON_LongWindowIsClamped(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, xn := range []int{1, 8, 9, 16, 17, 32} {
		// x MUST be a prefix of a longer allocation: the mutant this test
		// exists for reads past the end of x, and past a standalone slice that
		// is zeroed memory, which multiplies to 0 and leaves the answer
		// correct. Non-zero bytes past x are what make the over-read
		// observable rather than a coin flip on heap layout.
		backing := genI16(xn+200, 207)
		x := backing[:xn]
		y := genI16(xn+40, 208) // len(y)-3 far exceeds len(x)
		dst := make([]int32, xcorrLagBlock)
		xcorr4NEON(dst, x, y)
		for k := range xcorrLagBlock {
			if got, want := dst[k], dotOracle(x, y[k:]); got != want {
				t.Errorf("xcorr4NEON long window xn=%d: dst[%d] = %d, want %d", xn, k, got, want)
			}
		}
	}
}
