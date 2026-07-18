//go:build amd64

package i16

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// interleaveKernel/deinterleaveKernel describe one SIMD tier so the parity and
// no-overwrite checks run identically against AVX2 and SSE2.
type interleaveKernel struct {
	name      string
	available bool
	fn        func(dst, a, b []int16)
}

type deinterleaveKernel struct {
	name      string
	available bool
	fn        func(a, b, src []int16)
}

func interleaveKernels() []interleaveKernel {
	return []interleaveKernel{
		{"AVX2", cpu.X86.AVX2, interleave2AVX2},
		{"SSE2", cpu.X86.SSE2, interleave2SSE2},
	}
}

func deinterleaveKernels() []deinterleaveKernel {
	return []deinterleaveKernel{
		{"AVX2", cpu.X86.AVX2, deinterleave2AVX2},
		{"SSE2", cpu.X86.SSE2, deinterleave2SSE2},
	}
}

func TestInterleave2_ParityWithGo(t *testing.T) {
	for _, k := range interleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range paritySizes {
				a := make([]int16, n)
				b := make([]int16, n)
				fillPattern(a, b)

				gotSIMD := make([]int16, n*2)
				gotGo := make([]int16, n*2)
				k.fn(gotSIMD, a, b)
				interleave2Go(gotGo, a, b)

				for i := range gotGo {
					if gotSIMD[i] != gotGo[i] {
						t.Fatalf("n=%d: interleave2%s[%d] = %d, want %d (Go)", n, k.name, i, gotSIMD[i], gotGo[i])
					}
				}
			}
		})
	}
}

func TestDeinterleave2_ParityWithGo(t *testing.T) {
	for _, k := range deinterleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range paritySizes {
				src := make([]int16, n*2)
				for i := range src {
					src[i] = int16(i) ^ math.MinInt16
				}

				aSIMD := make([]int16, n)
				bSIMD := make([]int16, n)
				aGo := make([]int16, n)
				bGo := make([]int16, n)
				k.fn(aSIMD, bSIMD, src)
				deinterleave2Go(aGo, bGo, src)

				for i := range aGo {
					if aSIMD[i] != aGo[i] {
						t.Fatalf("n=%d: deinterleave2%s a[%d] = %d, want %d (Go)", n, k.name, i, aSIMD[i], aGo[i])
					}
					if bSIMD[i] != bGo[i] {
						t.Fatalf("n=%d: deinterleave2%s b[%d] = %d, want %d (Go)", n, k.name, i, bSIMD[i], bGo[i])
					}
				}
			}
		})
	}
}

// TestInterleave2_NoOverwrite guards the scalar tail: the kernel must not write
// past n*2 output elements even when n is not a multiple of the block.
func TestInterleave2_NoOverwrite(t *testing.T) {
	// Sweep tails on both tiers: 25 and 31 set n&8, so they exercise the 8-wide
	// AVX2 block plus a scalar tail (#149); 23 is the block-skipped path (one
	// AVX2 body + 7 tail; two SSE2 bodies + 7 tail).
	for _, k := range interleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range []int{23, 25, 31} {
				a := make([]int16, n)
				b := make([]int16, n)
				fillPattern(a, b)
				dst := make([]int16, n*2+8)
				for i := range dst {
					dst[i] = math.MaxInt16 // sentinel
				}
				k.fn(dst[:n*2], a, b)
				for i := n * 2; i < len(dst); i++ {
					if dst[i] != math.MaxInt16 {
						t.Errorf("interleave2%s n=%d wrote past end at dst[%d] = %d", k.name, n, i, dst[i])
					}
				}
			}
		})
	}
}

// TestDeinterleave2_NoOverwrite guards both output buffers' scalar tails.
func TestDeinterleave2_NoOverwrite(t *testing.T) {
	// 25 and 31 set n&8 (the 8-wide AVX2 block path, #149); 23 skips the block.
	for _, k := range deinterleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range []int{23, 25, 31} {
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
				k.fn(a[:n], b[:n], src)
				for i := n; i < len(a); i++ {
					if a[i] != math.MaxInt16 {
						t.Errorf("deinterleave2%s n=%d wrote past end of a at [%d] = %d", k.name, n, i, a[i])
					}
					if b[i] != math.MaxInt16 {
						t.Errorf("deinterleave2%s n=%d wrote past end of b at [%d] = %d", k.name, n, i, b[i])
					}
				}
			}
		})
	}
}

// dotKernel describes one amd64 SIMD tier so the dot parity checks run
// identically against AVX2 and SSE2.
type dotKernel struct {
	name      string
	available bool
	fn        func(a, b []int16) int32
}

func dotKernels() []dotKernel {
	return []dotKernel{
		{"AVX2", cpu.X86.AVX2, dotAVX2},
		{"SSE2", cpu.X86.SSE2, dotSSE2},
	}
}

// TestDotAMD64_ParityWithGo exercises each kernel directly rather than through
// DotProduct, so a dispatch threshold change can never quietly turn this into a
// test of the Go reference against itself.
func TestDotAMD64_ParityWithGo(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range dotLengths {
				a, b := genI16(n, 61), genI16(n, 62)
				if got, want := k.fn(a, b), dotGo(a, b); got != want {
					t.Errorf("dot%s n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotAMD64_MinInt16 pins the PMADDWD overflow case: a pair of
// (-32768 * -32768) sums to 2^31 inside one instruction and must wrap to
// MinInt32, matching the scalar reference, rather than saturate.
func TestDotAMD64_MinInt16(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for n := 1; n <= 64; n++ {
				a := make([]int16, n)
				b := make([]int16, n)
				for i := range a {
					a[i], b[i] = math.MinInt16, math.MinInt16
				}
				if got, want := k.fn(a, b), dotOracle(a, b); got != want {
					t.Errorf("dot%s all-MinInt16 n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotAMD64_Clamp verifies the in-assembly min(len(a), len(b)): the kernel
// must not read the longer operand past the shorter one's length.
//
// short MUST be a prefix of a longer allocation. A kernel that consumes past
// the clamped n reads the memory following short, and past a standalone slice
// that is zeroed, which multiplies to 0 and leaves every sum correct. Non-zero
// bytes after short are what make over-consumption observable rather than a
// coin flip on heap layout, and the slack has to cover a whole block of the
// widest kernel here (16 elements, dotAVX2's body). Same reasoning as
// TestXCorr4AMD64_LongWindowIsClamped; without it this test detects nothing on
// amd64 at any n.
//
// This is the only dot test that watches the kernels directly with non-zero
// memory past an operand. TestDotProduct_UnalignedOperands does it through the
// dispatcher, so it cannot reach the kernels below their thresholds.
func TestDotAMD64_Clamp(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range dotLengths {
				if n == 0 {
					continue
				}
				long, short := genI16(n+37, 63), genI16(n+64, 64)[:n]
				if got, want := k.fn(long, short), dotOracle(long, short); got != want {
					t.Errorf("dot%s clamp n=%d: got %d, want %d", k.name, n, got, want)
				}
				if got, want := k.fn(short, long), dotOracle(short, long); got != want {
					t.Errorf("dot%s clamp (swapped) n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotDispatch_ReachesSIMD asserts that DotProduct actually routes to a SIMD
// kernel rather than silently falling back to the Go reference. See the arm64
// counterpart for why this must be a white-box check: the kernels are
// bit-identical to dotGo, so a dead dispatcher passes every parity test.
//
// It must not call t.Parallel(): it reads package-level dispatch state.
func TestDotDispatch_ReachesSIMD(t *testing.T) {
	if hasSSE2 != cpu.X86.SSE2 {
		t.Fatalf("hasSSE2 = %v but cpu.X86.SSE2 = %v: dispatch flag is not wired to CPU detection", hasSSE2, cpu.X86.SSE2)
	}
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	// SSE2 is in the GOAMD64=v1 baseline, so on amd64 the dot product must
	// always have a SIMD path: a false hasSSE2 means every call runs scalar.
	if !hasSSE2 {
		t.Fatal("hasSSE2 is false on amd64: PMADDWD is SSE2 baseline, so DotProduct should never fall back to Go here")
	}
	if minSSE2Dot > 2*8 || minAVX2Dot > 2*16 {
		t.Fatalf("dispatch thresholds too high (SSE2 %d, AVX2 %d): DotProduct would not vectorize at the short lengths it was written for",
			minSSE2Dot, minAVX2Dot)
	}
}

// xcorr4Kernels mirrors dotKernels for the 4-lag cross-correlation tiers.
type xcorr4Kernel struct {
	name      string
	available bool
	fn        func(dst []int32, x, y []int16)
}

func xcorr4Kernels() []xcorr4Kernel {
	return []xcorr4Kernel{
		{"AVX2", cpu.X86.AVX2, xcorr4AVX2},
		{"SSE2", cpu.X86.SSE2, xcorr4SSE2},
	}
}

// xcorr4Lengths sweeps every length 1..80, plus 240 and 248. xcorr4AVX2 has
// three paths whose boundaries interact (16-wide loop, 8-wide block, scalar
// tail), and which of them a call reaches is a function of len(x) % 16, so the
// sweep pins every transition at several block counts. 240 is the aligned long
// case the motivating fixed-point caller uses; 248 is that length plus one
// 8-wide block, so the block is exercised at a high block count too (every other
// length here reaches it at four blocks or fewer).
//
// Note what this cannot do. #145 is a performance defect: the pre-fix kernel is
// bit-exact at every length, so no length list, dense or sparse, could have
// surfaced it, and none of these tests fail on the old code. The benchmarks are
// what make the sawtooth visible. The sweep is regression armor for the new
// paths, nothing more.
var xcorr4Lengths = func() []int {
	lengths := make([]int, 0, 82)
	for xn := 1; xn <= 80; xn++ {
		lengths = append(lengths, xn)
	}
	return append(lengths, 240, 248)
}()

// TestXCorr4AMD64_ParityWithGo drives each 4-lag kernel directly, over lengths
// the dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself.
func TestXCorr4AMD64_ParityWithGo(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, xn := range xcorr4Lengths {
				x := genI16(xn, 121)
				y := genI16(xn+3, 122) // exactly the window the dispatcher passes
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x, y[lag:]); got != want {
						t.Errorf("xcorr4%s xn=%d: dst[%d] = %d, want %d", k.name, xn, lag, got, want)
					}
				}
			}
		})
	}
}

// TestXCorr4AMD64_ShortWindowIsBounded feeds each kernel a y window shorter
// than its contract and asserts it clamps rather than reading past the end.
//
// This is the only test where y, not x, binds the n = min(len(x), len(y)-3)
// clamp, which makes it the only place the 8-wide block's reach into y is
// pinned. That reach has zero slack: the block's lag-3 load spans y bytes 6..21,
// touching y[10], and it runs whenever bit 3 of n is set, so n=8 needs exactly
// len(y) >= 11 and gets exactly 11. yn is therefore swept rather than sampled,
// so yn=11 (n=8, the tight case) is actually produced; the old sampled list
// reached the 8-wide block at a single point and never at its boundary.
//
// y is a prefix of a longer allocation for the same reason x is in
// LongWindowIsClamped, but the case for it is weaker and worth stating honestly:
// no mutant found so far actually needs it. Every y over-read anyone has
// constructed also miscounts, and the miscount is what the assertion catches
// (widening this clamp by one element is caught here either way). It stays
// because it costs one allocation and closes the same structural hole the x side
// has a demonstrated mutant for.
func TestXCorr4AMD64_ShortWindowIsBounded(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			x := genI16(32, 123)
			for yn := 0; yn <= 48; yn++ {
				backing := genI16(yn+200, 211)
				y := backing[:yn]
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y) // must not fault or read past y
				n := max(min(len(x), yn-3), 0)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x[:n], y[min(lag, yn):]); got != want {
						t.Errorf("xcorr4%s short window yn=%d: dst[%d] = %d, want %d", k.name, yn, lag, got, want)
					}
				}
			}
		})
	}
}

// TestXCorrDispatch_ReachesSIMD pins the dispatch STATE that XCorr's SIMD path
// depends on. See the arm64 counterpart for what it does NOT prove (nothing
// here establishes that xcorrI16 calls a kernel, because the kernel is
// bit-identical to the Go reference and a dead dispatcher passes every test)
// and for the build-tagged hook that would close that gap.
func TestXCorrDispatch_ReachesSIMD(t *testing.T) {
	if hasSSE2 != cpu.X86.SSE2 {
		t.Fatalf("hasSSE2 = %v but cpu.X86.SSE2 = %v: dispatch flag is not wired to CPU detection", hasSSE2, cpu.X86.SSE2)
	}
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if !hasSSE2 {
		t.Fatal("hasSSE2 is false on amd64: PMADDWD is SSE2 baseline, so XCorr should never fall back to Go here")
	}
	// One vector block each, matching the kernel bodies. A bound of 2x the
	// block would be a tautology against the real values (8 and 16).
	if minSSE2XCorr > 8 || minAVX2XCorr > 16 {
		t.Fatalf("XCorr thresholds too high (SSE2 %d, AVX2 %d): would not vectorize at the x lengths it was written for",
			minSSE2XCorr, minAVX2XCorr)
	}
}

// TestMulQ15AVX2_ParityWithGo drives the kernel directly, over lengths the
// dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself.
func TestMulQ15AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a, b := genI16(n, 141), genI16(n, 142)
		got := make([]int16, n)
		want := make([]int16, n)
		mulQ15AVX2(got, a, b)
		mulQ15Go(want, a, b)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mulQ15AVX2 n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestMulQ15AVX2_MinInt16 pins VPMULHRSW's wrap at the one product outside
// int16 range: (-32768)^2 rounds to +32768 and must land as -32768 in the
// vector lanes and in the scalar tail. This is the load-bearing runtime check
// for the instruction's non-saturating behavior; a saturating substitute
// would return 32767 in every position.
func TestMulQ15AVX2_MinInt16(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for n := 1; n <= 48; n++ {
		a := make([]int16, n)
		for i := range a {
			a[i] = math.MinInt16
		}
		got := make([]int16, n)
		mulQ15AVX2(got, a, a)
		for i := range got {
			if got[i] != math.MinInt16 {
				t.Fatalf("mulQ15AVX2 all-MinInt16 n=%d: dst[%d] = %d, want %d", n, i, got[i], math.MinInt16)
			}
		}
	}
}

// TestAbsAVX2_ParityWithGo drives the kernel directly across the full sweep;
// the wrap input MinInt16 rides at index 0 of every non-empty case.
func TestAbsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a := genI16(n, 143)
		if n > 0 {
			a[0] = math.MinInt16
		}
		got := make([]int16, n)
		want := make([]int16, n)
		absAVX2(got, a)
		absGo(want, a)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("absAVX2 n=%d: dst[%d] = %d, want %d", n, i, got[i], want[i])
			}
		}
	}
}

// TestMaxAbsAVX2_ParityWithGo drives the kernel directly across the sweep,
// then pins the widened extreme: a planted -32768 must come back as 32768
// from every lane position and from the scalar tail.
func TestMaxAbsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a := genI16(n, 144)
		if got, want := maxAbsAVX2(a), maxAbsGo(a); got != want {
			t.Errorf("maxAbsAVX2 n=%d: got %d, want %d", n, got, want)
		}
	}
	for _, n := range []int{16, 17, 24, 32, 33} {
		for pos := range n {
			a := make([]int16, n)
			a[pos] = math.MinInt16
			if got := maxAbsAVX2(a); got != 32768 {
				t.Fatalf("maxAbsAVX2 n=%d pos=%d: got %d, want 32768", n, pos, got)
			}
		}
	}
}

// TestMaxAbsAVX2_NoOverRead is the kernel-direct over-read check: the operand
// is a prefix of a longer allocation whose every element past the prefix is
// -32768. Zeroed past-slice memory is the identity element of this reduction,
// so only the planted extreme makes an over-read observable.
func TestMaxAbsAVX2_NoOverRead(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	backing := make([]int16, 64+16)
	for i := range backing {
		backing[i] = math.MinInt16
	}
	for _, n := range []int{1, 7, 15, 16, 17, 24, 31, 32, 33, 64} {
		a := backing[:n]
		for i := range a {
			a[i] = int16(i%50 - 25)
		}
		if got, want := maxAbsAVX2(a), maxAbsGo(a); got != want {
			t.Fatalf("maxAbsAVX2 n=%d: got %d, want %d (read past the operand?)", n, got, want)
		}
		for i := range a {
			backing[i] = math.MinInt16
		}
	}
}

// TestTier3Dispatch_ReachesSIMD pins the dispatch state the tier-3 SIMD paths
// depend on. It has to be a white-box check: the kernels are bit-identical to
// the Go references by design, so a dispatcher that silently routed every
// call to Go would pass every parity test in this package, and the
// kernel-direct tests above are threshold-independent by construction, so
// they cannot notice either. Unlike the dot ops there is no SSE2 leg here:
// below AVX2 the Go reference is the intended path. It must not call
// t.Parallel(): it reads package-level dispatch state.
//
// Scope, so the next reader does not over-trust it: this pins the INPUTS the
// dispatcher reads (the feature flag, and thresholds low enough to vectorize),
// not that the dispatcher consults them. It kills a mis-wired flag and an
// out-of-range threshold; it does not kill a dispatch branch deleted outright,
// which leaves the kernel dead while every test here still passes. Closing
// that needs the call to be observable, e.g. a counter behind a build tag.
func TestTier3Dispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2MulQ15 > 32 || minAVX2Abs > 32 || minAVX2MaxAbs > 32 {
		t.Fatalf("tier-3 AVX2 thresholds exceed two vector blocks (MulQ15 %d, Abs %d, MaxAbs %d): the ops would not vectorize at the frame lengths they were written for",
			minAVX2MulQ15, minAVX2Abs, minAVX2MaxAbs)
	}
}

// TestTier3AVX2Kernels_AllocFree enforces the zero-allocation contract
// directly at the kernel boundary (the public-API alloc tests cover the
// dispatch, these cover the kernels).
func TestTier3AVX2Kernels_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	a := make([]int16, n)
	b := make([]int16, n)
	dst := make([]int16, n)
	checks := []struct {
		name string
		fn   func()
	}{
		{"mulQ15AVX2", func() { mulQ15AVX2(dst, a, b) }},
		{"absAVX2", func() { absAVX2(dst, a) }},
		{"maxAbsAVX2", func() { _ = maxAbsAVX2(a) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestXCorr4AMD64_LongWindowIsClamped covers the other half of the kernel's
// n = min(len(x), len(y)-3) clamp. ShortWindowIsBounded only probes
// len(y)-3 < len(x); ParityWithGo passes len(y) == len(x)+3, where both
// operands of the min are equal and a mutant that dropped the min entirely
// would still agree.
//
// This test and ShortWindowIsBounded are the xcorr4 set's over-consumption
// detectors, and the backing prefix below is what makes THIS one work: a kernel
// that consumes past x reads that allocation's non-zero bytes, where past a
// standalone slice it would read zeroed memory that multiplies to 0 and leaves
// every lag correct. ParityWithGo is structurally blind to that at every length
// for exactly this reason, which is why the clamp tests carry the dense sweep
// too: the 8-wide block makes which residues over-consume depend on len(x) % 16.
// A mutant keeping the old ANDQ $15 tail (one that consumes 8 elements past x)
// passes ParityWithGo across the whole sweep and dies here.
//
// Neither test detects a pure over-READ, a widened load whose extra lanes never
// reach the sum. That needs a guard page, which this suite does not do.
func TestXCorr4AMD64_LongWindowIsClamped(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, xn := range xcorr4Lengths {
				// x MUST be a prefix of a longer allocation: the mutant this
				// test exists for reads past the end of x, and past a
				// standalone slice that is zeroed memory, which multiplies to
				// 0 and leaves the answer correct. Non-zero bytes past x are
				// what make the over-read observable rather than a coin flip
				// on heap layout. Without this, the test detects nothing on
				// amd64 at any xn.
				backing := genI16(xn+200, 209)
				x := backing[:xn]
				y := genI16(xn+40, 210) // len(y)-3 far exceeds len(x)
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x, y[lag:]); got != want {
						t.Errorf("xcorr4%s long window xn=%d: dst[%d] = %d, want %d", k.name, xn, lag, got, want)
					}
				}
			}
		})
	}
}
