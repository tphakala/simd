//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestAbsPow34AVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	checkAbsPow34Kernel(t, "absPow34AVX", absPow34AVX)
}

func TestAbsPow34SSE_ParityWithGo(t *testing.T) {
	if !cpu.X86.SSE2 {
		t.Skip("SSE2 not available")
	}
	checkAbsPow34Kernel(t, "absPow34SSE", absPow34SSE)
}

// TestAbsPow34Kernels_AllocFree asserts the kernels run allocation-free at the
// kernel boundary, mirroring the public-API alloc test.
func TestAbsPow34Kernels_AllocFree(t *testing.T) {
	const n = 1024
	src := make([]float32, n)
	dst := make([]float32, n)
	kernels := []struct {
		name string
		fn   func()
		skip bool
	}{
		{"absPow34AVX", func() { absPow34AVX(dst, src) }, !cpu.X86.AVX},
		{"absPow34SSE", func() { absPow34SSE(dst, src) }, !cpu.X86.SSE2},
	}
	for _, k := range kernels {
		if k.skip {
			continue
		}
		if got := testing.AllocsPerRun(100, k.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", k.name, got)
		}
	}
}

// TestAbsPow34Dispatch_amd64 runs the public AbsPow34 against the exact kernel the
// direct-branch dispatcher (absPow34_32) should select for the detected CPU
// features, over lengths that straddle the block boundaries. Because AVX, SSE, and
// the Go fallback are bit-identical by design, this output comparison confirms the
// dispatched path returns correct bits but cannot by itself prove WHICH kernel ran
// (a silent Go fallback would produce the same bits); that the SIMD branch is
// actually taken is confirmed by coverage instrumentation on a host that has the
// feature. It is white-box (reads the cpu flags the dispatcher reads) and must not
// call t.Parallel().
func TestAbsPow34Dispatch_amd64(t *testing.T) {
	in := tiledAbsPow34Inputs(64)

	var want func(dst, src []float32)
	switch {
	case cpu.X86.AVX:
		want = absPow34AVX
	case cpu.X86.SSE2:
		want = absPow34SSE
	default:
		want = absPow34Go
	}

	for _, n := range []int{1, 4, 7, 8, 15, 16, 33, len(in)} {
		src := in[:n]
		gotDispatch := make([]float32, n)
		gotKernel := make([]float32, n)
		AbsPow34(gotDispatch, src)
		want(gotKernel, src)
		for i := range gotDispatch {
			if !bitsEqF32(gotDispatch[i], gotKernel[i]) {
				t.Fatalf("n=%d: AbsPow34 dispatched to a different kernel than expected at lane %d: got %v want %v",
					n, i, gotDispatch[i], gotKernel[i])
			}
		}
	}
}
