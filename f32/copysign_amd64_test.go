//go:build amd64

package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// copySignInterestingInputs returns a compact mag/sign pair set that stresses the
// CopySign contract: signed zeros, infinities, both-signed NaN, the float32
// extremes, subnormals, and normals of both signs.
func copySignInterestingInputs() (mag, sign []float32) {
	m := copySignMagValues()
	s := copySignSignValues()
	mag = make([]float32, 0, len(m))
	sign = make([]float32, 0, len(m))
	for i := range m {
		mag = append(mag, m[i])
		sign = append(sign, s[i%len(s)])
	}
	return mag, sign
}

// tiledCopySignInputs repeats the interesting pairs until at least minLen long so
// a prefix at every length exercises the vector body and every scalar-tail
// remainder of the 8-wide AVX and 4-wide SSE kernels.
func tiledCopySignInputs(minLen int) (mag, sign []float32) {
	bm, bs := copySignInterestingInputs()
	for len(mag) < minLen {
		mag = append(mag, bm...)
		sign = append(sign, bs...)
	}
	return mag, sign
}

// checkCopySignKernel drives kern directly over every prefix length so a
// dispatcher threshold change can never quietly reduce this to a test of the Go
// reference against itself. Each result must be bit-identical to copySign32Go.
func checkCopySignKernel(t *testing.T, name string, kern func(dst, mag, sign []float32)) {
	t.Helper()
	mag, sign := tiledCopySignInputs(64)
	for n := 1; n <= len(mag); n++ {
		m := mag[:n]
		s := sign[:n]
		got := make([]float32, n)
		want := make([]float32, n)
		kern(got, m, s)
		copySign32Go(want, m, s)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("%s n=%d lane %d: mag=%v[%#08x] sign=%v[%#08x] got %v[%#08x] want %v[%#08x]",
					name, n, i, m[i], math.Float32bits(m[i]), s[i], math.Float32bits(s[i]),
					got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}

func TestCopySignAVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	checkCopySignKernel(t, "copySignAVX", copySignAVX)
}

func TestCopySignSSE_ParityWithGo(t *testing.T) {
	if !cpu.X86.SSE2 {
		t.Skip("SSE2 not available")
	}
	checkCopySignKernel(t, "copySignSSE", copySignSSE)
}

// TestCopySignKernels_AllocFree asserts the kernels run allocation-free at the
// kernel boundary, mirroring the public-API alloc test.
func TestCopySignKernels_AllocFree(t *testing.T) {
	const n = 1024
	mag := make([]float32, n)
	sign := make([]float32, n)
	dst := make([]float32, n)
	kernels := []struct {
		name string
		fn   func()
		skip bool
	}{
		{"copySignAVX", func() { copySignAVX(dst, mag, sign) }, !cpu.X86.AVX},
		{"copySignSSE", func() { copySignSSE(dst, mag, sign) }, !cpu.X86.SSE2},
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

// TestCopySignDispatch_amd64 runs the public CopySign against the exact kernel the
// direct-branch dispatcher (copySign32) should select for the detected CPU
// features, over lengths that straddle the block boundaries. Because AVX, SSE, and
// the Go fallback are bit-identical by design, this output comparison confirms the
// dispatched path returns correct bits but cannot by itself prove WHICH kernel ran
// (a silent Go fallback produces the same bits); that the SIMD branch is actually
// taken is confirmed by coverage instrumentation on a host that has the feature.
// It is white-box (reads the cpu flags the dispatcher reads) and must not call
// t.Parallel().
func TestCopySignDispatch_amd64(t *testing.T) {
	mag, sign := tiledCopySignInputs(64)

	var want func(dst, mag, sign []float32)
	switch {
	case cpu.X86.AVX:
		want = copySignAVX
	case cpu.X86.SSE2:
		want = copySignSSE
	default:
		want = copySign32Go
	}

	for _, n := range []int{1, 4, 7, 8, 15, 16, 33, len(mag)} {
		m := mag[:n]
		s := sign[:n]
		gotDispatch := make([]float32, n)
		gotKernel := make([]float32, n)
		CopySign(gotDispatch, m, s)
		want(gotKernel, m, s)
		for i := range gotDispatch {
			if !bitsEqF32(gotDispatch[i], gotKernel[i]) {
				t.Fatalf("n=%d: CopySign dispatched to a different kernel than expected at lane %d: got %v want %v",
					n, i, gotDispatch[i], gotKernel[i])
			}
		}
	}
}
