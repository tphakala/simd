//go:build arm64

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
// remainder of the 4-wide NEON kernel.
func tiledCopySignInputs(minLen int) (mag, sign []float32) {
	bm, bs := copySignInterestingInputs()
	for len(mag) < minLen {
		mag = append(mag, bm...)
		sign = append(sign, bs...)
	}
	return mag, sign
}

// TestCopySignNEON_ParityWithGo drives the hand-encoded NEON kernel directly over
// every prefix length, catching a wrong WORD (the BIT insert or the DUP mask
// broadcast), a mishandled 4-element remainder, a dropped scalar tail, or a lane
// error that SIMD-vs-SIMD agreement would miss. Each result must be bit-identical
// to copySign32Go.
func TestCopySignNEON_ParityWithGo(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	mag, sign := tiledCopySignInputs(64)
	for n := 1; n <= len(mag); n++ {
		m := mag[:n]
		s := sign[:n]
		got := make([]float32, n)
		want := make([]float32, n)
		copySignNEON(got, m, s)
		copySign32Go(want, m, s)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("copySignNEON n=%d lane %d: mag=%v[%#08x] sign=%v[%#08x] got %v[%#08x] want %v[%#08x]",
					n, i, m[i], math.Float32bits(m[i]), s[i], math.Float32bits(s[i]),
					got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}

// TestCopySignNEON_AllocFree asserts the kernel runs allocation-free at the kernel
// boundary.
func TestCopySignNEON_AllocFree(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	const n = 1024
	mag := make([]float32, n)
	sign := make([]float32, n)
	dst := make([]float32, n)
	if got := testing.AllocsPerRun(100, func() { copySignNEON(dst, mag, sign) }); got != 0 {
		t.Errorf("copySignNEON allocated %v times per run, want 0", got)
	}
}

// TestCopySignDispatch_arm64 pins the dispatch inputs copySign32 reads: hasNEON
// must reflect CPU detection so a mis-wired flag cannot silently strand every call
// on the Go path. It is white-box (reads package-level state) and must not call
// t.Parallel(). The NEON length threshold matches the abs/neg family (>= 4),
// verified by driving one below-threshold and one at-threshold length through the
// public CopySign and confirming both match the reference.
func TestCopySignDispatch_arm64(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	for _, n := range []int{3, 4} { // below the >=4 NEON threshold and at it
		mag := make([]float32, n)
		sign := make([]float32, n)
		for i := range mag {
			mag[i] = float32(i)*2.5 - 1
			sign[i] = float32(i) - 1.5 // mixes both signs
		}
		got := make([]float32, n)
		want := make([]float32, n)
		CopySign(got, mag, sign)
		copySign32Go(want, mag, sign)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("CopySign n=%d lane %d: got %v want %v", n, i, got[i], want[i])
			}
		}
	}
}
