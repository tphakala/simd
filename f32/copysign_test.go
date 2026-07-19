package f32

import (
	"math"
	"testing"
)

// Tests for CopySign: dst[i] = |mag[i]| carrying sign[i]'s sign bit, exactly the
// IEEE copysign bit formula (clear mag's sign bit, OR in sign's). The op is pure
// bit manipulation with no rounding, so it is bit-identical across amd64
// (VANDPS/VORPS), arm64 (BIT), and copySign32Go, and equals math.Copysign
// elementwise. The oracle is therefore just the bit formula / math.Copysign; no
// big.Float is needed.

// negNaN32 is a float32 NaN with the sign bit set, so the sign kernels have a
// negative-NaN sign operand to exercise.
func negNaN32() float32 {
	return math.Float32frombits(math.Float32bits(float32(math.NaN())) | (1 << float32SignBitPos))
}

// copySignOracle is the independent reference via math.Copysign. For finite and
// infinite magnitudes it is exact (only the sign bit moves, |mag| is already an
// exact float32); for a NaN magnitude it returns a NaN, which bitsEqF32 treats as
// equal regardless of payload.
func copySignOracle(mag, sign float32) float32 {
	return float32(math.Copysign(float64(mag), float64(sign)))
}

// copySignMagValues is the magnitude side of the value matrix: signed zeros,
// infinities, both-signed NaN, the float32 extremes, subnormals of both signs,
// and ordinary normals of both signs.
func copySignMagValues() []float32 {
	return []float32{
		0, float32(math.Copysign(0, -1)),
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), negNaN32(),
		math.MaxFloat32, -math.MaxFloat32,
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.Float32frombits(7),  // positive subnormal
		math.Float32frombits(70), // another subnormal
		3.5, -3.5, 1, -1, 1e30, -1e-30,
	}
}

// copySignSignValues is the sign side of the value matrix: signed zeros, +-1,
// +-Inf, and +-NaN. Every one has an unambiguous sign bit.
func copySignSignValues() []float32 {
	return []float32{
		0, float32(math.Copysign(0, -1)),
		1, -1,
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), negNaN32(),
	}
}

// signBitSet reports whether v's IEEE sign bit (bit 31) is set.
func signBitSet(v float32) bool {
	return math.Float32bits(v)&(1<<float32SignBitPos) != 0
}

// TestCopySign checks the dispatched CopySign, the pure-Go copySign32Go, and the
// math.Copysign oracle against the exact bit formula over a length sweep and the
// full mag x sign value matrix, then asserts the sign-of-sign predicate directly
// (a +0.0 sign never negates, a -0.0 sign always does).
func TestCopySign(t *testing.T) {
	mags := copySignMagValues()
	// Add a negative subnormal so both signs of subnormals are present.
	mags = append(mags, -math.Float32frombits(7))
	signs := copySignSignValues()

	// Flatten the matrix into parallel mag/sign slices, then run every prefix
	// length so the SIMD vector body and every scalar-tail remainder are covered.
	flatMag := make([]float32, 0, len(mags)*len(signs))
	flatSign := make([]float32, 0, len(mags)*len(signs))
	for _, m := range mags {
		for _, s := range signs {
			flatMag = append(flatMag, m)
			flatSign = append(flatSign, s)
		}
	}

	// Per-element expected via the exact bit formula.
	expect := func(m, s float32) float32 {
		bits := (math.Float32bits(m) &^ (1 << float32SignBitPos)) | (math.Float32bits(s) & (1 << float32SignBitPos))
		return math.Float32frombits(bits)
	}

	n := len(flatMag)
	for length := 1; length <= n; length++ {
		mag := flatMag[:length]
		sign := flatSign[:length]

		gotDispatch := make([]float32, length)
		gotGo := make([]float32, length)
		CopySign(gotDispatch, mag, sign)
		copySign32Go(gotGo, mag, sign)

		for i := range gotDispatch {
			want := expect(mag[i], sign[i])
			if !bitsEqF32(gotDispatch[i], want) {
				t.Fatalf("CopySign len=%d lane %d: mag=%v[%#08x] sign=%v[%#08x] got %v[%#08x] want %v[%#08x]",
					length, i, mag[i], math.Float32bits(mag[i]), sign[i], math.Float32bits(sign[i]),
					gotDispatch[i], math.Float32bits(gotDispatch[i]), want, math.Float32bits(want))
			}
			if !bitsEqF32(gotGo[i], want) {
				t.Fatalf("copySign32Go len=%d lane %d: got %v[%#08x] want %v[%#08x]",
					length, i, gotGo[i], math.Float32bits(gotGo[i]), want, math.Float32bits(want))
			}
			if oracle := copySignOracle(mag[i], sign[i]); !bitsEqF32(gotDispatch[i], oracle) {
				t.Fatalf("CopySign vs math.Copysign len=%d lane %d: got %v[%#08x] oracle %v[%#08x]",
					length, i, gotDispatch[i], math.Float32bits(gotDispatch[i]), oracle, math.Float32bits(oracle))
			}
		}
	}

	// Direct sign-predicate assertions on a finite non-zero magnitude, where the
	// result sign is observable (a NaN result's sign is not checked because
	// bitsEqF32 is payload-agnostic).
	const mag = 3.5
	dst := make([]float32, 1)
	for _, s := range signs {
		CopySign(dst, []float32{mag}, []float32{s})
		if got, want := signBitSet(dst[0]), signBitSet(s); got != want {
			t.Errorf("CopySign(%v, %v): result sign bit = %v, want %v (result=%v)", mag, s, got, want, dst[0])
		}
	}
	// The -0.0 sign must negate; the +0.0 sign must not.
	CopySign(dst, []float32{mag}, []float32{float32(math.Copysign(0, -1))})
	if dst[0] != -mag {
		t.Errorf("CopySign(%v, -0.0) = %v, want %v (sign bit predicate, not comparison)", mag, dst[0], -mag)
	}
	CopySign(dst, []float32{mag}, []float32{0})
	if dst[0] != mag {
		t.Errorf("CopySign(%v, +0.0) = %v, want %v", mag, dst[0], mag)
	}
}

// TestCopySign_AllocFree pins the zero-allocation contract; buffers live inside
// the measured closure so only genuine per-call heap traffic counts.
func TestCopySign_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var mag, sign, dst [1000]float32
		CopySign(dst[:], mag[:], sign[:])
	}); n != 0 {
		t.Errorf("CopySign forces %v caller allocations per run, want 0", n)
	}
}

// TestCopySign_TailUntouched plants non-zero sentinels past n=11 (one 8-wide AVX
// block + 3 tail, two 4-wide SSE/NEON blocks + 3 tail) so both vector bodies run
// and both scalar tails must stop exactly at n.
func TestCopySign_TailUntouched(t *testing.T) {
	const n = 11
	mag := make([]float32, n)
	sign := make([]float32, n)
	for i := range mag {
		mag[i] = float32(i) + 0.5
		sign[i] = -1 // negate every result so a stray write is visible
	}
	dst := make([]float32, n+8)
	const sentinel = float32(-987.125)
	for i := range dst {
		dst[i] = sentinel
	}
	CopySign(dst[:n], mag, sign)
	for i := n; i < len(dst); i++ {
		if dst[i] != sentinel {
			t.Errorf("CopySign wrote past end at dst[%d] = %v, want sentinel %v", i, dst[i], sentinel)
		}
	}
}

// TestCopySign_Clamp covers mismatched lengths across all three slices, in both
// directions, and the empty no-op.
func TestCopySign_Clamp(t *testing.T) {
	mag := make([]float32, 40)
	sign := make([]float32, 40)
	for i := range mag {
		mag[i] = float32(i) * 1.5
		sign[i] = float32(1 - 2*(i&1)) // alternate +1 / -1
	}

	// dst shortest: n = 25.
	short := make([]float32, 25)
	CopySign(short, mag, sign)
	for i := range short {
		if want := copySignOracle(mag[i], sign[i]); !bitsEqF32(short[i], want) {
			t.Fatalf("CopySign short dst: dst[%d] = %v, want %v", i, short[i], want)
		}
	}

	// mag shortest: n = 15; dst[15:] must be untouched.
	long := make([]float32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	CopySign(long, mag[:15], sign)
	for i := range 15 {
		if want := copySignOracle(mag[i], sign[i]); !bitsEqF32(long[i], want) {
			t.Fatalf("CopySign mag-short: dst[%d] = %v, want %v", i, long[i], want)
		}
	}
	for i := 15; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("CopySign wrote past clamp (mag short) at dst[%d] = %v", i, long[i])
		}
	}

	// sign shortest: n = 10; dst[10:] must be untouched.
	long2 := make([]float32, 40)
	for i := range long2 {
		long2[i] = -7
	}
	CopySign(long2, mag, sign[:10])
	for i := range 10 {
		if want := copySignOracle(mag[i], sign[i]); !bitsEqF32(long2[i], want) {
			t.Fatalf("CopySign sign-short: dst[%d] = %v, want %v", i, long2[i], want)
		}
	}
	for i := 10; i < len(long2); i++ {
		if long2[i] != -7 {
			t.Fatalf("CopySign wrote past clamp (sign short) at dst[%d] = %v", i, long2[i])
		}
	}

	// Empty no-op: a non-empty dst with empty inputs must be untouched.
	CopySign(nil, nil, nil)
	one := []float32{42}
	CopySign(one, nil, nil)
	if one[0] != 42 {
		t.Errorf("CopySign wrote on empty input: %v", one)
	}
	CopySign(one, []float32{1}, nil)
	if one[0] != 42 {
		t.Errorf("CopySign wrote with empty sign: %v", one)
	}
}

// TestCopySign_Aliasing verifies the documented in-place safety: dst may alias
// mag and/or sign since each output depends only on its own index.
func TestCopySign_Aliasing(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 33, 64} {
		mag := make([]float32, n)
		sign := make([]float32, n)
		for i := range mag {
			mag[i] = float32(i)*0.75 - 3
			sign[i] = float32(i)*0.5 - 5 // mixes both signs across the range
		}

		// dst == mag: result overwrites the magnitude buffer.
		want := make([]float32, n)
		CopySign(want, mag, sign)
		a := append([]float32(nil), mag...)
		CopySign(a, a, sign)
		for i := range a {
			if !bitsEqF32(a[i], want[i]) {
				t.Fatalf("CopySign dst==mag n=%d: dst[%d] = %v, want %v", n, i, a[i], want[i])
			}
		}

		// dst == sign: result overwrites the sign buffer.
		b := append([]float32(nil), sign...)
		CopySign(b, mag, b)
		for i := range b {
			if !bitsEqF32(b[i], want[i]) {
				t.Fatalf("CopySign dst==sign n=%d: dst[%d] = %v, want %v", n, i, b[i], want[i])
			}
		}

		// Full self-alias CopySign(x, x, x): |x| with x's own sign is x itself, an
		// identity that holds for every float32 (including NaN payloads and signed zeros).
		c := append([]float32(nil), mag...)
		selfWant := make([]float32, n)
		CopySign(selfWant, mag, mag)
		CopySign(c, c, c)
		for i := range c {
			if !bitsEqF32(c[i], selfWant[i]) {
				t.Fatalf("CopySign(x,x,x) n=%d: dst[%d] = %v, want %v", n, i, c[i], selfWant[i])
			}
			if !bitsEqF32(c[i], mag[i]) {
				t.Fatalf("CopySign(x,x,x) n=%d: dst[%d] = %v, want identity %v", n, i, c[i], mag[i])
			}
		}
	}
}

// TestCopySign_UnalignedOperands sweeps all eight element offsets, holding dst,
// mag, and sign at different offsets so none is reliably 16- or 32-byte aligned;
// an aligned-load or aligned-store substitution cannot survive the suite.
func TestCopySign_UnalignedOperands(t *testing.T) {
	const span = 320
	magBase := make([]float32, span)
	signBase := make([]float32, span)
	for i := range magBase {
		magBase[i] = float32(i)*0.5 - 40
		signBase[i] = float32(i)*0.25 - 30
	}
	backing := make([]float32, span)
	for _, n := range []int{4, 5, 7, 8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			mag := magBase[off+1 : off+1+n]
			sign := signBase[off+2 : off+2+n]
			dst := backing[off+3 : off+3+n]
			CopySign(dst, mag, sign)
			for i := range n {
				if want := copySignOracle(mag[i], sign[i]); !bitsEqF32(dst[i], want) {
					t.Fatalf("CopySign unaligned n=%d off=%d: dst[%d] = %v, want %v", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// FuzzCopySign differentially fuzzes the dispatched CopySign against copySign32Go
// over arbitrary bit patterns for both mag and sign (organically producing NaN,
// Inf, and subnormals). The two are bit-identical by construction, so any lane is
// compared exactly (NaN treated as equal to NaN).
func FuzzCopySign(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		all := f32sBits(raw)
		half := len(all) / 2
		mag := all[:half]
		sign := all[half : half*2]
		got := make([]float32, half)
		want := make([]float32, half)
		CopySign(got, mag, sign)
		copySign32Go(want, mag, sign)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("CopySign lane %d (mag=%v[%#08x] sign=%v[%#08x]): got %v[%#08x] want %v[%#08x] (len=%d)",
					i, mag[i], math.Float32bits(mag[i]), sign[i], math.Float32bits(sign[i]),
					got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]), half)
			}
		}
	})
}
