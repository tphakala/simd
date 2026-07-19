package f32

import (
	"math"
	"math/rand"
	"testing"
)

// Tests for AbsPow34: dst[i] = |src[i]|^(3/4) = sqrt(|x| * sqrt(|x|)). The
// contract is bit-exactness to the fused C form sqrtf(a*sqrtf(a)) on every
// backend, INCLUDING the intermediate float32 overflow to +Inf (|x| >~ 2^85.33)
// and underflow to 0 (tiny |x|). The oracle reproduces that intermediate: it uses
// an independent big.Float for both square roots but a genuine float32 multiply
// in between, so it validates both the correct rounding of each sqrt and the
// saturation contract. See the AbsPow34 doc comment.

// absPow34Oracle is the independent reference. a = |x| (float32); s = correctly-
// rounded sqrt(a) via big.Float; p = a*s as a real float32 product (may overflow
// to +Inf or underflow to 0); result = correctly-rounded sqrt(p) via big.Float.
// bigSqrtToF32 lives in sqrt_rounding_test.go.
func absPow34Oracle(x float32) float32 {
	if x != x { // NaN in -> NaN out (payload not asserted)
		return float32(math.NaN())
	}
	a := float32(math.Abs(float64(x)))
	if math.IsInf(float64(a), 1) {
		return float32(math.Inf(1)) // |+-Inf|^0.75 = +Inf
	}
	if a == 0 {
		return 0
	}
	s := bigSqrtToF32(a)
	p := a * s // genuine float32 product: overflows to +Inf / underflows to 0
	switch {
	case math.IsInf(float64(p), 1):
		return float32(math.Inf(1))
	case p == 0:
		return 0
	default:
		return bigSqrtToF32(p)
	}
}

// absPow34SweepInputs gathers the design sweep: specials and signed zeros, the
// float32 extremes, subnormals (which all underflow the intermediate to 0), a
// dense band around 1.0, fine grids straddling the 2^85.33 overflow and the
// 2^-99.33 underflow thresholds (both signs), and random draws across the whole
// range.
func absPow34SweepInputs(short bool) []float32 {
	in := make([]float32, 0, 1<<16)

	in = append(in,
		0, float32(math.Copysign(0, -1)),
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()),
		math.MaxFloat32, -math.MaxFloat32,
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		1, -1, 16, 81, 256, -256,
	)

	// Subnormals (and negatives): every one underflows the a^1.5 intermediate.
	for bits := uint32(1); bits <= 8192; bits++ {
		v := math.Float32frombits(bits)
		in = append(in, v, -v)
	}

	// Dense band around 1.0.
	oneBits := math.Float32bits(1.0)
	band := uint32(1 << 14)
	if short {
		band = 1 << 10
	}
	for d := uint32(1); d <= band; d++ {
		lo := math.Float32frombits(oneBits - d)
		hi := math.Float32frombits(oneBits + d)
		in = append(in, lo, -lo, hi, -hi)
	}

	// Fine grids straddling the overflow boundary (a^1.5 overflows near
	// a = 2^85.33) and the underflow boundary (a^1.5 underflows near a = 2^-99.33);
	// -85 is included as the design calls for symmetric sampling.
	for _, e := range []int{83, 84, 85, 86, 87, 88, -84, -85, -86, -98, -99, -100, -101} {
		base := math.Ldexp(1, e)
		for f := range 1024 {
			v := float32(base * (1.0 + float64(f)/1024.0))
			in = append(in, v, -v)
		}
	}

	rng := rand.New(rand.NewSource(0xAB0F34)) //nolint:gosec // deterministic test vectors
	randCount := 200000
	if short {
		randCount = 5000
	}
	for range randCount {
		x := math.Float32frombits(rng.Uint32()) // any bit pattern, both signs
		in = append(in, x)
	}

	return in
}

// absPow34InterestingInputs is a compact set of values that stress the AbsPow34
// contract: signed zeros, infinities, NaN, the float32 extremes, the intermediate
// overflow (2^85+) and underflow (2^-100) regions, subnormals, and ordinary
// magnitudes of both signs. The direct-kernel parity tests tile this to a length
// that bears a scalar tail and compare each kernel bit-for-bit to absPow34Go.
func absPow34InterestingInputs() []float32 {
	return []float32{
		0, float32(math.Copysign(0, -1)),
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		math.MaxFloat32, -math.MaxFloat32,
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.Float32frombits(3), math.Float32frombits(100), // subnormals
		1, -1, 2, 0.5, 4, 9, 16, 81, 256, -256, 1000.5, -0.25,
		float32(math.Ldexp(1, 85)), float32(math.Ldexp(1.5, 85)), // overflow region
		float32(math.Ldexp(1, 86)), -float32(math.Ldexp(1, 86)),
		float32(math.Ldexp(1, -100)), float32(math.Ldexp(1, -99)), // underflow region
		float32(math.Ldexp(1, 42)), -float32(math.Ldexp(1, 42)),
	}
}

// tiledAbsPow34Inputs repeats the interesting-value set until it is at least
// minLen long, so a prefix at every length exercises the vector body and every
// scalar-tail remainder of the 8-wide AVX, 4-wide SSE, and 4-wide NEON kernels.
func tiledAbsPow34Inputs(minLen int) []float32 {
	base := absPow34InterestingInputs()
	out := make([]float32, 0, minLen+len(base))
	for len(out) < minLen {
		out = append(out, base...)
	}
	return out
}

// checkAbsPow34Kernel drives kern directly over every prefix length so a
// threshold change in the dispatcher can never quietly reduce this to a test of
// the Go reference against itself. Each result must be bit-identical to
// absPow34Go, including the +Inf/0 saturation lanes and NaN (NaN treated equal).
func checkAbsPow34Kernel(t *testing.T, name string, kern func(dst, src []float32)) {
	t.Helper()
	in := tiledAbsPow34Inputs(64)
	for n := 1; n <= len(in); n++ {
		src := in[:n]
		got := make([]float32, n)
		want := make([]float32, n)
		kern(got, src)
		absPow34Go(want, src)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("%s n=%d: dst[%d] = %v [%#08x], want %v [%#08x] (src=%v)",
					name, n, i, got[i], math.Float32bits(got[i]),
					want[i], math.Float32bits(want[i]), src[i])
			}
		}
	}
}

// TestAbsPow34 checks the dispatched AbsPow34 and the pure-Go absPow34Go against
// the big.Float oracle bit-for-bit, and against each other. The full slice
// exercises the SIMD vector body; a one-element shift moves different values into
// the SIMD scalar tail.
func TestAbsPow34(t *testing.T) {
	in := absPow34SweepInputs(testing.Short())
	t.Logf("sweep size: %d inputs", len(in))

	check := func(name string, got, src []float32) {
		for i := range got {
			want := absPow34Oracle(src[i])
			if !bitsEqF32(got[i], want) {
				t.Fatalf("%s: AbsPow34(%v [%#08x]) = %v [%#08x], want %v [%#08x]",
					name, src[i], math.Float32bits(src[i]),
					got[i], math.Float32bits(got[i]),
					want, math.Float32bits(want))
			}
		}
	}

	dst := make([]float32, len(in))
	AbsPow34(dst, in)
	check("AbsPow34(full)", dst, in)

	ref := make([]float32, len(in))
	absPow34Go(ref, in)
	check("absPow34Go(full)", ref, in)

	if len(in) > 1 {
		src := in[1:]
		d2 := make([]float32, len(src))
		AbsPow34(d2, src)
		check("AbsPow34(shift1)", d2, src)
	}
}

// TestAbsPow34_AllocFree pins the zero-allocation contract; buffers are declared
// inside the measured closure so only genuine per-call heap traffic counts.
func TestAbsPow34_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var src, dst [1000]float32
		AbsPow34(dst[:], src[:])
	}); n != 0 {
		t.Errorf("AbsPow34 forces %v caller allocations per run, want 0", n)
	}
}

// TestAbsPow34_TailUntouched plants sentinels past n=11 (one 8-wide AVX block +
// 3 tail, two 4-wide NEON blocks + 3 tail) so both vector bodies run and both
// scalar tails must stop exactly at n.
func TestAbsPow34_TailUntouched(t *testing.T) {
	const n = 11
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) + 0.5
	}
	dst := make([]float32, n+8)
	for i := range dst {
		dst[i] = float32(math.NaN()) // sentinel: any write shows up as a non-NaN
	}
	sentinelBits := math.Float32bits(dst[0])
	AbsPow34(dst[:n], src)
	for i := n; i < len(dst); i++ {
		if math.Float32bits(dst[i]) != sentinelBits {
			t.Errorf("AbsPow34 wrote past end at dst[%d] = %v", i, dst[i])
		}
	}
}

// TestAbsPow34_Clamp covers mismatched lengths in both directions, the empty
// no-op, and nil operands.
func TestAbsPow34_Clamp(t *testing.T) {
	src := make([]float32, 40)
	for i := range src {
		src[i] = float32(i) * 1.5
	}

	short := make([]float32, 25)
	AbsPow34(short, src) // dst shorter: n = 25
	for i := range short {
		if want := absPow34Oracle(src[i]); !bitsEqF32(short[i], want) {
			t.Fatalf("AbsPow34 short dst: dst[%d] = %v, want %v", i, short[i], want)
		}
	}

	long := make([]float32, 40)
	for i := range long {
		long[i] = -7 // sentinel
	}
	AbsPow34(long, src[:25]) // src shorter: long[25:] untouched
	for i := 25; i < len(long); i++ {
		if long[i] != -7 {
			t.Fatalf("AbsPow34 wrote past clamp at dst[%d] = %v", i, long[i])
		}
	}

	AbsPow34(nil, nil)
	one := []float32{42}
	AbsPow34(one, nil)
	if one[0] != 42 {
		t.Errorf("AbsPow34 wrote on empty input: %v", one)
	}
}

// TestAbsPow34_Aliasing verifies the documented in-place safety: AbsPow34(x, x)
// equals the out-of-place result, since each output depends only on its own
// input element.
func TestAbsPow34_Aliasing(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 33, 64} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i)*0.75 - 3
		}
		want := make([]float32, n)
		AbsPow34(want, src)

		AbsPow34(src, src) // in place
		for i := range src {
			if !bitsEqF32(src[i], want[i]) {
				t.Fatalf("AbsPow34 in-place n=%d: dst[%d] = %v, want %v", n, i, src[i], want[i])
			}
		}
	}
}

// TestAbsPow34_UnalignedOperands sweeps all eight element offsets, holding dst
// and src at different offsets so neither is reliably 16- or 32-byte aligned; an
// aligned-load or aligned-store substitution cannot survive the suite.
func TestAbsPow34_UnalignedOperands(t *testing.T) {
	const span = 320
	base := make([]float32, span)
	for i := range base {
		base[i] = float32(i)*0.5 - 40
	}
	backing := make([]float32, span)
	for _, n := range []int{4, 5, 7, 8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			src := base[off+1 : off+1+n]
			dst := backing[off+3 : off+3+n]
			AbsPow34(dst, src)
			for i := range n {
				if want := absPow34Oracle(src[i]); !bitsEqF32(dst[i], want) {
					t.Fatalf("AbsPow34 unaligned n=%d off=%d: dst[%d] = %v, want %v", n, off, i, dst[i], want)
				}
			}
		}
	}
}

// FuzzAbsPow34 differentially fuzzes the dispatched AbsPow34 against absPow34Go
// over arbitrary bit patterns (organically producing NaN, Inf, and subnormals).
// The two are bit-identical by construction, so any lane may be compared exactly
// (NaN treated as equal to NaN).
func FuzzAbsPow34(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		src := f32sBits(raw)
		got := make([]float32, len(src))
		want := make([]float32, len(src))
		AbsPow34(got, src)
		absPow34Go(want, src)
		for i := range got {
			if !bitsEqF32(got[i], want[i]) {
				t.Fatalf("AbsPow34 lane %d (src=%v [%#08x]): got %v [%#08x] want %v [%#08x] (len=%d)",
					i, src[i], math.Float32bits(src[i]),
					got[i], math.Float32bits(got[i]),
					want[i], math.Float32bits(want[i]), len(src))
			}
		}
	})
}
