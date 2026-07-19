package f32

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// This file empirically establishes the documented Sqrt contract: f32.Sqrt is
// IEEE-754 correctly-rounded (round-to-nearest, ties to even) on every backend,
// including the pure-Go fallback whose float32(math.Sqrt(float64(x))) is a double
// rounding. The oracle is an independent math/big.Float square root rounded to
// float32; if the dispatched SIMD path AND the pure-Go path both match it
// bit-for-bit across the sweep, the doc claim holds on the running architecture
// (NEON here, AVX/SSE under GOARCH=amd64). See the Sqrt doc comment.

// bigSqrtToF32 returns the correctly-rounded float32 square root of v via a
// 240-bit big.Float, independent of math.Sqrt and of the hardware sqrt. 240 bits
// is far above 2*24+2, so rounding the big result to float32 is itself a benign
// double rounding and yields the correctly-rounded binary32 sqrt.
func bigSqrtToF32(v float32) float32 {
	bf := new(big.Float).SetPrec(240).SetFloat64(float64(v)) // exact: float32 -> float64
	bf.Sqrt(bf)
	out, _ := bf.Float32()
	return out
}

// sqrtCorrectOracle is the correctly-rounded reference for sqrt over the tested
// domain: non-negative finite inputs plus 0, +Inf, and NaN. Negatives (which the
// sweep does not use) and NaN map to NaN, matching every backend.
func sqrtCorrectOracle(x float32) float32 {
	switch {
	case x != x: // NaN
		return float32(math.NaN())
	case x < 0:
		return float32(math.NaN())
	case math.IsInf(float64(x), 1):
		return float32(math.Inf(1))
	case x == 0:
		return x // sqrt(+0) = +0, sqrt(-0) = -0 (sign preserved)
	default:
		return bigSqrtToF32(x)
	}
}

// sqrtSweepInputs builds the representative input sweep from the design: specials
// (0, -0, +Inf, NaN), small integers and half-integers, all powers of two and
// four, a dense contiguous band of float32 bit patterns around 1.0, a full-range
// exponent sweep, subnormals, constructed near-midpoint targets, and random draws
// across the whole positive range. Exhaustive portions shrink under -short.
func sqrtSweepInputs(short bool) []float32 {
	in := make([]float32, 0, 1<<21)

	// Specials, then every small integer and half-integer up to 2048. The perfect
	// squares among them have exact roots; the rest have irrational roots that
	// exercise the rounding decision.
	in = append(in, 0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.NaN()))
	for k := 0; k <= 2048; k++ {
		in = append(in, float32(k), float32(k)*0.5) // all integers and half-integers 0..2048
	}

	// All powers of two (even exponents are also the powers of four): exact roots
	// at 2^(2k), and 2^k*sqrt(2) irrational roots at 2^(2k+1).
	for e := -149; e <= 127; e++ {
		in = append(in, float32(math.Ldexp(1, e)))
	}

	// Dense contiguous band of float32 values straddling 1.0. This is where the
	// bulk of the rounding decisions live.
	oneBits := math.Float32bits(1.0)
	band := uint32(1 << 18)
	if short {
		band = 1 << 12
	}
	for d := uint32(1); d <= band; d++ {
		in = append(in, math.Float32frombits(oneBits-d), math.Float32frombits(oneBits+d))
	}

	// Full-range exponent sweep: every binade, several mantissa fractions.
	for e := -140; e <= 120; e++ {
		base := math.Ldexp(1, e)
		for f := range 16 {
			in = append(in, float32(base*(1.0+float64(f)/16.0)))
		}
	}

	// Subnormals: the smallest positive float32 values.
	for bits := uint32(1); bits <= 4096; bits++ {
		in = append(in, math.Float32frombits(bits))
	}

	rng := rand.New(rand.NewSource(0x59A17)) //nolint:gosec // deterministic test vectors

	// Constructed near-midpoint targets: for random representable r, the exact
	// midpoint m between r and its successor is squared and rounded to float32, so
	// sqrt of the result sits on a rounding boundary. Feeding m*m and its float32
	// neighbours probes whether the backend rounds the boundary the same way the
	// oracle does.
	midCount := 40000
	if short {
		midCount = 2000
	}
	for range midCount {
		e := rng.Intn(120) - 60
		r := float32(math.Ldexp(1+rng.Float64(), e))
		next := math.Nextafter32(r, float32(math.Inf(1)))
		m := (float64(r) + float64(next)) / 2 // exact in float64: r, next are adjacent float32
		x := float32(m * m)
		in = append(in,
			x,
			math.Nextafter32(x, float32(math.Inf(1))),
			math.Nextafter32(x, 0),
		)
	}

	// Random draws across the whole positive float32 range (any bit pattern,
	// keeping only non-negative finite values).
	randCount := 200000
	if short {
		randCount = 5000
	}
	for range randCount {
		x := math.Float32frombits(rng.Uint32() & 0x7fffffff) // clear sign bit
		if math.IsInf(float64(x), 0) || x != x {
			continue
		}
		in = append(in, x)
	}

	return in
}

// TestSqrt_CorrectlyRounded verifies both the dispatched f32.Sqrt (the running
// architecture's SIMD kernel, NEON here or AVX/SSE under Rosetta) and the pure-Go
// sqrt32Go against the big.Float oracle, bit-for-bit, over the whole sweep. A
// non-multiple-of-4 total length plus a one-element shift exercise the SIMD
// scalar tail with different values in it.
func TestSqrt_CorrectlyRounded(t *testing.T) {
	in := sqrtSweepInputs(testing.Short())
	t.Logf("sweep size: %d inputs", len(in))

	check := func(name string, got []float32, src []float32) {
		for i := range got {
			want := sqrtCorrectOracle(src[i])
			if !bitsEqF32(got[i], want) {
				t.Fatalf("%s: Sqrt(%v [%#08x]) = %v [%#08x], want %v [%#08x]",
					name, src[i], math.Float32bits(src[i]),
					got[i], math.Float32bits(got[i]),
					want, math.Float32bits(want))
			}
		}
	}

	// Dispatched path over the full slice (SIMD vector body + tail).
	dst := make([]float32, len(in))
	Sqrt(dst, in)
	check("Sqrt(full)", dst, in)

	// Shifted by one so different values land in the SIMD scalar tail.
	if len(in) > 1 {
		src := in[1:]
		d2 := make([]float32, len(src))
		Sqrt(d2, src)
		check("Sqrt(shift1)", d2, src)
	}

	// Small lengths (1..3) exercise the arm64 NEON dispatcher's sub-block Go
	// fallback (len < 4) and the scalar-only path on every backend, which the large
	// full and shift1 slices above never reach. A mid-sweep window keeps the values
	// normal rather than the leading specials.
	mid := len(in) / 2
	for n := 1; n <= 3 && mid+n <= len(in); n++ {
		src := in[mid : mid+n]
		dn := make([]float32, n)
		Sqrt(dn, src)
		check("Sqrt(smallN)", dn, src)
	}

	// Pure-Go path directly: this is the double-rounding claim under test,
	// independent of which SIMD kernel the dispatcher would pick.
	dgo := make([]float32, len(in))
	sqrt32Go(dgo, in)
	check("sqrt32Go", dgo, in)
}
