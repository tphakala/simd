package f32

import (
	"encoding/binary"
	"math"
	"testing"
)

// Differential fuzz targets for f32: the dispatched public op must agree with
// the package's pure-Go reference for every length. The dominant SIMD bug class
// is tail/remainder handling at arbitrary lengths around the 8/16-lane unrolls,
// which fixed-length parity tests cannot cover; the seeds bracket those
// boundaries and the fuzzer widens the space. Seeds run under plain `go test`;
// `go test -fuzz=FuzzXxx` explores further.
//
// Comparison policy is per op class:
//   - Pure movement and sign ops (Abs, Neg, InterleaveN/DeinterleaveN) copy or
//     flip bits, so they are bit-identical including NaN payloads: exact compare.
//   - Exactly-rounded element ops (Add, Mul, Sub) are single IEEE operations:
//     bit-exact with NaN treated as equal to NaN.
//   - Min/Max/Clamp select values; their NaN handling differs between VMINPS/
//     VMAXPS and the scalar `<`/`>` reference, so they run on finite inputs only.
//   - Accumulating ops (Sum, DotProduct, FMA, CumulativeSum, ConvolveValid,
//     ConvolveDecimate) reorder additions or fuse the multiply-add, so they are
//     compared within a rigorous epsilon * sum-of-magnitudes bound on bounded
//     [-1,1] inputs (which also avoids catastrophic-cancellation false positives).

// eps32 is the float32 unit roundoff (2^-23).
const eps32 = 1.1920929e-7

// f32sBits reinterprets raw as little-endian float32s, one per 4-byte chunk.
// This organically produces NaN, Inf, and denormals for the bit-exact targets.
func f32sBits(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// f32sUnit maps each 4-byte chunk to a finite float32 in [-1, 1), keeping the
// accumulating targets well-conditioned so the epsilon bound holds.
func f32sUnit(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		u := int32(binary.LittleEndian.Uint32(raw[i*4:]))
		out[i] = float32(u) / 2147483648.0 // / 2^31
	}
	return out
}

// f32sFinite is f32sBits with NaN and Inf remapped to a finite value, for the
// select ops (Min/Max/Clamp) whose non-finite handling differs by design.
func f32sFinite(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		u := binary.LittleEndian.Uint32(raw[i*4:])
		f := math.Float32frombits(u)
		if f != f || math.IsInf(float64(f), 0) {
			f = float32(int32(u)) / 65536.0
		}
		out[i] = f
	}
	return out
}

func addByteLenSeeds(f *testing.F) {
	f.Helper()
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257} {
		raw := make([]byte, n*4)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

// eqF32 reports whether two results are equal for differential purposes: any NaN
// equals any NaN (payload-agnostic), +0.0 equals -0.0 (the SIMD sign kernels and
// the scalar reference legitimately disagree on the sign of a zero result, which
// is numerically irrelevant), and every other value must be bit-identical.
func eqF32(g, w float32) bool {
	switch {
	case math.IsNaN(float64(g)) && math.IsNaN(float64(w)):
		return true
	case g == 0 && w == 0:
		return true
	default:
		return math.Float32bits(g) == math.Float32bits(w)
	}
}

// exactEqualF32 fails on the first lane that differs (per eqF32).
func exactEqualF32(t *testing.T, op string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", op, len(got), len(want))
	}
	for i := range got {
		if !eqF32(got[i], want[i]) {
			t.Fatalf("%s: lane %d got %x want %x (len=%d)", op, i, math.Float32bits(got[i]), math.Float32bits(want[i]), len(got))
		}
	}
}

// sumTol bounds how far two summation orders of nTerms terms whose magnitudes
// total scaleAbs may diverge. Each order is within ~nTerms*u*scaleAbs of the true
// value (u = 2^-24), so they lie within ~2*nTerms*u of each other; eps32 = 2^-23
// = 2u plus an 8-term slack gives margin. A real tail bug that drops a term
// shifts the result by O(term magnitude), far exceeding this bound, so it is
// still caught.
func sumTol(nTerms int, scaleAbs float64) float64 {
	return (float64(nTerms)+8)*eps32*scaleAbs + 1e-6
}

func FuzzF32ElementwiseExact(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]float32, h)
		want := make([]float32, h)

		Add(got, a, b)
		addGo(want, a, b)
		exactEqualF32(t, "Add", got, want)
		Mul(got, a, b)
		mulGo(want, a, b)
		exactEqualF32(t, "Mul", got, want)
		Sub(got, a, b)
		subGo(want, a, b)
		exactEqualF32(t, "Sub", got, want)
		Abs(got, a)
		absGo(want, a)
		exactEqualF32(t, "Abs", got, want)
		Neg(got, a)
		negGo(want, a)
		exactEqualF32(t, "Neg", got, want)
	})
}

func FuzzF32MinMaxClamp(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		a := f32sFinite(raw)
		if len(a) == 0 {
			return
		}
		if got, want := Min(a), minGo(a); !eqF32(got, want) {
			t.Fatalf("Min got %v want %v (len=%d)", got, want, len(a))
		}
		if got, want := Max(a), maxGo(a); !eqF32(got, want) {
			t.Fatalf("Max got %v want %v (len=%d)", got, want, len(a))
		}
		// Clamp to a finite range derived from the data so bounds are ordered.
		lo, hi := a[0], a[len(a)-1]
		if lo > hi {
			lo, hi = hi, lo
		}
		got := make([]float32, len(a))
		want := make([]float32, len(a))
		Clamp(got, a, lo, hi)
		clampGo(want, a, lo, hi)
		exactEqualF32(t, "Clamp", got, want)
	})
}

// closeReduceF32 asserts got and want agree within sumTol(nTerms, scale).
func closeReduceF32(t *testing.T, op string, got, want float32, nTerms int, scale float64) {
	t.Helper()
	tol := sumTol(nTerms, scale)
	if d := math.Abs(float64(got) - float64(want)); d > tol {
		t.Fatalf("%s: got %v want %v |diff|=%g tol=%g", op, got, want, d, tol)
	}
}

func sumAbsF32(xs ...[]float32) float64 {
	var s float64
	for _, x := range xs {
		for _, v := range x {
			s += math.Abs(float64(v))
		}
	}
	return s
}

func FuzzF32Reductions(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sUnit(raw)
		// Sum over the whole slice.
		closeReduceF32(t, "Sum", Sum(v), sumGo(v), len(v), sumAbsF32(v))

		// DotProduct over equal halves.
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		var dotScale float64
		for i := range a {
			dotScale += math.Abs(float64(a[i]) * float64(b[i]))
		}
		closeReduceF32(t, "DotProduct", DotProduct(a, b), dotProductGo(a, b), h, dotScale)
	})
}

func FuzzF32FMA(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sUnit(raw)
		third := len(v) / 3
		if third == 0 {
			return
		}
		a, b, c := v[:third], v[third:2*third], v[2*third:3*third]
		got := make([]float32, third)
		want := make([]float32, third)
		FMA(got, a, b, c)
		fmaGo(want, a, b, c)
		// Fused vs unfused differ by at most ~1 ulp of |a*b|+|c| per lane.
		for i := range got {
			scale := math.Abs(float64(a[i])*float64(b[i])) + math.Abs(float64(c[i]))
			tol := 4*eps32*scale + 1e-6
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > tol {
				t.Fatalf("FMA lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func FuzzF32CumulativeSum(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		a := f32sUnit(raw)
		got := make([]float32, len(a))
		want := make([]float32, len(a))
		CumulativeSum(got, a)
		cumulativeSum32Go(want, a)
		var prefixAbs float64
		for i := range a {
			prefixAbs += math.Abs(float64(a[i]))
			tol := sumTol(i+1, prefixAbs)
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > tol {
				t.Fatalf("CumulativeSum lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func FuzzF32Convolve(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sUnit(raw)
		if len(v) < 4 {
			return
		}
		kLen := 1 + int(raw[0])%min(len(v)-1, 16)
		kernel := v[:kLen]
		signal := v[kLen:]
		if len(signal) < kLen {
			return
		}
		// ConvolveValid: each output is a kLen-term dot of |x|<=1 values.
		validLen := len(signal) - kLen + 1
		dst := make([]float32, validLen)
		ref := make([]float32, validLen)
		ConvolveValid(dst, signal, kernel)
		convolveValid32Go(ref, signal, kernel)
		// Each output is a kLen-term dot of |x|<=1 values, so Σ|terms| <= kLen.
		convTol := sumTol(kLen, float64(kLen))
		for i := range dst {
			if d := math.Abs(float64(dst[i]) - float64(ref[i])); d > convTol {
				t.Fatalf("ConvolveValid lane %d got %v want %v |diff|=%g tol=%g", i, dst[i], ref[i], d, convTol)
			}
		}

		// ConvolveDecimate with derived factor/phase.
		factor := 1 + int(raw[1])%4
		phase := int(raw[2]) % factor
		span := len(signal) - kLen - phase
		if span < 0 {
			return
		}
		n := span/factor + 1
		ddst := make([]float32, n)
		dref := make([]float32, n)
		ConvolveDecimate(ddst, signal, kernel, factor, phase)
		convolveDecimate32Go(dref, signal, kernel, factor, phase)
		for i := range ddst {
			if d := math.Abs(float64(ddst[i]) - float64(dref[i])); d > convTol {
				t.Fatalf("ConvolveDecimate lane %d (factor=%d phase=%d) got %v want %v |diff|=%g tol=%g", i, factor, phase, ddst[i], dref[i], d, convTol)
			}
		}
	})
}

func FuzzF32InterleaveN(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sBits(raw)
		if len(v) < 2 {
			return
		}
		nc := 1 + int(raw[0])%10
		streamLen := len(v) / nc
		if streamLen == 0 {
			return
		}
		srcs := make([][]float32, nc)
		for c := range srcs {
			srcs[c] = v[c*streamLen : (c+1)*streamLen]
		}
		dst := make([]float32, nc*streamLen)
		ref := make([]float32, nc*streamLen)
		InterleaveN(dst, srcs)
		interleaveNGo(ref, srcs, streamLen)
		exactEqualF32(t, "InterleaveN", dst, ref)

		// DeinterleaveN is the inverse: split the interleaved buffer back.
		dsts := make([][]float32, nc)
		drefs := make([][]float32, nc)
		for c := range dsts {
			dsts[c] = make([]float32, streamLen)
			drefs[c] = make([]float32, streamLen)
		}
		DeinterleaveN(dsts, dst)
		deinterleaveNGo(drefs, ref, streamLen)
		for c := range dsts {
			exactEqualF32(t, "DeinterleaveN", dsts[c], drefs[c])
		}
	})
}
