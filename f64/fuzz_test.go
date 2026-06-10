package f64

import (
	"encoding/binary"
	"math"
	"testing"
)

// Differential fuzz targets for f64, mirroring the f32 set. The dispatched public
// op must agree with the package's pure-Go reference for every length; the
// dominant SIMD bug class is tail/remainder handling at arbitrary lengths around
// the 4/8-lane unrolls. Comparison policy matches f32 (see f32/fuzz_test.go):
// movement/sign and exactly-rounded element ops are bit-exact (NaN==NaN, ±0
// equal), Min/Max/Clamp run on finite inputs, and accumulating ops are compared
// within an n-scaled epsilon * sum-of-magnitudes bound on bounded [-1,1] inputs.

// eps64 is the float64 machine epsilon (2^-52 = 2u).
const eps64 = 2.220446049250313e-16

func f64sBits(raw []byte) []float64 {
	out := make([]float64, len(raw)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

func f64sUnit(raw []byte) []float64 {
	out := make([]float64, len(raw)/8)
	for i := range out {
		u := int64(binary.LittleEndian.Uint64(raw[i*8:]))
		out[i] = float64(u) / 9223372036854775808.0 // / 2^63
	}
	return out
}

func f64sFinite(raw []byte) []float64 {
	out := make([]float64, len(raw)/8)
	for i := range out {
		u := binary.LittleEndian.Uint64(raw[i*8:])
		f := math.Float64frombits(u)
		if f != f || math.IsInf(f, 0) {
			f = float64(int64(u)) / 4294967296.0 // / 2^32
		}
		out[i] = f
	}
	return out
}

func addByteLenSeeds(f *testing.F) {
	f.Helper()
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257} {
		raw := make([]byte, n*8)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

// eqF64: any NaN equals any NaN, +0.0 equals -0.0, else bit-identical.
func eqF64(g, w float64) bool {
	switch {
	case math.IsNaN(g) && math.IsNaN(w):
		return true
	case g == 0 && w == 0:
		return true
	default:
		return math.Float64bits(g) == math.Float64bits(w)
	}
}

func exactEqualF64(t *testing.T, op string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", op, len(got), len(want))
	}
	for i := range got {
		if !eqF64(got[i], want[i]) {
			t.Fatalf("%s: lane %d got %x want %x (len=%d)", op, i, math.Float64bits(got[i]), math.Float64bits(want[i]), len(got))
		}
	}
}

// sumTol bounds the divergence between two summation orders of nTerms terms whose
// magnitudes total scaleAbs (see f32/fuzz_test.go sumTol).
func sumTol(nTerms int, scaleAbs float64) float64 {
	return (float64(nTerms)+8)*eps64*scaleAbs + 1e-12
}

func sumAbsF64(xs ...[]float64) float64 {
	var s float64
	for _, x := range xs {
		for _, v := range x {
			s += math.Abs(v)
		}
	}
	return s
}

func FuzzF64ElementwiseExact(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]float64, h)
		want := make([]float64, h)

		Add(got, a, b)
		addGo(want, a, b)
		exactEqualF64(t, "Add", got, want)
		Mul(got, a, b)
		mulGo(want, a, b)
		exactEqualF64(t, "Mul", got, want)
		Sub(got, a, b)
		subGo(want, a, b)
		exactEqualF64(t, "Sub", got, want)
		Abs(got, a)
		absGo(want, a)
		exactEqualF64(t, "Abs", got, want)
		Neg(got, a)
		negGo(want, a)
		exactEqualF64(t, "Neg", got, want)
	})
}

func FuzzF64MinMaxClamp(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		a := f64sFinite(raw)
		if len(a) == 0 {
			return
		}
		if got, want := Min(a), minGo(a); !eqF64(got, want) {
			t.Fatalf("Min got %v want %v (len=%d)", got, want, len(a))
		}
		if got, want := Max(a), maxGo(a); !eqF64(got, want) {
			t.Fatalf("Max got %v want %v (len=%d)", got, want, len(a))
		}
		lo, hi := a[0], a[len(a)-1]
		if lo > hi {
			lo, hi = hi, lo
		}
		got := make([]float64, len(a))
		want := make([]float64, len(a))
		Clamp(got, a, lo, hi)
		clampGo(want, a, lo, hi)
		exactEqualF64(t, "Clamp", got, want)
	})
}

func FuzzF64Reductions(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sUnit(raw)
		if d := math.Abs(Sum(v) - sumGo(v)); d > sumTol(len(v), sumAbsF64(v)) {
			t.Fatalf("Sum got %v want %v |diff|=%g", Sum(v), sumGo(v), d)
		}
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		var dotScale float64
		for i := range a {
			dotScale += math.Abs(a[i] * b[i])
		}
		if d := math.Abs(DotProduct(a, b) - dotProductGo(a, b)); d > sumTol(h, dotScale) {
			t.Fatalf("DotProduct got %v want %v |diff|=%g", DotProduct(a, b), dotProductGo(a, b), d)
		}
	})
}

func FuzzF64FMA(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sUnit(raw)
		third := len(v) / 3
		if third == 0 {
			return
		}
		a, b, c := v[:third], v[third:2*third], v[2*third:3*third]
		got := make([]float64, third)
		want := make([]float64, third)
		FMA(got, a, b, c)
		fmaGo(want, a, b, c)
		for i := range got {
			scale := math.Abs(a[i]*b[i]) + math.Abs(c[i])
			tol := 4*eps64*scale + 1e-12
			if d := math.Abs(got[i] - want[i]); d > tol {
				t.Fatalf("FMA lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func FuzzF64CumulativeSum(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		a := f64sUnit(raw)
		got := make([]float64, len(a))
		want := make([]float64, len(a))
		CumulativeSum(got, a)
		cumulativeSum64Go(want, a)
		var prefixAbs float64
		for i := range a {
			prefixAbs += math.Abs(a[i])
			tol := sumTol(i+1, prefixAbs)
			if d := math.Abs(got[i] - want[i]); d > tol {
				t.Fatalf("CumulativeSum lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func FuzzF64Convolve(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sUnit(raw)
		if len(v) < 4 {
			return
		}
		kLen := 1 + int(raw[0])%min(len(v)-1, 16)
		kernel := v[:kLen]
		signal := v[kLen:]
		if len(signal) < kLen {
			return
		}
		validLen := len(signal) - kLen + 1
		dst := make([]float64, validLen)
		ref := make([]float64, validLen)
		ConvolveValid(dst, signal, kernel)
		convolveValid64Go(ref, signal, kernel)
		convTol := sumTol(kLen, float64(kLen))
		for i := range dst {
			if d := math.Abs(dst[i] - ref[i]); d > convTol {
				t.Fatalf("ConvolveValid lane %d got %v want %v |diff|=%g tol=%g", i, dst[i], ref[i], d, convTol)
			}
		}

		factor := 1 + int(raw[1])%4
		phase := int(raw[2]) % factor
		span := len(signal) - kLen - phase
		if span < 0 {
			return
		}
		n := span/factor + 1
		ddst := make([]float64, n)
		dref := make([]float64, n)
		ConvolveDecimate(ddst, signal, kernel, factor, phase)
		convolveDecimate64Go(dref, signal, kernel, factor, phase)
		for i := range ddst {
			if d := math.Abs(ddst[i] - dref[i]); d > convTol {
				t.Fatalf("ConvolveDecimate lane %d (factor=%d phase=%d) got %v want %v |diff|=%g tol=%g", i, factor, phase, ddst[i], dref[i], d, convTol)
			}
		}
	})
}

func FuzzF64InterleaveN(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		if len(v) < 2 {
			return
		}
		nc := 1 + int(raw[0])%10
		streamLen := len(v) / nc
		if streamLen == 0 {
			return
		}
		srcs := make([][]float64, nc)
		for c := range srcs {
			srcs[c] = v[c*streamLen : (c+1)*streamLen]
		}
		dst := make([]float64, nc*streamLen)
		ref := make([]float64, nc*streamLen)
		InterleaveN(dst, srcs)
		interleaveNGo(ref, srcs, streamLen)
		exactEqualF64(t, "InterleaveN", dst, ref)

		dsts := make([][]float64, nc)
		drefs := make([][]float64, nc)
		for c := range dsts {
			dsts[c] = make([]float64, streamLen)
			drefs[c] = make([]float64, streamLen)
		}
		DeinterleaveN(dsts, dst)
		deinterleaveNGo(drefs, ref, streamLen)
		for c := range dsts {
			exactEqualF64(t, "DeinterleaveN", dsts[c], drefs[c])
		}
	})
}

// FuzzF64Log differentially fuzzes the dispatched Log/Log2/Log10 against the
// pure-Go references over arbitrary bit patterns, including NaN, infinities,
// negatives, zeros, and subnormals. Specials compare by class, finite values
// within the documented kernel tolerance.
func FuzzF64Log(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		if len(v) == 0 {
			return
		}
		got := make([]float64, len(v))
		want := make([]float64, len(v))

		check := func(op string) {
			for i := range got {
				g, w := got[i], want[i]
				switch {
				case math.IsNaN(w):
					if !math.IsNaN(g) {
						t.Fatalf("%s[%d](%v): got %v want NaN", op, i, v[i], g)
					}
				case math.IsInf(w, 0) || w == 0:
					if g != w {
						t.Fatalf("%s[%d](%v): got %v want %v", op, i, v[i], g, w)
					}
				default:
					diff := math.Abs(g - w)
					if math.Abs(w) > 1e-12 {
						diff /= math.Abs(w)
					}
					if diff > logRelTol64 {
						t.Fatalf("%s[%d](%g): got %v want %v (err %g)", op, i, v[i], g, w, diff)
					}
				}
			}
		}
		Log(got, v)
		logGo(want, v)
		check("Log")
		Log2(got, v)
		log2Go(want, v)
		check("Log2")
		Log10(got, v)
		log10Go(want, v)
		check("Log10")
	})
}

// FuzzF64Pow differentially fuzzes Pow and PowElem against math.Pow for
// positive finite bases (the SIMD precondition; other inputs dispatch to the
// scalar path, which is exercised too via the raw exponent). Lanes whose
// |p*ln(x)| lands near the overflow/underflow thresholds (>700) or whose
// true result is subnormal are skipped: the kernel's relative error can flip
// the result class there, and subnormal results lose precision gradually
// (see powAVX).
func FuzzF64Pow(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		if len(v) < 3 {
			return
		}
		p := v[0]
		rest := v[1:]
		h := len(rest) / 2
		base, exps := rest[:h], rest[h:2*h]
		for i, x := range base {
			x = math.Abs(x)
			if !(x > 0 && x <= math.MaxFloat64) {
				x = 1.5 + float64(i)
			}
			base[i] = x
		}

		checkLane := func(op string, i int, x, pw, g float64) {
			want := math.Pow(x, pw)
			if math.IsNaN(want) {
				if !math.IsNaN(g) {
					t.Fatalf("%s[%d](%g, %g): got %v want NaN", op, i, x, pw, g)
				}
				return
			}
			if y := pw * refLn64(x); math.Abs(y) > 700 || math.IsNaN(y) {
				// Near the exact overflow/underflow thresholds the kernel's
				// ~3e-6 error in p*ln(x) can flip the +Inf/0 class.
				return
			}
			if want != 0 && math.Abs(want) < 4e-308 {
				return // subnormal results may flush to 0 in the exp core
			}
			diff := math.Abs(g - want)
			if math.Abs(want) > 1e-12 {
				diff /= math.Abs(want)
			}
			if diff > powRelTol64 {
				t.Fatalf("%s[%d](%g, %g): got %v want %v (err %g)", op, i, x, pw, g, want, diff)
			}
		}

		got := make([]float64, h)
		Pow(got, base, p)
		for i := range got {
			checkLane("Pow", i, base[i], p, got[i])
		}
		PowElem(got, base, exps)
		for i := range got {
			checkLane("PowElem", i, base[i], exps[i], got[i])
		}
	})
}
