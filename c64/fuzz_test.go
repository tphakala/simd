package c64

import (
	"encoding/binary"
	"math"
	"testing"
)

// Differential fuzz targets for c64, mirroring the f32/f64/i32 set. The
// dispatched public op must agree with the package's pure-Go reference for every
// length; the dominant SIMD bug class is tail/remainder handling at arbitrary
// lengths around the lane unrolls.
//
// Comparison policy: pure-movement ops (Add, Sub, Conj, FromReal) are a single
// rounding per lane and so are bit-exact (compared via raw float32 bits, with
// NaN==NaN and +0==-0). The arithmetic ops (Mul, MulConj, Scale, AbsSq, Abs)
// fuse a multiply-add in the SIMD path that the scalar reference rounds
// separately, so on bounded [-1,1] inputs they are compared with the same
// tolerance the parity tests use (complexClose/floatClose). The accumulating dot
// products use a length-scaled bound.

const fuzzEps32 = 1.1920928955078125e-07 // 2^-23, float32 machine epsilon (2u)

// c64sBits reinterprets raw bytes as complex64, 8 bytes (re||im float32 bits)
// per element. This organically produces NaN, Inf, denormals, and signed zeros.
func c64sBits(raw []byte) []complex64 {
	out := make([]complex64, len(raw)/8)
	for i := range out {
		re := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*8:]))
		im := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*8+4:]))
		out[i] = complex(re, im)
	}
	return out
}

// c64sUnit maps raw bytes to complex64 with both components in [-1, 1], keeping
// the arithmetic bounded so the tolerance comparison is meaningful (no NaN/Inf).
func c64sUnit(raw []byte) []complex64 {
	out := make([]complex64, len(raw)/8)
	for i := range out {
		re := float32(int32(binary.LittleEndian.Uint32(raw[i*8:]))) / 2147483648.0  // / 2^31
		im := float32(int32(binary.LittleEndian.Uint32(raw[i*8+4:]))) / 2147483648.0 // / 2^31
		out[i] = complex(re, im)
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

// eqF32Bits: any NaN equals any NaN, +0 equals -0, else bit-identical.
func eqF32Bits(g, w float32) bool {
	switch {
	case g != g && w != w:
		return true
	case g == 0 && w == 0:
		return true
	default:
		return math.Float32bits(g) == math.Float32bits(w)
	}
}

func eqC64(g, w complex64) bool {
	return eqF32Bits(real(g), real(w)) && eqF32Bits(imag(g), imag(w))
}

func exactEqualC64(t *testing.T, op string, got, want []complex64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", op, len(got), len(want))
	}
	for i := range got {
		if !eqC64(got[i], want[i]) {
			t.Fatalf("%s: lane %d got (%#08x,%#08x) want (%#08x,%#08x) (len=%d)", op, i,
				math.Float32bits(real(got[i])), math.Float32bits(imag(got[i])),
				math.Float32bits(real(want[i])), math.Float32bits(imag(want[i])), len(got))
		}
	}
}

// sumTol32 bounds the divergence between two float32 summation orders of nTerms
// terms whose magnitudes total scaleAbs (mirrors the f32 fuzz sumTol).
func sumTol32(nTerms int, scaleAbs float64) float64 {
	return (float64(nTerms)+8)*fuzzEps32*scaleAbs + 1e-5
}

func FuzzC64Movement(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c64sBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]complex64, h)
		want := make([]complex64, h)

		Add(got, a, b)
		addGo(want, a, b)
		exactEqualC64(t, "Add", got, want)

		Sub(got, a, b)
		subGo(want, a, b)
		exactEqualC64(t, "Sub", got, want)

		Conj(got, a)
		conjGo(want, a)
		exactEqualC64(t, "Conj", got, want)

		reals := make([]float32, h)
		for i := range reals {
			reals[i] = real(a[i])
		}
		FromReal(got, reals)
		fromRealGo(want, reals)
		exactEqualC64(t, "FromReal", got, want)
	})
}

func FuzzC64Arithmetic(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c64sUnit(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]complex64, h)
		want := make([]complex64, h)

		Mul(got, a, b)
		mulGo(want, a, b)
		for i := range got {
			if !complexClose(got[i], want[i]) {
				t.Fatalf("Mul lane %d got %v want %v (len=%d)", i, got[i], want[i], h)
			}
		}

		MulConj(got, a, b)
		mulConjGo(want, a, b)
		for i := range got {
			if !complexClose(got[i], want[i]) {
				t.Fatalf("MulConj lane %d got %v want %v (len=%d)", i, got[i], want[i], h)
			}
		}

		if h > 0 {
			s := a[0]
			Scale(got, a, s)
			scaleGo(want, a, s)
			for i := range got {
				if !complexClose(got[i], want[i]) {
					t.Fatalf("Scale lane %d got %v want %v (len=%d)", i, got[i], want[i], h)
				}
			}
		}

		asq := make([]float32, h)
		asqRef := make([]float32, h)
		AbsSq(asq, a)
		absSqGo(asqRef, a)
		for i := range asq {
			if !floatClose(asq[i], asqRef[i]) {
				t.Fatalf("AbsSq lane %d got %v want %v (len=%d)", i, asq[i], asqRef[i], h)
			}
		}

		ab := make([]float32, h)
		abRef := make([]float32, h)
		Abs(ab, a)
		absGo(abRef, a)
		for i := range ab {
			if !floatClose(ab[i], abRef[i]) {
				t.Fatalf("Abs lane %d got %v want %v (len=%d)", i, ab[i], abRef[i], h)
			}
		}
	})
}

func FuzzC64DotProduct(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c64sUnit(raw)
		h := len(v) / 2
		if h == 0 {
			return
		}
		a, b := v[:h], v[h:2*h]
		var scaleAbs float64
		for i := range a {
			ar, ai := float64(real(a[i])), float64(imag(a[i]))
			br, bi := float64(real(b[i])), float64(imag(b[i]))
			scaleAbs += math.Abs(ar*br) + math.Abs(ai*bi) + math.Abs(ar*bi) + math.Abs(ai*br)
		}
		tol := sumTol32(h, scaleAbs)

		got := DotProduct(a, b)
		want := dotProductGo(a, b)
		if d := math.Abs(float64(real(got)) - float64(real(want))); d > tol {
			t.Fatalf("DotProduct re got %v want %v |diff|=%g tol=%g (len=%d)", real(got), real(want), d, tol, h)
		}
		if d := math.Abs(float64(imag(got)) - float64(imag(want))); d > tol {
			t.Fatalf("DotProduct im got %v want %v |diff|=%g tol=%g (len=%d)", imag(got), imag(want), d, tol, h)
		}

		gotC := DotProductConj(a, b)
		wantC := dotProductConjGo(a, b)
		if d := math.Abs(float64(real(gotC)) - float64(real(wantC))); d > tol {
			t.Fatalf("DotProductConj re got %v want %v |diff|=%g tol=%g (len=%d)", real(gotC), real(wantC), d, tol, h)
		}
		if d := math.Abs(float64(imag(gotC)) - float64(imag(wantC))); d > tol {
			t.Fatalf("DotProductConj im got %v want %v |diff|=%g tol=%g (len=%d)", imag(gotC), imag(wantC), d, tol, h)
		}
	})
}
