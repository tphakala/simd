package c128

import (
	"encoding/binary"
	"math"
	"testing"
)

// Differential fuzz targets for c128, mirroring the c64 set. The dispatched
// public op must agree with the package's pure-Go reference for every length;
// the dominant SIMD bug class is tail/remainder handling at arbitrary lengths
// around the lane unrolls.
//
// Comparison policy: pure-movement ops (Add, Sub, Conj, FromReal) are bit-exact
// (raw float64 bits, NaN==NaN, +0==-0). The arithmetic ops (Mul, MulConj, Scale,
// AbsSq, Abs) fuse a multiply-add the scalar reference rounds separately, so on
// bounded [-1,1] inputs they are compared within the parity-test tolerance
// (complexClose / epsilon). The accumulating dot products use a length-scaled
// bound.

const fuzzEps64 = 2.220446049250313e-16 // 2^-52, float64 machine epsilon (2u)

func c128sBits(raw []byte) []complex128 {
	out := make([]complex128, len(raw)/16)
	for i := range out {
		re := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*16:]))
		im := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*16+8:]))
		out[i] = complex(re, im)
	}
	return out
}

func c128sUnit(raw []byte) []complex128 {
	out := make([]complex128, len(raw)/16)
	for i := range out {
		re := float64(int64(binary.LittleEndian.Uint64(raw[i*16:]))) / 9223372036854775808.0   // / 2^63
		im := float64(int64(binary.LittleEndian.Uint64(raw[i*16+8:]))) / 9223372036854775808.0 // / 2^63
		out[i] = complex(re, im)
	}
	return out
}

func addByteLenSeeds(f *testing.F) {
	f.Helper()
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257} {
		raw := make([]byte, n*16)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

// eqF64Bits: any NaN equals any NaN, +0 equals -0, else bit-identical.
func eqF64Bits(g, w float64) bool {
	switch {
	case math.IsNaN(g) && math.IsNaN(w):
		return true
	case g == 0 && w == 0:
		return true
	default:
		return math.Float64bits(g) == math.Float64bits(w)
	}
}

func eqC128(g, w complex128) bool {
	return eqF64Bits(real(g), real(w)) && eqF64Bits(imag(g), imag(w))
}

func exactEqualC128(t *testing.T, op string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", op, len(got), len(want))
	}
	for i := range got {
		if !eqC128(got[i], want[i]) {
			t.Fatalf("%s: lane %d got (%#016x,%#016x) want (%#016x,%#016x) (len=%d)", op, i,
				math.Float64bits(real(got[i])), math.Float64bits(imag(got[i])),
				math.Float64bits(real(want[i])), math.Float64bits(imag(want[i])), len(got))
		}
	}
}

func floatClose64(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func sumTol64(nTerms int, scaleAbs float64) float64 {
	return (float64(nTerms)+8)*fuzzEps64*scaleAbs + 1e-12
}

func FuzzC128Movement(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c128sBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]complex128, h)
		want := make([]complex128, h)

		Add(got, a, b)
		addGo(want, a, b)
		exactEqualC128(t, "Add", got, want)

		Sub(got, a, b)
		subGo(want, a, b)
		exactEqualC128(t, "Sub", got, want)

		Conj(got, a)
		conjGo(want, a)
		exactEqualC128(t, "Conj", got, want)

		reals := make([]float64, h)
		for i := range reals {
			reals[i] = real(a[i])
		}
		FromReal(got, reals)
		fromRealGo(want, reals)
		exactEqualC128(t, "FromReal", got, want)
	})
}

func FuzzC128Arithmetic(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c128sUnit(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]complex128, h)
		want := make([]complex128, h)

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

		asq := make([]float64, h)
		asqRef := make([]float64, h)
		AbsSq(asq, a)
		absSqGo(asqRef, a)
		for i := range asq {
			if !floatClose64(asq[i], asqRef[i]) {
				t.Fatalf("AbsSq lane %d got %v want %v (len=%d)", i, asq[i], asqRef[i], h)
			}
		}

		ab := make([]float64, h)
		abRef := make([]float64, h)
		Abs(ab, a)
		absGo(abRef, a)
		for i := range ab {
			if !floatClose64(ab[i], abRef[i]) {
				t.Fatalf("Abs lane %d got %v want %v (len=%d)", i, ab[i], abRef[i], h)
			}
		}
	})
}

func FuzzC128DotProduct(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := c128sUnit(raw)
		h := len(v) / 2
		if h == 0 {
			return
		}
		a, b := v[:h], v[h:2*h]
		var scaleAbs float64
		for i := range a {
			ar, ai := real(a[i]), imag(a[i])
			br, bi := real(b[i]), imag(b[i])
			scaleAbs += math.Abs(ar*br) + math.Abs(ai*bi) + math.Abs(ar*bi) + math.Abs(ai*br)
		}
		tol := sumTol64(h, scaleAbs)

		got := DotProduct(a, b)
		want := dotProductGo(a, b)
		if d := math.Abs(real(got) - real(want)); d > tol {
			t.Fatalf("DotProduct re got %v want %v |diff|=%g tol=%g (len=%d)", real(got), real(want), d, tol, h)
		}
		if d := math.Abs(imag(got) - imag(want)); d > tol {
			t.Fatalf("DotProduct im got %v want %v |diff|=%g tol=%g (len=%d)", imag(got), imag(want), d, tol, h)
		}

		gotC := DotProductConj(a, b)
		wantC := dotProductConjGo(a, b)
		if d := math.Abs(real(gotC) - real(wantC)); d > tol {
			t.Fatalf("DotProductConj re got %v want %v |diff|=%g tol=%g (len=%d)", real(gotC), real(wantC), d, tol, h)
		}
		if d := math.Abs(imag(gotC) - imag(wantC)); d > tol {
			t.Fatalf("DotProductConj im got %v want %v |diff|=%g tol=%g (len=%d)", imag(gotC), imag(wantC), d, tol, h)
		}
	})
}
