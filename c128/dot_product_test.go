package c128

import (
	"math"
	"testing"
)

// dpSeq builds a deterministic complex128 slice mixing signs and magnitudes so
// the kernel's lane reduction is exercised across every unroll remainder.
func dpSeq(n, seed int) []complex128 {
	s := make([]complex128, n)
	for i := range s {
		re := float64((i*7+seed)%13) - 6
		im := float64((i*5+seed*3)%11) - 5
		s[i] = complex(re*0.25, im*0.25)
	}
	return s
}

// dpCloseF reports whether two float64 components agree within the tolerance used
// for f64 reductions, comparing NaN and Inf by class (the SIMD path accumulates
// in a different order than the scalar reference).
func dpCloseF(got, want float64) bool {
	if math.IsNaN(want) || math.IsNaN(got) {
		return math.IsNaN(want) == math.IsNaN(got)
	}
	if math.IsInf(want, 0) || math.IsInf(got, 0) {
		return math.IsInf(want, 1) == math.IsInf(got, 1) && math.IsInf(want, -1) == math.IsInf(got, -1)
	}
	return math.Abs(got-want) <= 1e-9*math.Abs(want)+1e-12
}

func dpClose(got, want complex128) bool {
	return dpCloseF(real(got), real(want)) && dpCloseF(imag(got), imag(want))
}

// TestDotProductParity validates the public dispatch (AVX on amd64, NEON on
// arm64, Go elsewhere) against the scalar references across every remainder.
func TestDotProductParity(t *testing.T) {
	for n := 0; n <= 20; n++ {
		a := dpSeq(n, 1)
		b := dpSeq(n, 4)
		if got, want := DotProduct(a, b), dotProductGo(a, b); !dpClose(got, want) {
			t.Errorf("DotProduct n=%d = %v, want %v", n, got, want)
		}
		if got, want := DotProductConj(a, b), dotProductConjGo(a, b); !dpClose(got, want) {
			t.Errorf("DotProductConj n=%d = %v, want %v", n, got, want)
		}
	}
}

// TestDotProductMismatchedLengths checks the min(len(a), len(b)) contract.
func TestDotProductMismatchedLengths(t *testing.T) {
	a := dpSeq(15, 1)
	b := dpSeq(9, 4)
	if got, want := DotProduct(a, b), dotProductGo(a, b); !dpClose(got, want) {
		t.Errorf("DotProduct mismatched = %v, want %v", got, want)
	}
	if got, want := DotProductConj(b, a), dotProductConjGo(b, a); !dpClose(got, want) {
		t.Errorf("DotProductConj mismatched = %v, want %v", got, want)
	}
}

// TestDotProductEmpty confirms empty input returns 0.
func TestDotProductEmpty(t *testing.T) {
	if got := DotProduct(nil, nil); got != 0 {
		t.Errorf("DotProduct(nil) = %v, want 0", got)
	}
	if got := DotProductConj([]complex128{}, []complex128{}); got != 0 {
		t.Errorf("DotProductConj(empty) = %v, want 0", got)
	}
}

// TestDotProductSpecialValues feeds NaN/Inf inputs and asserts the result
// propagates the same special class as the reference.
func TestDotProductSpecialValues(t *testing.T) {
	inf := math.Inf(1)
	nan := math.NaN()
	cases := [][2][]complex128{
		{{complex(1, 2), complex(nan, 1), complex(3, 4)}, {complex(5, 6), complex(7, 8), complex(9, 1)}},
		{{complex(inf, 0), complex(1, 1)}, {complex(2, 2), complex(3, 3)}},
		{{complex(1, inf), complex(1, 1), complex(2, 2)}, {complex(2, 2), complex(3, 3), complex(4, 4)}},
	}
	for ci, c := range cases {
		a, b := c[0], c[1]
		if got, want := DotProduct(a, b), dotProductGo(a, b); !dpClose(got, want) {
			t.Errorf("case %d DotProduct = %v, want %v", ci, got, want)
		}
		if got, want := DotProductConj(a, b), dotProductConjGo(a, b); !dpClose(got, want) {
			t.Errorf("case %d DotProductConj = %v, want %v", ci, got, want)
		}
	}
}

// TestDotProductAllocs confirms the public APIs stay allocation-free.
func TestDotProductAllocs(t *testing.T) {
	a := dpSeq(256, 1)
	b := dpSeq(256, 2)
	if allocs := testing.AllocsPerRun(100, func() { _ = DotProduct(a, b) }); allocs != 0 {
		t.Errorf("DotProduct allocs = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = DotProductConj(a, b) }); allocs != 0 {
		t.Errorf("DotProductConj allocs = %v, want 0", allocs)
	}
}
