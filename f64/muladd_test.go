package f64

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// mulAddRef is a scalar tolerance oracle for MulAdd: dst[i] += a[i]*b[i] with a
// separate multiply and add (two roundings). Note the f64 pure-Go fma path
// itself fuses via math.FMA, so this reference is a tolerance oracle, not a
// bit-exact match to any tier.
func mulAddRef(dst, a, b []float64) {
	for i := range dst {
		dst[i] += a[i] * b[i]
	}
}

// TestMulAdd checks MulAdd against two references across every dispatch tier:
// (1) it must be bit-identical to FMA(dst, a, b, dst), since MulAdd is defined
// as exactly that call; (2) it must match the scalar dst += a*b within the
// fused-vs-split tolerance the FMA kernel already permits.
func TestMulAdd(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64, 127, 1000} {
			a, b, dst0 := makeTestData64(n)

			// (1) Definitional invariant: MulAdd == FMA(dst, a, b, dst) exactly.
			got := append([]float64(nil), dst0...)
			MulAdd(got, a, b)
			wantFMA := append([]float64(nil), dst0...)
			FMA(wantFMA, a, b, wantFMA)
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(wantFMA[i]) {
					t.Fatalf("MulAdd vs FMA(dst,a,b,dst) n=%d lane %d: got %v want %v", n, i, got[i], wantFMA[i])
				}
			}

			// (2) Parity vs the scalar reference, within tolerance.
			wantRef := append([]float64(nil), dst0...)
			mulAddRef(wantRef, a, b)
			assert.InDeltaSlice(t, wantRef, got, refTolerance64, "MulAdd n=%d", n)
		}
	})
}

// TestMulAddEdgeCases covers empty and mismatched-length inputs.
func TestMulAddEdgeCases(t *testing.T) {
	// Empty is a no-op and must not panic.
	MulAdd(nil, nil, nil)
	MulAdd([]float64{}, []float64{}, []float64{})

	// Mismatched lengths clamp to the shortest; trailing dst is untouched.
	dst := []float64{10, 20, 30, 40}
	a := []float64{2, 3}
	b := []float64{5, 5, 5}
	MulAdd(dst, a, b) // n = min(4,2,3) = 2
	want := []float64{10 + 2*5, 20 + 3*5, 30, 40}
	assert.Equal(t, want, dst, "MulAdd mismatched lengths")
}

// TestMulAddAllocFree asserts the separate-destination path allocates nothing on
// every dispatch tier. The in-place overlay path is covered by the aliasing
// zero-alloc sweep.
func TestMulAddAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float64, 1000)
		a := make([]float64, 1000)
		b := make([]float64, 1000)
		for i := range a {
			a[i] = aliasGenF64(i)
			b[i] = aliasGenF64(i + 7)
		}
		if got := testing.AllocsPerRun(10, func() { MulAdd(dst, a, b) }); got != 0 {
			t.Errorf("MulAdd allocated %v times per run, want 0", got)
		}
	})
}

// FuzzF64MulAdd fuzzes MulAdd against the scalar reference within tolerance.
func FuzzF64MulAdd(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sUnit(raw)
		third := len(v) / 3
		if third == 0 {
			return
		}
		a, b, dst0 := v[:third], v[third:2*third], v[2*third:3*third]
		got := append([]float64(nil), dst0...)
		MulAdd(got, a, b)
		want := append([]float64(nil), dst0...)
		mulAddRef(want, a, b)
		// Fused vs unfused differ by at most ~1 ulp of |a*b|+|dst| per lane.
		for i := range got {
			scale := math.Abs(a[i]*b[i]) + math.Abs(dst0[i])
			tol := 4*eps64*scale + 1e-12
			if d := math.Abs(got[i] - want[i]); d > tol {
				t.Fatalf("MulAdd lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func BenchmarkMulAdd_1000(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	dst := make([]float64, 1000)

	// Four streams of traffic per element: read a, read c, read dst, write dst.
	b.SetBytes(1000 * 8 * 4)

	for b.Loop() {
		MulAdd(dst, a, c)
	}
}
