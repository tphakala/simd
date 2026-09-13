package f32

import (
	"math"
	"testing"
)

// mulAddRef is the scalar reference for MulAdd: dst[i] += a[i]*b[i] with a
// separate multiply and add (two roundings), matching the pure-Go fallback.
func mulAddRef(dst, a, b []float32) {
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
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 1000} {
			a := make([]float32, n)
			b := make([]float32, n)
			dst0 := make([]float32, n)
			for i := range a {
				a[i] = genF32(i)
				b[i] = genF32(i + 37)
				dst0[i] = genF32(i + 101)
			}

			// (1) Definitional invariant: MulAdd == FMA(dst, a, b, dst), bit for bit
			// (same kernel, same args), so assert bit-exactness, not tolerance.
			got := append([]float32(nil), dst0...)
			MulAdd(got, a, b)
			wantFMA := append([]float32(nil), dst0...)
			FMA(wantFMA, a, b, wantFMA)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(wantFMA[i]) {
					t.Fatalf("MulAdd vs FMA(dst,a,b,dst) n=%d lane %d: got %v want %v", n, i, got[i], wantFMA[i])
				}
			}

			// (2) Parity vs the scalar reference, within FMA's tolerance.
			wantRef := append([]float32(nil), dst0...)
			mulAddRef(wantRef, a, b)
			for i := range got {
				scale := math.Abs(float64(a[i])*float64(b[i])) + math.Abs(float64(dst0[i]))
				tol := 4*eps32*scale + 1e-6
				if d := math.Abs(float64(got[i]) - float64(wantRef[i])); d > tol {
					t.Fatalf("MulAdd n=%d lane %d got %v want %v |diff|=%g tol=%g", n, i, got[i], wantRef[i], d, tol)
				}
			}
		}
	})
}

// TestMulAddEdgeCases covers empty and mismatched-length inputs.
func TestMulAddEdgeCases(t *testing.T) {
	// Empty is a no-op and must not panic.
	MulAdd(nil, nil, nil)
	MulAdd([]float32{}, []float32{}, []float32{})

	// Mismatched lengths clamp to the shortest; trailing dst is untouched.
	dst := []float32{10, 20, 30, 40}
	a := []float32{2, 3}
	b := []float32{5, 5, 5}
	MulAdd(dst, a, b) // n = min(4,2,3) = 2
	want := []float32{10 + 2*5, 20 + 3*5, 30, 40}
	assertFloat32SlicesEqual(t, want, dst, "MulAdd mismatched lengths")
}

// TestMulAddAllocFree asserts the separate-destination path allocates nothing on
// every dispatch tier. The in-place overlay path is covered by the aliasing
// zero-alloc sweep.
func TestMulAddAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float32, 1000)
		a := make([]float32, 1000)
		b := make([]float32, 1000)
		for i := range a {
			a[i] = genF32(i)
			b[i] = genF32(i + 7)
		}
		if got := testing.AllocsPerRun(10, func() { MulAdd(dst, a, b) }); got != 0 {
			t.Errorf("MulAdd allocated %v times per run, want 0", got)
		}
	})
}

// FuzzF32MulAdd fuzzes MulAdd against the scalar reference within tolerance.
func FuzzF32MulAdd(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sUnit(raw)
		third := len(v) / 3
		if third == 0 {
			return
		}
		a, b, dst0 := v[:third], v[third:2*third], v[2*third:3*third]
		got := append([]float32(nil), dst0...)
		MulAdd(got, a, b)
		want := append([]float32(nil), dst0...)
		mulAddRef(want, a, b)
		// Fused vs unfused differ by at most ~1 ulp of |a*b|+|dst| per lane.
		for i := range got {
			scale := math.Abs(float64(a[i])*float64(b[i])) + math.Abs(float64(dst0[i]))
			tol := 4*eps32*scale + 1e-6
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > tol {
				t.Fatalf("MulAdd lane %d got %v want %v |diff|=%g tol=%g", i, got[i], want[i], d, tol)
			}
		}
	})
}

func BenchmarkMulAdd_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 1000)

	// Four streams of traffic per element: read a, read c, read dst, write dst.
	b.SetBytes(1000 * 4 * 4)

	for b.Loop() {
		MulAdd(dst, a, c)
	}
}
