package f32

import (
	"math"
	"testing"
)

// affineRef is the scalar reference for Affine: dst[i] = alpha*a[i] + beta with a
// separate multiply and add (two roundings), matching the pure-Go fallback and the
// split-rounding kernels.
func affineRef(dst, a []float32, alpha, beta float32) {
	for i := range dst {
		// float32() rounds the product before the add so the arm64 backend cannot
		// contract this into a single FMADD (see affineGo).
		dst[i] = float32(a[i]*alpha) + beta
	}
}

// TestAffine checks Affine against two bit-exact references on every dispatch
// tier. Affine is split-rounding (multiply then add, two roundings), so it must
// be bit-identical to (1) Scale then AddScalar, the pair it fuses, and (2) the
// scalar affineRef. No tolerance: any bit difference is a bug.
func TestAffine(t *testing.T) {
	params := []struct{ alpha, beta float32 }{
		{10, -1.5}, {0.5, 2}, {-2, 0.25}, {1, 0}, {0, 3.5},
	}
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 255, 1000} {
			src := make([]float32, n)
			for i := range src {
				src[i] = genF32(i)
			}
			for _, p := range params {
				got := make([]float32, n)
				Affine(got, src, p.alpha, p.beta)

				// (1) bit-identical to Scale then AddScalar (the pair it fuses).
				wantSA := append([]float32(nil), src...)
				Scale(wantSA, wantSA, p.alpha)
				AddScalar(wantSA, wantSA, p.beta)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(wantSA[i]) {
						t.Fatalf("Affine vs Scale+AddScalar n=%d a=%v b=%v lane %d: got %v want %v",
							n, p.alpha, p.beta, i, got[i], wantSA[i])
					}
				}

				// (2) bit-identical to the scalar reference.
				wantRef := make([]float32, n)
				affineRef(wantRef, src, p.alpha, p.beta)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(wantRef[i]) {
						t.Fatalf("Affine vs affineRef n=%d a=%v b=%v lane %d: got %v want %v",
							n, p.alpha, p.beta, i, got[i], wantRef[i])
					}
				}
			}
		}
	})
}

// TestAffineInPlace checks the documented in-place overlay (dst may alias src).
func TestAffineInPlace(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{1, 4, 7, 16, 31, 128} {
			src := make([]float32, n)
			for i := range src {
				src[i] = genF32(i)
			}
			want := make([]float32, n)
			affineRef(want, src, 3, -0.5)
			got := append([]float32(nil), src...)
			Affine(got, got, 3, -0.5)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("Affine in-place n=%d lane %d: got %v want %v", n, i, got[i], want[i])
				}
			}
		}
	})
}

// TestAffineEdgeCases covers empty and mismatched-length inputs.
func TestAffineEdgeCases(t *testing.T) {
	Affine(nil, nil, 2, 1)
	Affine([]float32{}, []float32{}, 2, 1)

	// Mismatched lengths clamp to the shortest; trailing dst is untouched.
	dst := []float32{10, 20, 30, 40}
	src := []float32{2, 3}
	Affine(dst, src, 10, 1) // n = min(4,2) = 2
	want := []float32{2*10 + 1, 3*10 + 1, 30, 40}
	assertFloat32SlicesEqual(t, want, dst, "Affine mismatched lengths")
}

// TestAffineAllocFree asserts the separate-destination path allocates nothing on
// every dispatch tier.
func TestAffineAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float32, 1000)
		src := make([]float32, 1000)
		for i := range src {
			src[i] = genF32(i)
		}
		if got := testing.AllocsPerRun(10, func() { Affine(dst, src, 1.5, 0.25) }); got != 0 {
			t.Errorf("Affine allocated %v times per run, want 0", got)
		}
	})
}

// log10FlooredRef is the scalar reference: log10 of the input floored at floor (an
// exact lower clamp), matching Clamp(dst,src,floor,+Inf) then Log10.
func log10FlooredRef(src []float32, floor float32) []float32 {
	out := make([]float32, len(src))
	for i, v := range src {
		if v < floor {
			v = floor
		}
		out[i] = float32(math.Log10(float64(v)))
	}
	return out
}

// TestLog10Floored checks Log10Floored on every dispatch tier. It is defined as
// Clamp(dst,src,floor,+Inf) then Log10, so (1) it must be bit-identical to that
// explicit composition, and (2) it must match the scalar log10(max(src,floor))
// reference within the Log10 kernel's relative tolerance. Inputs include zeros and
// negatives, which the floor maps to a finite log10(floor).
func TestLog10Floored(t *testing.T) {
	const floor = 1e-4
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{0, 1, 3, 4, 7, 8, 15, 16, 31, 63, 255, 1023} {
			src := make([]float32, n)
			for i := range src {
				switch i % 5 {
				case 0:
					src[i] = 0 // exercise the floor: log10(0) would be -Inf
				case 1:
					src[i] = -genF32Pos(i) // negative: floored to a finite result
				default:
					src[i] = genF32Pos(i)
				}
			}

			got := make([]float32, n)
			Log10Floored(got, src, floor)

			// (1) bit-identical to the explicit Clamp-then-Log10 composition.
			comp := make([]float32, n)
			Clamp(comp, src, floor, float32(math.Inf(1)))
			Log10(comp, comp)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(comp[i]) {
					t.Fatalf("Log10Floored vs Clamp+Log10 n=%d lane %d: got %v want %v", n, i, got[i], comp[i])
				}
			}

			// (2) parity with the scalar reference within tolerance, every result finite.
			ref := log10FlooredRef(src, floor)
			for i := range got {
				if math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(got[i])) {
					t.Fatalf("Log10Floored n=%d lane %d not finite: got %v (src %v)", n, i, got[i], src[i])
				}
				if re := relErrF32(got[i], ref[i]); re > logRelTol32 {
					t.Fatalf("Log10Floored n=%d lane %d: got %v want %v relerr %g > %g",
						n, i, got[i], ref[i], re, logRelTol32)
				}
			}
		}
	})
}

// TestLog10FlooredInPlace checks the documented in-place overlay (dst may alias src).
func TestLog10FlooredInPlace(t *testing.T) {
	const floor = 1e-3
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{1, 4, 7, 16, 63} {
			src := make([]float32, n)
			for i := range src {
				if i%3 == 0 {
					src[i] = 0
				} else {
					src[i] = genF32Pos(i)
				}
			}
			want := make([]float32, n)
			Log10Floored(want, src, floor)
			got := append([]float32(nil), src...)
			Log10Floored(got, got, floor)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("Log10Floored in-place n=%d lane %d: got %v want %v", n, i, got[i], want[i])
				}
			}
		}
	})
}

// TestLog10FlooredEdgeCases covers empty and mismatched-length inputs and the
// finite-result guarantee for a zero input.
func TestLog10FlooredEdgeCases(t *testing.T) {
	Log10Floored(nil, nil, 1e-3)
	Log10Floored([]float32{}, []float32{}, 1e-3)

	dst := []float32{5, 6, 7, 8}
	src := []float32{0, 100}     // 0 -> floor, 100 -> log10(100) = 2
	Log10Floored(dst, src, 1e-2) // n = min(4,2) = 2
	if dst[2] != 7 || dst[3] != 8 {
		t.Fatalf("Log10Floored overran: tail = %v %v, want 7 8", dst[2], dst[3])
	}
	if math.IsInf(float64(dst[0]), 0) || math.IsNaN(float64(dst[0])) {
		t.Fatalf("Log10Floored(0) not finite: %v", dst[0])
	}
	if d := math.Abs(float64(dst[1]) - 2); d > 1e-5 {
		t.Fatalf("Log10Floored(100) = %v, want ~2", dst[1])
	}
}

// TestLog10FlooredAllocFree asserts zero allocations on every tier.
func TestLog10FlooredAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float32, 1000)
		src := make([]float32, 1000)
		for i := range src {
			src[i] = genF32Pos(i)
		}
		if got := testing.AllocsPerRun(10, func() { Log10Floored(dst, src, 1e-4) }); got != 0 {
			t.Errorf("Log10Floored allocated %v times per run, want 0", got)
		}
	})
}

func BenchmarkAffine_1000(b *testing.B) {
	src := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range src {
		src[i] = genF32(i)
	}
	b.SetBytes(1000 * 4 * 2) // read src, write dst
	for b.Loop() {
		Affine(dst, src, 10, -1.5)
	}
}

func BenchmarkLog10Floored_1000(b *testing.B) {
	src := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range src {
		src[i] = genF32Pos(i)
	}
	b.SetBytes(1000 * 4 * 2)
	for b.Loop() {
		Log10Floored(dst, src, 1e-4)
	}
}

// FuzzF32Affine differentially fuzzes Affine against Scale then AddScalar over
// arbitrary bit patterns (NaN, Inf, subnormals included). Both perform the same
// two roundings, so the result must be bit-identical on every dispatch tier; a
// kernel that fused into an FMADD would diverge here.
func FuzzF32Affine(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sBits(raw)
		if len(v) < 3 {
			return
		}
		alpha, beta := v[0], v[1]
		src := v[2:]
		got := make([]float32, len(src))
		Affine(got, src, alpha, beta)
		want := append([]float32(nil), src...)
		Scale(want, want, alpha)
		AddScalar(want, want, beta)
		exactEqualF32(t, "Affine", got, want)
	})
}

// FuzzF32Log10Floored fuzzes Log10Floored against its defining composition (Clamp
// to the floor, then Log10) for bit-identity on every tier, and checks the
// finite-result guarantee: a positive finite floor maps every finite input to a
// finite log10.
func FuzzF32Log10Floored(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sBits(raw)
		if len(v) < 2 {
			return
		}
		floor := v[0]
		src := v[1:]
		got := make([]float32, len(src))
		Log10Floored(got, src, floor)
		want := make([]float32, len(src))
		Clamp(want, src, floor, float32(math.Inf(1)))
		Log10(want, want)
		exactEqualF32(t, "Log10Floored", got, want)
		// Finite-result guarantee: for a positive finite floor, every finite input
		// yields a finite result (a zero/negative/tiny input is lifted to floor).
		if floor > 0 && !math.IsInf(float64(floor), 0) {
			for i, g := range got {
				s := float64(src[i])
				if math.IsNaN(s) || math.IsInf(s, 0) {
					continue // out-of-domain input; the guarantee does not apply
				}
				if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
					t.Fatalf("Log10Floored(finite src=%v, floor=%v) not finite: %v", src[i], floor, g)
				}
			}
		}
	})
}
