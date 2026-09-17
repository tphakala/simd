package f64

import (
	"math"
	"testing"
)

// affineRef is the scalar reference for Affine: dst[i] = alpha*a[i] + beta with a
// separate multiply and add (two roundings), matching the pure-Go fallback and the
// split-rounding kernels.
func affineRef(dst, a []float64, alpha, beta float64) {
	// Two passes force the product to round to float64 before beta is added, so an
	// FMA-capable backend cannot contract this into a single FMADD (a same-width
	// float64() conversion is not a guaranteed fusion barrier; see affineGo).
	for i := range dst {
		dst[i] = a[i] * alpha
	}
	for i := range dst {
		dst[i] += beta
	}
}

// TestAffine checks Affine against two bit-exact references on every dispatch
// tier. Affine is split-rounding (multiply then add, two roundings), so it must
// be bit-identical to (1) Scale then AddScalar, the pair it fuses, and (2) the
// scalar affineRef. No tolerance: any bit difference is a bug.
func TestAffine(t *testing.T) {
	params := []struct{ alpha, beta float64 }{
		{10, -1.5}, {0.5, 2}, {-2, 0.25}, {1, 0}, {0, 3.5},
	}
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 255, 1000} {
			src := make([]float64, n)
			for i := range src {
				src[i] = aliasGenF64(i)
			}
			for _, p := range params {
				got := make([]float64, n)
				Affine(got, src, p.alpha, p.beta)

				// (1) bit-identical to Scale then AddScalar (the pair it fuses).
				wantSA := append([]float64(nil), src...)
				Scale(wantSA, wantSA, p.alpha)
				AddScalar(wantSA, wantSA, p.beta)
				for i := range got {
					if math.Float64bits(got[i]) != math.Float64bits(wantSA[i]) {
						t.Fatalf("Affine vs Scale+AddScalar n=%d a=%v b=%v lane %d: got %v want %v",
							n, p.alpha, p.beta, i, got[i], wantSA[i])
					}
				}

				// (2) bit-identical to the scalar reference.
				wantRef := make([]float64, n)
				affineRef(wantRef, src, p.alpha, p.beta)
				for i := range got {
					if math.Float64bits(got[i]) != math.Float64bits(wantRef[i]) {
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
		for _, n := range []int{1, 2, 4, 7, 16, 31, 128} {
			src := make([]float64, n)
			for i := range src {
				src[i] = aliasGenF64(i)
			}
			want := make([]float64, n)
			affineRef(want, src, 3, -0.5)
			got := append([]float64(nil), src...)
			Affine(got, got, 3, -0.5)
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("Affine in-place n=%d lane %d: got %v want %v", n, i, got[i], want[i])
				}
			}
		}
	})
}

// TestAffineEdgeCases covers empty and mismatched-length inputs.
func TestAffineEdgeCases(t *testing.T) {
	Affine(nil, nil, 2, 1)
	Affine([]float64{}, []float64{}, 2, 1)

	// Mismatched lengths clamp to the shortest; trailing dst is untouched.
	dst := []float64{10, 20, 30, 40}
	src := []float64{2, 3}
	Affine(dst, src, 10, 1) // n = min(4,2) = 2
	want := []float64{2*10 + 1, 3*10 + 1, 30, 40}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("Affine mismatched lengths lane %d: got %v want %v", i, dst[i], want[i])
		}
	}
}

// TestAffineAllocFree asserts the separate-destination path allocates nothing on
// every dispatch tier.
func TestAffineAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float64, 1000)
		src := make([]float64, 1000)
		for i := range src {
			src[i] = aliasGenF64(i)
		}
		if got := testing.AllocsPerRun(10, func() { Affine(dst, src, 1.5, 0.25) }); got != 0 {
			t.Errorf("Affine allocated %v times per run, want 0", got)
		}
	})
}

// log10FlooredRef is the scalar reference: log10 of the input floored at floor (an
// exact lower clamp), matching Clamp(dst,src,floor,+Inf) then Log10.
func log10FlooredRef(src []float64, floor float64) []float64 {
	out := make([]float64, len(src))
	for i, v := range src {
		if v < floor {
			v = floor
		}
		out[i] = math.Log10(v)
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
			src := make([]float64, n)
			for i := range src {
				switch i % 5 {
				case 0:
					src[i] = 0 // exercise the floor: log10(0) would be -Inf
				case 1:
					src[i] = -aliasGenF64Pos(i) // negative: floored to a finite result
				default:
					src[i] = aliasGenF64Pos(i)
				}
			}

			got := make([]float64, n)
			Log10Floored(got, src, floor)

			// (1) bit-identical to the explicit Clamp-then-Log10 composition.
			comp := make([]float64, n)
			Clamp(comp, src, floor, math.Inf(1))
			Log10(comp, comp)
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(comp[i]) {
					t.Fatalf("Log10Floored vs Clamp+Log10 n=%d lane %d: got %v want %v", n, i, got[i], comp[i])
				}
			}

			// (2) parity with the scalar reference within tolerance, every result finite.
			ref := log10FlooredRef(src, floor)
			for i := range got {
				if math.IsInf(got[i], 0) || math.IsNaN(got[i]) {
					t.Fatalf("Log10Floored n=%d lane %d not finite: got %v (src %v)", n, i, got[i], src[i])
				}
				if re := relErrF64(got[i], ref[i]); re > logRelTol64 {
					t.Fatalf("Log10Floored n=%d lane %d: got %v want %v relerr %g > %g",
						n, i, got[i], ref[i], re, logRelTol64)
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
		for _, n := range []int{1, 2, 4, 7, 16, 63} {
			src := make([]float64, n)
			for i := range src {
				if i%3 == 0 {
					src[i] = 0
				} else {
					src[i] = aliasGenF64Pos(i)
				}
			}
			want := make([]float64, n)
			Log10Floored(want, src, floor)
			got := append([]float64(nil), src...)
			Log10Floored(got, got, floor)
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
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
	Log10Floored([]float64{}, []float64{}, 1e-3)

	dst := []float64{5, 6, 7, 8}
	src := []float64{0, 100}     // 0 -> floor, 100 -> log10(100) = 2
	Log10Floored(dst, src, 1e-2) // n = min(4,2) = 2
	if dst[2] != 7 || dst[3] != 8 {
		t.Fatalf("Log10Floored overran: tail = %v %v, want 7 8", dst[2], dst[3])
	}
	if math.IsInf(dst[0], 0) || math.IsNaN(dst[0]) {
		t.Fatalf("Log10Floored(0) not finite: %v", dst[0])
	}
	if d := math.Abs(dst[1] - 2); d > 1e-12 {
		t.Fatalf("Log10Floored(100) = %v, want ~2", dst[1])
	}
}

// TestLog10FlooredAllocFree asserts zero allocations on every tier.
func TestLog10FlooredAllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float64, 1000)
		src := make([]float64, 1000)
		for i := range src {
			src[i] = aliasGenF64Pos(i)
		}
		if got := testing.AllocsPerRun(10, func() { Log10Floored(dst, src, 1e-4) }); got != 0 {
			t.Errorf("Log10Floored allocated %v times per run, want 0", got)
		}
	})
}

func BenchmarkAffine_1000(b *testing.B) {
	src := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range src {
		src[i] = aliasGenF64(i)
	}
	b.SetBytes(1000 * 8 * 2) // read src, write dst
	for b.Loop() {
		Affine(dst, src, 10, -1.5)
	}
}

func BenchmarkLog10Floored_1000(b *testing.B) {
	src := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range src {
		src[i] = aliasGenF64Pos(i)
	}
	b.SetBytes(1000 * 8 * 2)
	for b.Loop() {
		Log10Floored(dst, src, 1e-4)
	}
}

// FuzzF64Affine differentially fuzzes Affine against Scale then AddScalar over
// arbitrary bit patterns (NaN, Inf, subnormals included). Both perform the same
// two roundings, so the result must be bit-identical on every dispatch tier; a
// kernel that fused into an FMADD would diverge here.
func FuzzF64Affine(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		if len(v) < 3 {
			return
		}
		alpha, beta := v[0], v[1]
		src := v[2:]
		got := make([]float64, len(src))
		Affine(got, src, alpha, beta)
		want := append([]float64(nil), src...)
		Scale(want, want, alpha)
		AddScalar(want, want, beta)
		exactEqualF64(t, "Affine", got, want)
	})
}

// FuzzF64Log10Floored fuzzes Log10Floored against its defining composition (Clamp
// to the floor, then Log10) for bit-identity on every tier, and checks the
// finite-result guarantee: a positive finite floor maps every finite input to a
// finite log10.
func FuzzF64Log10Floored(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f64sBits(raw)
		if len(v) < 2 {
			return
		}
		floor := v[0]
		src := v[1:]
		got := make([]float64, len(src))
		Log10Floored(got, src, floor)
		want := make([]float64, len(src))
		Clamp(want, src, floor, math.Inf(1))
		Log10(want, want)
		exactEqualF64(t, "Log10Floored", got, want)
		// Finite-result guarantee: for a positive finite floor, every finite input
		// yields a finite result (a zero/negative/tiny input is lifted to floor).
		if floor > 0 && !math.IsInf(floor, 0) {
			for i, g := range got {
				s := src[i]
				if math.IsNaN(s) || math.IsInf(s, 0) {
					continue // out-of-domain input; the guarantee does not apply
				}
				if math.IsNaN(g) || math.IsInf(g, 0) {
					t.Fatalf("Log10Floored(finite src=%v, floor=%v) not finite: %v", src[i], floor, g)
				}
			}
		}
	})
}
