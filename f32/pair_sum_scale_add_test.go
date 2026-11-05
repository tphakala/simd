package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// pairSumInputs fills acc, a and b with values spread over wide magnitudes and
// signs. With windowed set, a and b are two windows of one buffer at an odd
// shift (the symmetric-FIR shape), otherwise independent slices.
func pairSumInputs(rng *rand.Rand, n int, windowed bool) (acc, a, b []float32) {
	draw := func() float32 {
		return float32(math.Ldexp(rng.Float64()*2-1, rng.Intn(40)-20))
	}
	acc = make([]float32, n)
	for i := range acc {
		acc[i] = draw()
	}
	if windowed {
		shift := 3 + rng.Intn(19)
		buf := make([]float32, n+shift)
		for i := range buf {
			buf[i] = draw()
		}
		return acc, buf[:n], buf[shift : shift+n]
	}
	a = make([]float32, n)
	b = make([]float32, n)
	for i := range a {
		a[i] = draw()
		b[i] = draw()
	}
	return acc, a, b
}

func TestPairSumScaleAdd(t *testing.T) {
	sizes := []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 257, 1024}
	scales := []float32{1, -1, 0.5, 3.14159, -2.5e-3, 32768, 1e30}
	rng := rand.New(rand.NewSource(1))
	for _, n := range sizes {
		for _, c := range scales {
			for _, windowed := range []bool{false, true} {
				acc, a, b := pairSumInputs(rng, n, windowed)
				got := make([]float32, n)
				ref := make([]float32, n)
				want := make([]float32, n)
				tmp := make([]float32, n)
				PairSumScaleAdd(got, acc, a, b, c)
				pairSumScaleAddGo(ref, acc, a, b, c)
				Add(tmp, a, b)
				Scale(tmp, tmp, c)
				Add(want, acc, tmp)
				for i := range got {
					if !bitsEqF32(got[i], ref[i]) || !bitsEqF32(got[i], want[i]) {
						t.Fatalf("n=%d c=%g windowed=%v i=%d: got %#x go %#x composition %#x",
							n, c, windowed, i, math.Float32bits(got[i]), math.Float32bits(ref[i]), math.Float32bits(want[i]))
					}
				}
			}
		}
	}
}

func TestPairSumScaleAdd_RoundingBoundaries(t *testing.T) {
	const (
		u11 = 1.0 / (1 << 11)
		u12 = 1.0 / (1 << 12)
		u24 = 1.0 / (1 << 24)
	)
	negZero := float32(math.Copysign(0, -1))
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	tests := []struct {
		name         string
		acc, a, b, c float32
		want         uint32
		wantNaN      bool
	}{
		{"fma boundary", -(1 + u11), 1, u12, 1 + u12, 0x00000000, false},
		{"reassociation boundary", 0, 1, u24, 3, 0x40400000, false},
		{"overflow of the pair sum", 0, math.MaxFloat32, math.MaxFloat32, 0.5, 0x7f800000, false},
		{"sign of zero, negative scale", negZero, 1, -1, -1, 0x80000000, false},
		{"sign of zero, positive scale", negZero, 1, -1, 1, 0x00000000, false},
		{"subnormal kept, rounds to zero", 0, math.SmallestNonzeroFloat32, 0, 0.5, 0x00000000, false},
		{"subnormal kept, rounds to one", 0, math.SmallestNonzeroFloat32, 0, 0.75, 0x00000001, false},
		{"inf minus inf", 0, inf, -inf, 1, 0, true},
		{"zero times inf", 0, 0, 0, inf, 0, true},
		{"nan input", 0, nan, 1, 1, 0, true},
	}
	const n = 19 // lands in vector lanes and the scalar tail on both kernels
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rep := func(x float32) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = x
				}
				return s
			}
			check := func(impl string, got []float32) {
				for i, g := range got {
					if tc.wantNaN {
						if g == g {
							t.Fatalf("%s lane %d: got %#x want NaN", impl, i, math.Float32bits(g))
						}
						continue
					}
					if math.Float32bits(g) != tc.want {
						t.Fatalf("%s lane %d: got %#x want %#x", impl, i, math.Float32bits(g), tc.want)
					}
				}
			}
			dst := make([]float32, n)
			PairSumScaleAdd(dst, rep(tc.acc), rep(tc.a), rep(tc.b), tc.c)
			check("PairSumScaleAdd", dst)
			// The direct Go call pins the Go barriers on architectures where the
			// public path never reaches the reference at n >= 4.
			pairSumScaleAddGo(dst, rep(tc.acc), rep(tc.a), rep(tc.b), tc.c)
			check("pairSumScaleAddGo", dst)
		})
	}
}

func TestPairSumScaleAdd_ShortSlices(t *testing.T) {
	const full = 21
	const sentinel = float32(-777)
	mk := func(n int, v float32) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = v + float32(i)*0.37
		}
		return s
	}
	for _, short := range []string{"dst", "acc", "a", "b"} {
		for _, n := range []int{0, 1, 5, 9} {
			lens := map[string]int{"dst": full, "acc": full, "a": full, "b": full}
			lens[short] = n
			dst := make([]float32, lens["dst"])
			for i := range dst {
				dst[i] = sentinel
			}
			acc, a, b := mk(lens["acc"], 1), mk(lens["a"], 2), mk(lens["b"], -3)
			PairSumScaleAdd(dst, acc, a, b, 1.5)
			want := make([]float32, n)
			pairSumScaleAddGo(want, acc[:n], a[:n], b[:n], 1.5)
			for i := range dst {
				exp := sentinel
				if i < n {
					exp = want[i]
				}
				if !bitsEqF32(dst[i], exp) {
					t.Fatalf("short %s n=%d i=%d: got %v want %v", short, n, i, dst[i], exp)
				}
			}
		}
	}
	PairSumScaleAdd(nil, mk(4, 1), mk(4, 1), mk(4, 1), 2) // zero-length dst must not panic
}

// TestPairSumScaleAdd_AllocFree declares the arrays inside the measured closure so
// a dispatch that leaks its parameters (a function variable, a missing
// //go:noescape) forces the caller to heap-allocate and fails here.
func TestPairSumScaleAdd_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(100, func() {
		var dst, acc, a, b [37]float32
		PairSumScaleAdd(dst[:], acc[:], a[:], b[:], 0.5)
	}); n != 0 {
		t.Errorf("PairSumScaleAdd forces %v caller allocations per run, want 0", n)
	}
}

func BenchmarkPairSumScaleAdd(b *testing.B) {
	for _, n := range []int{16, 128, 1024} {
		rng := rand.New(rand.NewSource(2))
		acc, x, y := pairSumInputs(rng, n, true)
		dst := make([]float32, n)
		tmp := make([]float32, n)
		b.Run(fmt.Sprintf("Fused_n%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for range b.N {
				PairSumScaleAdd(dst, acc, x, y, 0.37)
			}
		})
		b.Run(fmt.Sprintf("Composition_n%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for range b.N {
				Add(tmp, x, y)
				Scale(tmp, tmp, 0.37)
				Add(dst, acc, tmp)
			}
		})
	}
}
