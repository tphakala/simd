package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// =============================================================================
// INT32 -> FLOAT32 SCALE-ADD TESTS (fused dequantize-accumulate)
// =============================================================================
//
// Int32ToFloat32ScaleAdd computes dst[i] = a[i] + float32(src[i])*scale with the
// product rounded to float32 before the add (two roundings, never an FMA). The
// contract is bit-identical results on every dispatch path AND bit-identical to the
// two-pass Int32ToFloat32Scale + Add composition it replaces, so every comparison
// below is exact (Float32bits), not tolerance-based.

// twoPassScaleAddRef is the reference composition the fused kernel replaces:
// tmp = float32(src)*scale, then dst = a + tmp. It carries the same two roundings,
// so a correct fused kernel matches it bit-for-bit.
func twoPassScaleAddRef(dst, a []float32, src []int32, scale float32) {
	n := min(len(dst), len(a), len(src))
	tmp := make([]float32, n)
	Int32ToFloat32Scale(tmp, src[:n], scale)
	Add(dst[:n], a[:n], tmp)
}

// assertBitsEqualF32 fails if got and want differ in any bit on any element.
func assertBitsEqualF32(t *testing.T, got, want []float32, ctx string) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("%s: [%d] got %v (%#08x), want %v (%#08x)",
				ctx, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// fillScaleAddInputs fills a and src with deterministic pseudo-random values that
// exercise sign, magnitude and the low int32 bits that float32 conversion rounds
// away.
func fillScaleAddInputs(rng *rand.Rand, a []float32, src []int32) {
	for i := range a {
		a[i] = (rng.Float32() - 0.5) * 1e6
	}
	for i := range src {
		// Mix small counts, mid-range values and full-width int32s so the float32
		// conversion loses low bits on some lanes (the interesting rounding case).
		switch i % 4 {
		case 0:
			src[i] = int32(rng.Intn(64))
		case 1:
			src[i] = rng.Int31() - (1 << 30)
		case 2:
			src[i] = rng.Int31() // may exceed float32's 24-bit mantissa
		default:
			src[i] = -rng.Int31()
		}
	}
}

func TestInt32ToFloat32ScaleAdd(t *testing.T) {
	// Cover the AVX block+scalar-tail boundary (amd64 n>=8), the NEON block+tail
	// (arm64 n>=4), the pure-Go path (small n), and several full blocks.
	sizes := []int{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 256, 257, 1000}
	scales := []float32{1, 0.5, 1.0 / 32768, 1.0 / 2147483648, -2.5, 3.14159}
	rng := rand.New(rand.NewSource(0x5add))

	for _, n := range sizes {
		for _, scale := range scales {
			a := make([]float32, n)
			src := make([]int32, n)
			fillScaleAddInputs(rng, a, src)

			got := make([]float32, n)
			Int32ToFloat32ScaleAdd(got, a, src, scale)

			// vs pure-Go reference (bit-exact)
			goRef := make([]float32, n)
			int32ToFloat32ScaleAddGo(goRef, a, src, scale)
			assertBitsEqualF32(t, got, goRef, "dispatched vs Go")

			// vs two-pass Int32ToFloat32Scale + Add (bit-exact, the motivating
			// equivalence)
			twoPass := make([]float32, n)
			twoPassScaleAddRef(twoPass, a, src, scale)
			assertBitsEqualF32(t, got, twoPass, "dispatched vs two-pass")
		}
	}
}

// TestInt32ToFloat32ScaleAdd_InPlace pins the documented aliasing rule: dst may
// overlay a exactly (the in-place accumulate a[i] += float32(src[i])*scale). The
// kernel loads a before it stores dst within each block, so the exact overlay is
// safe; the result must match the same computation into a distinct buffer.
func TestInt32ToFloat32ScaleAdd_InPlace(t *testing.T) {
	sizes := []int{4, 8, 9, 17, 33, 64, 257}
	rng := rand.New(rand.NewSource(0xacc))
	const scale = 1.0 / 32768

	for _, n := range sizes {
		a := make([]float32, n)
		src := make([]int32, n)
		fillScaleAddInputs(rng, a, src)

		// Expected: fresh out = a + src*scale.
		want := make([]float32, n)
		Int32ToFloat32ScaleAdd(want, a, src, scale)

		// In place: dst aliases a exactly.
		inPlace := make([]float32, n)
		copy(inPlace, a)
		Int32ToFloat32ScaleAdd(inPlace, inPlace, src, scale)

		assertBitsEqualF32(t, inPlace, want, "in-place vs fresh")
	}
}

// TestInt32ToFloat32ScaleAdd_ShortSlices pins the min-length reconciliation: the
// number of elements processed is min(len(dst), len(a), len(src)), and elements of
// dst beyond that count are left untouched.
func TestInt32ToFloat32ScaleAdd_ShortSlices(t *testing.T) {
	const full = 40
	const scale = 0.25
	rng := rand.New(rand.NewSource(0x511c))

	cases := []struct{ dstLen, aLen, srcLen int }{
		{full, full, full},
		{full, full - 5, full},   // a shortest
		{full, full, full - 7},   // src shortest
		{full - 9, full, full},   // dst shortest
		{full - 3, full - 6, full - 1},
	}
	for _, c := range cases {
		a := make([]float32, full)
		src := make([]int32, full)
		fillScaleAddInputs(rng, a, src)
		const sentinel = -7.5
		dst := make([]float32, full)
		for i := range dst {
			dst[i] = sentinel
		}

		Int32ToFloat32ScaleAdd(dst[:c.dstLen], a[:c.aLen], src[:c.srcLen], scale)

		n := min(c.dstLen, c.aLen, c.srcLen)
		want := make([]float32, n)
		twoPassScaleAddRef(want, a[:c.aLen], src[:c.srcLen], scale)
		assertBitsEqualF32(t, dst[:n], want, "short-slice processed range")
		for i := n; i < full; i++ {
			if dst[i] != sentinel {
				t.Errorf("dst[%d] = %v, want untouched sentinel %v (processed=%d)", i, dst[i], sentinel, n)
			}
		}
	}
}

func TestInt32ToFloat32ScaleAdd_Empty(t *testing.T) {
	// Nil and zero-length must return without touching anything or panicking.
	Int32ToFloat32ScaleAdd(nil, nil, nil, 1.5)
	dst := []float32{42}
	Int32ToFloat32ScaleAdd(dst[:0], []float32{1}, []int32{2}, 1.5)
	if dst[0] != 42 {
		t.Errorf("zero-length call wrote output: dst[0] = %v, want 42", dst[0])
	}
}

// TestInt32ToFloat32ScaleAdd_AllocFree pins the zero-allocation guarantee. Direct
// cpu-flag dispatch keeps //go:noescape effective.
func TestInt32ToFloat32ScaleAdd_AllocFree(t *testing.T) {
	const n = 256
	a := make([]float32, n)
	src := make([]int32, n)
	dst := make([]float32, n)
	rng := rand.New(rand.NewSource(1))
	fillScaleAddInputs(rng, a, src)
	fn := func() { Int32ToFloat32ScaleAdd(dst, a, src, 1.0/32768) }
	if got := testing.AllocsPerRun(50, fn); got != 0 {
		t.Errorf("Int32ToFloat32ScaleAdd allocated %v times per run, want 0", got)
	}
}

// FuzzInt32ToFloat32ScaleAdd is the differential fuzz: the dispatched kernel must
// bit-match the pure-Go reference for arbitrary src bytes and scale.
func FuzzInt32ToFloat32ScaleAdd(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 1, 2, 3, 4}, math.Float32bits(1.0/32768), uint32(0x3f000000))
	f.Add([]byte{0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0x80}, uint32(0x3f800000), uint32(0xc0000000))

	f.Fuzz(func(t *testing.T, srcBytes []byte, scaleBits, aBits uint32) {
		n := len(srcBytes) / 4
		if n == 0 {
			return
		}
		src := make([]int32, n)
		for i := range n {
			src[i] = int32(uint32(srcBytes[4*i]) | uint32(srcBytes[4*i+1])<<8 |
				uint32(srcBytes[4*i+2])<<16 | uint32(srcBytes[4*i+3])<<24)
		}
		scale := math.Float32frombits(scaleBits)
		a := make([]float32, n)
		av := math.Float32frombits(aBits)
		for i := range a {
			// Vary the accumulator per lane while staying deterministic in the input.
			a[i] = av + float32(i)
		}

		got := make([]float32, n)
		Int32ToFloat32ScaleAdd(got, a, src, scale)
		want := make([]float32, n)
		int32ToFloat32ScaleAddGo(want, a, src, scale)

		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("[%d] dispatched %#08x != Go %#08x (src=%d scale=%v a=%v)",
					i, math.Float32bits(got[i]), math.Float32bits(want[i]), src[i], scale, a[i])
			}
		}
	})
}

// BenchmarkInt32ToFloat32ScaleAdd compares the fused kernel against the two-pass
// Int32ToFloat32Scale + Add it replaces. Small n (8..32) is where the win is the
// dropped temporary and pass rather than vector width; the motivating call sites
// (go-aac NMR trellis prep) run at n = 11..17.
func BenchmarkInt32ToFloat32ScaleAdd(b *testing.B) {
	sizes := []int{8, 12, 16, 32, 128, 1024}
	const scale = 1.0 / 32768

	for _, n := range sizes {
		a := make([]float32, n)
		src := make([]int32, n)
		dst := make([]float32, n)
		tmp := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n)))
		fillScaleAddInputs(rng, a, src)

		b.Run(fmt.Sprintf("Fused_n%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4 * 3) // reads a, src; writes dst
			for range b.N {
				Int32ToFloat32ScaleAdd(dst, a, src, scale)
			}
		})
		b.Run(fmt.Sprintf("TwoPass_n%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4 * 5) // src->tmp, then a+tmp->dst
			for range b.N {
				Int32ToFloat32Scale(tmp, src, scale)
				Add(dst, a, tmp)
			}
		})
	}
}
