package f64

import (
	"errors"
	"math"
	"math/bits"
	"math/rand/v2"
	"reflect"
	"strconv"
	"sync"
	"testing"
)

// fftUnitRoundoff is u = 2^-53, the unit roundoff of float64. eps64 is 2^-52.
const fftUnitRoundoff = eps64 / 2

// fftTolerance is the relative-L2 bound the FFT accuracy tests share:
// 2*u*max(1, log2 n). An FFT with directly computed twiddles has error
// O(u*log2 n) in relative L2.
func fftTolerance(n int) float64 {
	return 2 * fftUnitRoundoff * float64(max(1, bits.Len(uint(n))-1))
}

// fftTestInput returns n complex values with real and imaginary parts drawn
// from [-1, 1) by a seeded generator.
func fftTestInput(n int, seed uint64) (re, im []float64) {
	rng := rand.New(rand.NewPCG(seed, uint64(n)))
	re = make([]float64, n)
	im = make([]float64, n)
	for i := range re {
		re[i] = rng.Float64()*2 - 1
		im[i] = rng.Float64()*2 - 1
	}
	return re, im
}

// dot2 is a compensated accumulator (Ogita, Rump and Oishi's Dot2): products are
// split exactly with an FMA (TwoProduct) and sums with TwoSum, so the result is
// as accurate as a double-double evaluation rounded once.
type dot2 struct{ s, c float64 }

func (d *dot2) add(a, b float64) {
	p := a * b
	e := math.FMA(a, b, -p)
	s := d.s + p
	z := s - d.s
	q := (d.s - (s - z)) + (p - z)
	d.s = s
	d.c += q + e
}

func (d *dot2) value() float64 { return d.s + d.c }

// compensatedDFT returns the bins of the forward DFT of (xr, xi),
// X[k] = sum_j x[j]*exp(-2*pi*i*j*k/n). The twiddle index j*k mod n advances by
// an add with wrap, so there is no overflow and no large angle; its own error is
// about one twiddle rounding.
func compensatedDFT(xr, xi []float64, bins []int) (outRe, outIm []float64) {
	n := len(xr)
	wr := make([]float64, n)
	wi := make([]float64, n)
	for r := range n {
		s, c := math.Sincos(2 * math.Pi * float64(r) / float64(n))
		wr[r], wi[r] = c, s
	}
	outRe = make([]float64, len(bins))
	outIm = make([]float64, len(bins))
	for b, k := range bins {
		var accRe, accIm dot2
		r := 0
		for j := range n {
			// (xr + i*xi) * (c - i*s)
			accRe.add(xr[j], wr[r])
			accRe.add(xi[j], wi[r])
			accIm.add(xi[j], wr[r])
			accIm.add(-xr[j], wi[r])
			if r += k; r >= n {
				r -= n
			}
		}
		outRe[b], outIm[b] = accRe.value(), accIm.value()
	}
	return outRe, outIm
}

// relL2 is the relative L2 distance between two split-complex vectors.
func relL2(gotRe, gotIm, wantRe, wantIm []float64) float64 {
	var num, den float64
	for i := range gotRe {
		dr, di := gotRe[i]-wantRe[i], gotIm[i]-wantIm[i]
		num += dr*dr + di*di
		den += wantRe[i]*wantRe[i] + wantIm[i]*wantIm[i]
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

func powersOfTwo(maxN int) []int {
	var out []int
	for n := 1; n <= maxN; n <<= 1 {
		out = append(out, n)
	}
	return out
}

// fftSampledBins returns every bin for small n, and for large n the bins
// 0, 1, n-1 and every n/64-th one.
func fftSampledBins(n int) []int {
	const fullBinsLimit = 1024
	const strideDivisor = 64
	if n <= fullBinsLimit {
		bins := make([]int, n)
		for k := range bins {
			bins[k] = k
		}
		return bins
	}
	bins := []int{0, 1, n - 1}
	for k := n / strideDivisor; k < n; k += n / strideDivisor {
		bins = append(bins, k)
	}
	return bins
}

// TestFFTPlanForwardMatchesCompensatedDFT pins the transform's value: a wrong
// stage schedule, twiddle slice or offset, sign, or bit-reversal shift moves bins
// by O(1), and a recurrence-based twiddle would exceed the O(u*log2 n) bound.
// Measured relL2/(u*max(1, log2 n)) is at most 0.60 for n <= 4096 on an Apple M5
// Pro (NEON) and 0.58 with SIMD_DISABLE=all; on an Intel i5-7260U it is at most
// 0.64 with AVX+FMA and with SIMD_DISABLE=avx or all. Tier agreement comes from
// running this test on each tier.
func TestFFTPlanForwardMatchesCompensatedDFT(t *testing.T) {
	const maxN = 1 << 16
	p, err := NewFFTPlan(maxN)
	if err != nil {
		t.Fatal(err)
	}
	sizes := append(powersOfTwo(1<<10), 1<<12, maxN)
	for _, n := range sizes {
		xr, xi := fftTestInput(n, 1)
		re := append([]float64(nil), xr...)
		im := append([]float64(nil), xi...)
		if got := p.Forward(re, im); got != n {
			t.Fatalf("n=%d: Forward returned %d", n, got)
		}
		bins := fftSampledBins(n)
		wantRe, wantIm := compensatedDFT(xr, xi, bins)
		gotRe := make([]float64, len(bins))
		gotIm := make([]float64, len(bins))
		for b, k := range bins {
			gotRe[b], gotIm[b] = re[k], im[k]
		}
		if e := relL2(gotRe, gotIm, wantRe, wantIm); e > fftTolerance(n) {
			t.Errorf("n=%d: relL2 = %g, want <= %g", n, e, fftTolerance(n))
		}
	}
}

// TestFFTPlanRoundTrip pins Inverse: Inverse(Forward(x))/n must return x, so a
// missing or wrong swap, or any scaling inside Inverse, fails. Measured 4.6e-16
// (Apple M5 Pro) and 4.8e-16 (Intel i5-7260U, AVX+FMA and pure Go) against a
// bound of 3.6e-15 at n = 65536.
func TestFFTPlanRoundTrip(t *testing.T) {
	const maxN = 1 << 16
	p, err := NewFFTPlan(maxN)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range powersOfTwo(maxN) {
		xr, xi := fftTestInput(n, 2)
		re := append([]float64(nil), xr...)
		im := append([]float64(nil), xi...)
		p.Forward(re, im)
		if got := p.Inverse(re, im); got != n {
			t.Fatalf("n=%d: Inverse returned %d", n, got)
		}
		inv := 1 / float64(n) // exact: n is a power of two
		for i := range re {
			re[i] *= inv
			im[i] *= inv
		}
		if e := relL2(re, im, xr, xi); e > fftTolerance(n) {
			t.Errorf("n=%d: round trip relL2 = %g, want <= %g", n, e, fftTolerance(n))
		}
	}
}

// TestFFTPlanSubSizeBitIdentical pins that one large plan serves every smaller
// power of two bit for bit like a dedicated plan, for Forward and Inverse. It
// catches a wrong bit-reversal shift, a maxN-dependent table offset, or a stage
// loop bounded by maxN.
func TestFFTPlanSubSizeBitIdentical(t *testing.T) {
	const maxN = 1 << 16
	big, err := NewFFTPlan(maxN)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range powersOfTwo(maxN) {
		small, err := NewFFTPlan(n)
		if err != nil {
			t.Fatal(err)
		}
		xr, xi := fftTestInput(n, 3)
		for name, run := range map[string]func(p *FFTPlan, re, im []float64) int{
			"Forward": (*FFTPlan).Forward,
			"Inverse": (*FFTPlan).Inverse,
		} {
			ar, ai := append([]float64(nil), xr...), append([]float64(nil), xi...)
			br, bi := append([]float64(nil), xr...), append([]float64(nil), xi...)
			run(big, ar, ai)
			run(small, br, bi)
			if i, ok := bitsDiff(ar, br); !ok {
				t.Fatalf("n=%d %s: re[%d] differs between the %d plan and a dedicated plan", n, name, i, maxN)
			}
			if i, ok := bitsDiff(ai, bi); !ok {
				t.Fatalf("n=%d %s: im[%d] differs between the %d plan and a dedicated plan", n, name, i, maxN)
			}
		}
	}
}

// TestFFTPlanConcurrentUse shares one plan between goroutines that own their
// buffers and checks each result against a sequential run bit for bit. Under
// -race it also catches scratch added to the plan later.
func TestFFTPlanConcurrentUse(t *testing.T) {
	const iterations = 20
	p, err := NewFFTPlan(4096)
	if err != nil {
		t.Fatal(err)
	}
	sizes := []int{4096, 1024, 64, 2}
	type want struct{ fwdRe, fwdIm, invRe, invIm []float64 }
	wants := make([]want, len(sizes))
	inputs := make([][2][]float64, len(sizes))
	for g, n := range sizes {
		xr, xi := fftTestInput(n, uint64(10+g))
		inputs[g] = [2][]float64{xr, xi}
		re := append([]float64(nil), xr...)
		im := append([]float64(nil), xi...)
		p.Forward(re, im)
		w := want{fwdRe: append([]float64(nil), re...), fwdIm: append([]float64(nil), im...)}
		p.Inverse(re, im)
		w.invRe, w.invIm = re, im
		wants[g] = w
	}
	same := func(a, b []float64) bool {
		_, ok := bitsDiff(a, b)
		return ok
	}
	var wg sync.WaitGroup
	for g, n := range sizes {
		wg.Go(func() {
			re, im := make([]float64, n), make([]float64, n)
			for range iterations {
				copy(re, inputs[g][0])
				copy(im, inputs[g][1])
				p.Forward(re, im)
				if !same(re, wants[g].fwdRe) || !same(im, wants[g].fwdIm) {
					t.Errorf("goroutine %d (n=%d): Forward differs from the sequential run", g, n)
					return
				}
				p.Inverse(re, im)
				if !same(re, wants[g].invRe) || !same(im, wants[g].invIm) {
					t.Errorf("goroutine %d (n=%d): Inverse differs from the sequential run", g, n)
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestFFTPlanAllocFree(t *testing.T) {
	p, err := NewFFTPlan(1 << 16)
	if err != nil {
		t.Fatal(err)
	}
	re, im := fftTestInput(4096, 4) // a sub-size of the plan
	if a := testing.AllocsPerRun(10, func() { p.Forward(re, im) }); a != 0 {
		t.Errorf("Forward allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(10, func() { p.Inverse(re, im) }); a != 0 {
		t.Errorf("Inverse allocated %v times per run, want 0", a)
	}
}

func TestNewFFTPlanRejectsInvalidSize(t *testing.T) {
	bad := []int{0, -1, 3, 6, 12, maxFFTSize + 1, 2 * maxFFTSize}
	for _, n := range bad {
		p, err := NewFFTPlan(n)
		if !errors.Is(err, ErrFFTSize) || p != nil {
			t.Errorf("NewFFTPlan(%d) = (%v, %v), want (nil, ErrFFTSize)", n, p, err)
		}
	}
	for _, n := range []int{1, 2, 4, 1024} {
		p, err := NewFFTPlan(n)
		if err != nil {
			t.Errorf("NewFFTPlan(%d) unexpected error: %v", n, err)
			continue
		}
		if p.MaxLen() != n {
			t.Errorf("NewFFTPlan(%d).MaxLen() = %d", n, p.MaxLen())
		}
	}
}

// TestFFTPlanInvalidLengthIsNoOp pins the validation: an invalid call returns 0
// and leaves both slices bitwise unchanged. A hole would hand a non-power-of-two
// to the core, which computes garbage silently instead of panicking.
func TestFFTPlanInvalidLengthIsNoOp(t *testing.T) {
	const maxN = 16
	p, err := NewFFTPlan(maxN)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name         string
		lenRe, lenIm int
	}{
		{"empty", 0, 0},
		{"length mismatch", 8, 9},
		{"length mismatch other way", 9, 8},
		{"not a power of two 3", 3, 3},
		{"not a power of two 6", 6, 6},
		{"above MaxLen", 2 * maxN, 2 * maxN},
	}
	for name, run := range map[string]func(re, im []float64) int{"Forward": p.Forward, "Inverse": p.Inverse} {
		for _, c := range cases {
			re, _ := fftTestInput(c.lenRe, 5)
			im, _ := fftTestInput(c.lenIm, 6)
			wantRe := append([]float64(nil), re...)
			wantIm := append([]float64(nil), im...)
			if got := run(re, im); got != 0 {
				t.Errorf("%s %s: returned %d, want 0", name, c.name, got)
			}
			if i, ok := bitsDiff(re, wantRe); !ok {
				t.Fatalf("%s %s: re[%d] was modified", name, c.name, i)
			}
			if i, ok := bitsDiff(im, wantIm); !ok {
				t.Fatalf("%s %s: im[%d] was modified", name, c.name, i)
			}
		}
		re, im := fftTestInput(maxN, 7)
		if got := run(re, im); got != maxN {
			t.Errorf("%s valid call returned %d, want %d", name, got, maxN)
		}
	}
}

// fftPlanFootprint is the documented memory formula of FFTPlan in bytes: 4*maxN
// of int32 indices, 2*(maxN-1) float64 radix-2 twiddles, and 2*(4^q-1)/3 float64
// radix-4 twiddles with q = floor(log2(maxN)/2).
func fftPlanFootprint(maxN int) int {
	const (
		indexBytes   = 4
		float64Bytes = 8
		pair         = 2
	)
	q := (bits.Len(uint(maxN)) - 1) / 2
	radix4 := (1<<(2*q) - 1) / 3
	return indexBytes*maxN + pair*float64Bytes*(maxN-1) + pair*float64Bytes*radix4
}

// sliceBytes sums Len()*elemSize over every slice reachable through the struct
// fields of v. It fails on field kinds it does not know, so a field added to the
// plan cannot escape the footprint check.
func sliceBytes(t *testing.T, v reflect.Value) int {
	t.Helper()
	total := 0
	for i := range v.NumField() {
		f := v.Field(i)
		switch f.Kind() {
		case reflect.Slice:
			total += f.Len() * int(f.Type().Elem().Size())
		case reflect.Struct:
			total += sliceBytes(t, f)
		case reflect.Int:
			// scalar bookkeeping, not table memory
		default:
			t.Fatalf("field %s has unexpected kind %s", v.Type().Field(i).Name, f.Kind())
		}
	}
	return total
}

// TestFFTPlanFootprint pins the memory figure quoted in the FFTPlan docs. It
// also catches scratch added to the plan, which would break the concurrency
// claim.
func TestFFTPlanFootprint(t *testing.T) {
	for _, n := range []int{1, 2, 4, 8, 16, 1024, 1 << 16} {
		p, err := NewFFTPlan(n)
		if err != nil {
			t.Fatal(err)
		}
		got := sliceBytes(t, reflect.ValueOf(*p))
		if want := fftPlanFootprint(n); got != want {
			t.Errorf("maxN=%d: plan holds %d bytes, formula says %d", n, got, want)
		}
	}
	const documented = 1660224 // maxN = 65536
	if got := fftPlanFootprint(1 << 16); got != documented {
		t.Errorf("formula at 65536 = %d, documented figure is %d", got, documented)
	}
}

func BenchmarkFFTPlanForward(b *testing.B) {
	const maxN = 1 << 16
	p, err := NewFFTPlan(maxN)
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{256, 4096, 8192, 32768, 65536} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			xr, xi := fftTestInput(n, 8)
			re, im := make([]float64, n), make([]float64, n)
			b.ReportAllocs()
			// The input is refreshed every iteration because repeated unscaled
			// transforms grow the magnitudes without bound; the copy is a few
			// percent of the transform.
			for b.Loop() {
				copy(re, xr)
				copy(im, xi)
				p.Forward(re, im)
			}
		})
	}
}

func BenchmarkNewFFTPlan(b *testing.B) {
	for _, n := range []int{1 << 15, 1 << 16} {
		b.Run("maxN"+strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := NewFFTPlan(n); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
