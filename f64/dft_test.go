package f64

import (
	"errors"
	"math"
	"math/bits"
	"reflect"
	"strconv"
	"testing"
)

// dftTestCapacity is the capacity of the plan the accuracy and sequence tests
// share, so those tests also exercise retargeting across sizes. It is the
// largest length they transform.
const dftTestCapacity = 32769

// dftInnerSize is the size of the power-of-two FFTs a transform of length n runs:
// n itself when n is a power of two, otherwise nextPow2(2n-1).
func dftInnerSize(n int) int {
	if n&(n-1) == 0 {
		return n
	}
	return nextPow2(2*n - 1)
}

// dftTolerance is the relative-L2 bound for DFTPlan accuracy tests: factor *
// u * max(1, log2 M), where M is the size of the FFTs the transform runs.
func dftTolerance(n int, factor float64) float64 {
	return factor * fftUnitRoundoff * float64(max(1, bits.Len(uint(dftInnerSize(n)))-1))
}

// dftBins returns every bin for n <= 257 and otherwise 32 bins: 0, 1, n-1 and 29
// evenly spaced ones.
func dftBins(n int) []int {
	const (
		allBinsLimit = 257
		spaced       = 29
	)
	if n <= allBinsLimit {
		return fftSampledBins(n) // every bin up to 1024
	}
	bins := []int{0, 1, n - 1}
	for i := 1; i <= spaced; i++ {
		bins = append(bins, i*n/(spaced+1))
	}
	return bins
}

// TestDFTPlanForwardMatchesCompensatedDFT pins the transform's value on prime,
// composite and power-of-two lengths against the compensated reference DFT. One
// plan serves every case, so retargeting across sizes is covered too.
// Measured relL2/(u*max(1, log2 M)) peaks at n = 7 and is at most 1.66 on an
// Apple M5 Pro and a Cortex-A76 (NEON, and 1.62 with SIMD_DISABLE=all) and at most
// 1.60 on an Intel i5-7260U (AVX+FMA, and 1.58 with SIMD_DISABLE=avx or all); the
// bound of 5 keeps about 3x headroom.
func TestDFTPlanForwardMatchesCompensatedDFT(t *testing.T) {
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		t.Fatal(err)
	}
	sizes := make([]int, 0, 64+13)
	for n := 1; n <= 64; n++ {
		sizes = append(sizes, n)
	}
	sizes = append(sizes, 97, 100, 255, 257, 1000, 1025, 4093, 4097, 8191, 32749, 32767, 32768, 32769)
	worst, worstN := 0.0, 0
	for _, n := range sizes {
		xr, xi := fftTestInput(n, 21)
		re := append([]float64(nil), xr...)
		im := append([]float64(nil), xi...)
		if got := p.Forward(re, im); got != n {
			t.Fatalf("n=%d: Forward returned %d", n, got)
		}
		bins := dftBins(n)
		wantRe, wantIm := compensatedDFT(xr, xi, bins)
		gotRe := make([]float64, len(bins))
		gotIm := make([]float64, len(bins))
		for b, k := range bins {
			gotRe[b], gotIm[b] = re[k], im[k]
		}
		e := relL2(gotRe, gotIm, wantRe, wantIm)
		if r := e / dftTolerance(n, 1); r > worst {
			worst, worstN = r, n
		}
		if tol := dftTolerance(n, dftAccuracyFactor); e > tol {
			t.Errorf("n=%d: relL2 = %g, want <= %g", n, e, tol)
		}
	}
	t.Logf("worst relL2/(u*max(1, log2 M)) = %.3f at n=%d", worst, worstN)
}

// TestDFTPlanRoundTrip pins Inverse: Inverse(Forward(x))/n must return x, so a
// wrong swap or any scaling inside Inverse fails. Measured
// relL2/(u*max(1, log2 M)) is at most 1.55 on the M5 Pro, the A76 and with
// SIMD_DISABLE=all, and 1.45 on the i5-7260U; the bound of 5 keeps about 3x
// headroom.
func TestDFTPlanRoundTrip(t *testing.T) {
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for _, n := range []int{3, 5, 6, 100, 1000, 4097, 32749, 32769} {
		xr, xi := fftTestInput(n, 22)
		re := append([]float64(nil), xr...)
		im := append([]float64(nil), xi...)
		p.Forward(re, im)
		if got := p.Inverse(re, im); got != n {
			t.Fatalf("n=%d: Inverse returned %d", n, got)
		}
		inv := 1 / float64(n)
		for i := range re {
			re[i] *= inv
			im[i] *= inv
		}
		e := relL2(re, im, xr, xi)
		worst = max(worst, e/dftTolerance(n, 1))
		if tol := dftTolerance(n, dftRoundTripFactor); e > tol {
			t.Errorf("n=%d: round trip relL2 = %g, want <= %g", n, e, tol)
		}
	}
	t.Logf("worst round trip relL2/(u*max(1, log2 M)) = %.3f", worst)
}

// Multiples of u*max(1, log2 M) the accuracy and round-trip tests allow.
const (
	dftAccuracyFactor  = 5
	dftRoundTripFactor = 5
)

// dftLengthSequence revisits lengths, interleaves powers of two with
// non-powers of two, both shrinks and grows the Bluestein target, and has two
// consecutive lengths (6 and 7) that share the inner size 16.
var dftLengthSequence = []int{1000, 7, 32769, 1000, 4096, 6, 7, 32749, 7, 1000, 3, 2, 1, 32769}

// TestDFTPlanLengthSequenceBitIdentical pins that a result depends only on n and
// the CPU tier: one plan driven through a sequence of lengths must match a plan
// built for exactly that length bit for bit, and a power-of-two length must match
// FFTPlan. It catches stale scratch from an earlier target, an inner size taken
// from the capacity, and a skipped or mis-keyed retarget.
func TestDFTPlanLengthSequenceBitIdentical(t *testing.T) {
	shared, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		t.Fatal(err)
	}
	type result struct{ re, im []float64 }
	dedicated := map[int]result{}
	for _, n := range dftLengthSequence {
		if _, ok := dedicated[n]; ok {
			continue
		}
		d, err := NewDFTPlan(n)
		if err != nil {
			t.Fatal(err)
		}
		re, im := fftTestInput(n, 23)
		d.Forward(re, im)
		dedicated[n] = result{re, im}
	}
	for step, n := range dftLengthSequence {
		re, im := fftTestInput(n, 23)
		if got := shared.Forward(re, im); got != n {
			t.Fatalf("step %d, n=%d: Forward returned %d", step, n, got)
		}
		want := dedicated[n]
		if i, ok := bitsDiff(re, want.re); !ok {
			t.Fatalf("step %d, n=%d: re[%d] differs from a dedicated plan", step, n, i)
		}
		if i, ok := bitsDiff(im, want.im); !ok {
			t.Fatalf("step %d, n=%d: im[%d] differs from a dedicated plan", step, n, i)
		}
		if n&(n-1) != 0 {
			continue
		}
		f, err := NewFFTPlan(n)
		if err != nil {
			t.Fatal(err)
		}
		fr, fi := fftTestInput(n, 23)
		f.Forward(fr, fi)
		if i, ok := bitsDiff(re, fr); !ok {
			t.Fatalf("step %d, n=%d: re[%d] differs from FFTPlan", step, n, i)
		}
		if i, ok := bitsDiff(im, fi); !ok {
			t.Fatalf("step %d, n=%d: im[%d] differs from FFTPlan", step, n, i)
		}
	}
}

// TestDFTPlanForwardLargeLength covers a length whose chirp index t*t exceeds
// the range of a 32-bit int (t >= 46341, so n >= 46342), where a 32-bit
// product would give the wrong phase. With a 64-bit int it fails only when the
// reduction is narrowed explicitly, for example to int32.
func TestDFTPlanForwardLargeLength(t *testing.T) {
	const n = 46349
	p, err := NewDFTPlan(n)
	if err != nil {
		t.Fatal(err)
	}
	xr, xi := fftTestInput(n, 30)
	re := append([]float64(nil), xr...)
	im := append([]float64(nil), xi...)
	if got := p.Forward(re, im); got != n {
		t.Fatalf("Forward returned %d", got)
	}
	bins := dftBins(n)
	wantRe, wantIm := compensatedDFT(xr, xi, bins)
	gotRe := make([]float64, len(bins))
	gotIm := make([]float64, len(bins))
	for b, k := range bins {
		gotRe[b], gotIm[b] = re[k], im[k]
	}
	if e, tol := relL2(gotRe, gotIm, wantRe, wantIm), dftTolerance(n, dftAccuracyFactor); e > tol {
		t.Errorf("n=%d: relL2 = %g, want <= %g", n, e, tol)
	}
}

func TestDFTPlanAllocFree(t *testing.T) {
	p, err := NewDFTPlan(4097)
	if err != nil {
		t.Fatal(err)
	}
	// Every non-power-of-two Forward retargets, because each length differs from
	// the previous non-power-of-two one. The capacity stays small because
	// AllocsPerRun runs the closure several times.
	lengths := []int{1000, 7, 4096, 4097, 3}
	bufs := make([][2][]float64, len(lengths))
	for i, n := range lengths {
		re, im := fftTestInput(n, 24)
		bufs[i] = [2][]float64{re, im}
	}
	a := testing.AllocsPerRun(5, func() {
		for _, b := range bufs {
			p.Forward(b[0], b[1])
			p.Inverse(b[0], b[1])
		}
	})
	if a != 0 {
		t.Errorf("Forward and Inverse allocated %v times per run, want 0", a)
	}
}

func TestNewDFTPlanRejectsInvalidCapacity(t *testing.T) {
	for _, c := range []int{0, -1, maxDFTCapacity + 1} {
		p, err := NewDFTPlan(c)
		if !errors.Is(err, ErrDFTCapacity) || p != nil {
			t.Errorf("NewDFTPlan(%d) = (%v, %v), want (nil, ErrDFTCapacity)", c, p, err)
		}
	}
	for _, c := range []int{1, 2, 3, 1000} {
		p, err := NewDFTPlan(c)
		if err != nil {
			t.Errorf("NewDFTPlan(%d) unexpected error: %v", c, err)
			continue
		}
		if p.Capacity() != c {
			t.Errorf("NewDFTPlan(%d).Capacity() = %d", c, p.Capacity())
		}
	}
}

// TestDFTPlanInvalidLengthIsNoOp pins the validation: an invalid call returns 0
// and leaves both slices bitwise unchanged; lengths up to the capacity succeed.
func TestDFTPlanInvalidLengthIsNoOp(t *testing.T) {
	const capacity = 100
	p, err := NewDFTPlan(capacity)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name         string
		lenRe, lenIm int
	}{
		{"empty", 0, 0},
		{"im longer than re", 10, 11},
		{"im shorter than re", 10, 4},
		{"above capacity", capacity + 1, capacity + 1},
		{"power of two above capacity", 128, 128},
	}
	for name, run := range map[string]func(re, im []float64) int{"Forward": p.Forward, "Inverse": p.Inverse} {
		for _, c := range cases {
			re, _ := fftTestInput(c.lenRe, 25)
			im, _ := fftTestInput(c.lenIm, 26)
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
		// The capacity itself, and a power of two at or above half of it.
		for _, n := range []int{capacity, 64} {
			re, im := fftTestInput(n, 27)
			if got := run(re, im); got != n {
				t.Errorf("%s n=%d returned %d, want %d", name, n, got, n)
			}
		}
	}
}

// dftPlanFootprint is the documented memory formula of DFTPlan in bytes, with
// Mc = nextPow2(2*capacity-1): the FFTPlan tables for Mc, plus 2*capacity float64
// of chirp, plus 4*Mc float64 of kernel and work buffers.
func dftPlanFootprint(capacity int) int {
	const float64Bytes = 8
	mc := nextPow2(2*capacity - 1)
	return fftPlanFootprint(mc) + 2*float64Bytes*capacity + 4*float64Bytes*mc
}

// TestDFTPlanFootprint pins the memory figures quoted in the DFTPlan docs and the
// README.
func TestDFTPlanFootprint(t *testing.T) {
	for _, c := range []int{1, 2, 3, 4097, 8193, dftTestCapacity} {
		p, err := NewDFTPlan(c)
		if err != nil {
			t.Fatal(err)
		}
		got := sliceBytes(t, reflect.ValueOf(*p))
		if want := dftPlanFootprint(c); got != want {
			t.Errorf("capacity=%d: plan holds %d bytes, formula says %d", c, got, want)
		}
	}
	documented := map[int]int{4097: 1004880, 8193: 1922384, 32769: 7689552}
	for c, want := range documented {
		if got := dftPlanFootprint(c); got != want {
			t.Errorf("formula at capacity %d = %d, documented figure is %d", c, got, want)
		}
	}
}

// dolphTables holds the per-length tables of the Dolph-Chebyshev scenario:
// cos(pi*j/m) for j in [0, 2m) and the phase factors exp(+i*pi*k*(m-1)/m) for
// k <= (m-1)/2.
type dolphTables struct {
	m                int
	cosTab           []float64
	phaseRe, phaseIm []float64
}

func newDolphTables(m int) *dolphTables {
	half := (m - 1) / 2
	d := &dolphTables{m: m, cosTab: make([]float64, 2*m), phaseRe: make([]float64, half+1), phaseIm: make([]float64, half+1)}
	for j := range d.cosTab {
		d.cosTab[j] = math.Cos(math.Pi * float64(j) / float64(m))
	}
	for k := range d.phaseRe {
		// (k*(m-1)) mod 2m in int64, so the angle stays below 2*pi.
		r := int64(k) * int64(m-1) % int64(2*m)
		s, c := math.Sincos(math.Pi * float64(r) / float64(m))
		d.phaseRe[k], d.phaseIm[k] = c, s
	}
	return d
}

// chebyshevSamples returns the spectrum samples p[k] = T_(m-1)(x0*cos(pi*k/m)) for
// k <= (m-1)/2, with x0 set so the sidelobe plateau sits attenuationDB below the
// peak. It is the construction of a Dolph-Chebyshev window by frequency sampling.
func chebyshevSamples(m int, attenuationDB float64) []float64 {
	order := float64(m - 1)
	x0 := math.Cosh(math.Acosh(math.Pow(10, attenuationDB/20)) / order)
	p := make([]float64, (m-1)/2+1)
	for k := range p {
		x := x0 * math.Cos(math.Pi*float64(k)/float64(m))
		if x > 1 {
			p[k] = math.Cosh(order * math.Acosh(x))
		} else {
			p[k] = math.Cos(order * math.Acos(x))
		}
	}
	return p
}

// direct evaluates sample i of the symmetric window by the direct cosine sum
// w[i] = p[0] + 2*sum_{k=1..K} p[k]*cos(pi*k*d/m), d = (m-1)-2*i, K = (m-1)/2.
func (d *dolphTables) direct(p []float64, i int) float64 {
	i = min(i, d.m-1-i) // the window is symmetric
	step := (d.m - 1) - 2*i
	period := 2 * d.m
	var s float64
	idx := 0
	for k := 1; k < len(p); k++ {
		idx += step
		if idx >= period {
			idx -= period
		}
		s += p[k] * d.cosTab[idx]
	}
	return p[0] + 2*s
}

// TestDFTPlanDolphChebyshevWindow runs the consumer scenario that motivated the
// plan: the Dolph-Chebyshev window of length m is the real part of a length-m DFT
// of the Chebyshev spectrum samples with a linear phase. It compares the DFTPlan
// route with the direct cosine sum at sampled indices, on odd and even m, on both
// the Bluestein and the power-of-two path, with a 1e10 dynamic range at 200 dB.
// Measured max |w - direct|/peak is at most 9.3e-15 on an Apple M5 Pro and a
// Cortex-A76 (NEON and SIMD_DISABLE=all) and 9.0e-15 on an Intel i5-7260U,
// against a bound of 1e-13.
func TestDFTPlanDolphChebyshevWindow(t *testing.T) {
	const (
		bound    = 1e-13
		fullDB   = 200
		longerM  = 4096
		spacings = 64
	)
	type dolphCase struct {
		m  int
		db float64
	}
	lengths := []int{2, 3, 8, 9, 257, 4096, 4097, 32767, 32768, 32769}
	longLengths := lengths[5:]
	cases := make([]dolphCase, 0, len(lengths)+2*len(longLengths))
	for _, m := range lengths {
		cases = append(cases, dolphCase{m, fullDB})
	}
	for _, m := range longLengths {
		cases = append(cases, dolphCase{m, 20}, dolphCase{m, 106.5})
	}
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		t.Fatal(err)
	}
	tables := map[int]*dolphTables{}
	worst := 0.0
	for _, c := range cases {
		m := c.m
		tab := tables[m]
		if tab == nil {
			tab = newDolphTables(m)
			tables[m] = tab
		}
		ps := chebyshevSamples(m, c.db)
		re := make([]float64, m)
		im := make([]float64, m)
		for k, pk := range ps {
			re[k] = pk * tab.phaseRe[k]
			im[k] = pk * tab.phaseIm[k]
		}
		if got := p.Forward(re, im); got != m {
			t.Fatalf("m=%d: Forward returned %d", m, got)
		}
		w := re
		peak, argmin := 0.0, 0
		for i := range w {
			w[i] = 2*w[i] - ps[0]
			peak = max(peak, w[i])
			if w[i] < w[argmin] {
				argmin = i
			}
			if !(w[i] > 0) || math.IsInf(w[i], 0) {
				t.Fatalf("m=%d at %g dB: w[%d] = %g, want finite and strictly positive", m, c.db, i, w[i])
			}
		}
		idx := []int{0, argmin, (m - 1) / 2, m - 1}
		if m >= longerM {
			for i := m / spacings; i < m; i += m / spacings {
				idx = append(idx, i)
			}
		} else {
			idx = append(idx, fftSampledBins(m)...)
		}
		for _, i := range idx {
			e := math.Abs(w[i]-tab.direct(ps, i)) / peak
			worst = max(worst, e)
			if e > bound {
				t.Errorf("m=%d at %g dB: |w[%d] - direct|/peak = %g, want <= %g", m, c.db, i, e, bound)
			}
		}
	}
	t.Logf("worst |w - direct|/peak = %.3g", worst)
}

func BenchmarkDFTPlanForward(b *testing.B) {
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{1000, 4096, 8191, 32767, 32768, 32769} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			xr, xi := fftTestInput(n, 28)
			re, im := make([]float64, n), make([]float64, n)
			copy(re, xr)
			copy(im, xi)
			p.Forward(re, im) // retarget outside the timed loop
			b.ReportAllocs()
			for b.Loop() {
				copy(re, xr)
				copy(im, xi)
				p.Forward(re, im)
			}
		})
	}
}

// BenchmarkDFTPlanForwardRetarget alternates two non-power-of-two lengths, so
// every call retargets the plan.
func BenchmarkDFTPlanForwardRetarget(b *testing.B) {
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		b.Fatal(err)
	}
	lengths := []int{32767, 32769}
	var bufs [2][2][]float64
	for i, n := range lengths {
		bufs[i][0], bufs[i][1] = fftTestInput(n, 29)
	}
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		p.Forward(bufs[i][0], bufs[i][1])
		i ^= 1
	}
}

// BenchmarkDFTPlanRetarget times the retarget alone, the cost a Forward call pays
// when its length differs from the plan's current non-power-of-two target.
func BenchmarkDFTPlanRetarget(b *testing.B) {
	p, err := NewDFTPlan(dftTestCapacity)
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{1000, 32767, 32769} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				p.retarget(n)
			}
		})
	}
}

func BenchmarkNewDFTPlan(b *testing.B) {
	for _, c := range []int{4097, dftTestCapacity} {
		b.Run("capacity"+strconv.Itoa(c), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := NewDFTPlan(c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
