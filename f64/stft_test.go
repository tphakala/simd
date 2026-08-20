package f64

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

// dftBin computes a single DFT bin X[k] = sum_n frame[n] * exp(-i 2pi k n / N)
// directly, as an independent reference for the FFT-based STFT.
func dftBin(frame []float64, k int) complex128 {
	n := len(frame)
	var re, im float64
	for t := range n {
		ang := -2 * math.Pi * float64(k) * float64(t) / float64(n)
		s, c := math.Sincos(ang)
		re += frame[t] * c
		im += frame[t] * s
	}
	return complex(re, im)
}

func hann(nfft int) []float64 {
	w := make([]float64, nfft)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(nfft))
	}
	return w
}

// testSignal builds a deterministic pseudo-random-ish real signal.
func testSignal(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = math.Sin(0.3*float64(i)) + 0.5*math.Cos(0.11*float64(i)+1) - 0.25*math.Sin(0.027*float64(i))
	}
	return s
}

func cmplxClose(t *testing.T, ctx string, got, want complex128, scale float64) {
	t.Helper()
	tol := 1e-9*scale + 1e-9
	if d := math.Hypot(real(got)-real(want), imag(got)-imag(want)); d > tol {
		t.Fatalf("%s: got %v want %v (|diff|=%g tol=%g)", ctx, got, want, d, tol)
	}
}

func TestNewSTFTPlanErrors(t *testing.T) {
	for _, bad := range []int{0, 1, 3, 5, 6, 7, 9, 100, 1000} {
		if _, err := NewSTFTPlan(bad); err == nil {
			t.Errorf("NewSTFTPlan(%d) = nil error, want ErrNotPowerOfTwo", bad)
		}
	}
	for _, good := range []int{2, 4, 8, 16, 1024} {
		p, err := NewSTFTPlan(good)
		if err != nil {
			t.Errorf("NewSTFTPlan(%d) unexpected error: %v", good, err)
			continue
		}
		if p.NFFT() != good || p.NumBins() != good/2+1 {
			t.Errorf("NewSTFTPlan(%d): NFFT=%d NumBins=%d", good, p.NFFT(), p.NumBins())
		}
	}
}

// TestSTFTAgainstDFT is the core correctness gate: every bin of every frame must
// match a direct DFT of the windowed frame, across nfft sizes, hops, and with or
// without a window.
func TestSTFTAgainstDFT(t *testing.T) {
	signal := testSignal(5000)
	// The size list spans both radix-4 schedule shapes: even log2(half) runs only
	// radix-4 stages (nfft 8/32/128 = 1/2/3 stages, no trailing), odd log2(half)
	// finishes with one trailing radix-2 stage (nfft 4/16/64/256/1024).
	for _, nfft := range []int{2, 4, 8, 16, 32, 64, 128, 256, 1024} {
		for _, useWin := range []bool{false, true} {
			plan, err := NewSTFTPlan(nfft)
			if err != nil {
				t.Fatal(err)
			}
			var window []float64
			if useWin {
				window = hann(nfft)
			}
			hop := max(nfft/2, 1)
			nf := plan.NumFrames(len(signal), hop, NoPad)
			dst := make([][]complex128, nf)
			for f := range dst {
				dst[f] = make([]complex128, plan.NumBins())
			}
			got := plan.STFT(dst, signal, window, hop, NoPad)
			if got != nf {
				t.Fatalf("nfft=%d: STFT wrote %d frames, want %d", nfft, got, nf)
			}

			frame := make([]float64, nfft)
			for f := range nf {
				base := f * hop
				var scale float64
				for i := range nfft {
					v := signal[base+i]
					if window != nil {
						v *= window[i]
					}
					frame[i] = v
					scale += math.Abs(v)
				}
				for k := range plan.NumBins() {
					want := dftBin(frame, k)
					ctx := fmt.Sprintf("nfft=%d win=%v frame=%d bin=%d", nfft, useWin, f, k)
					cmplxClose(t, ctx, dst[f][k], want, scale)
				}
			}
		}
	}
}

// TestSTFTPowerMatchesSTFT verifies STFTPower equals |STFT|^2 bin-for-bin.
func TestSTFTPowerMatchesSTFT(t *testing.T) {
	signal := testSignal(4096)
	plan, _ := NewSTFTPlan(512)
	window := hann(512)
	hop := 128
	nf := plan.NumFrames(len(signal), hop, NoPad)

	spec := make([][]complex128, nf)
	pow := make([][]float64, nf)
	for f := range spec {
		spec[f] = make([]complex128, plan.NumBins())
		pow[f] = make([]float64, plan.NumBins())
	}
	plan.STFT(spec, signal, window, hop, NoPad)
	plan.STFTPower(pow, signal, window, hop, NoPad)

	for f := range nf {
		for k := range plan.NumBins() {
			want := real(spec[f][k])*real(spec[f][k]) + imag(spec[f][k])*imag(spec[f][k])
			if d := math.Abs(pow[f][k] - want); d > 1e-9*(1+want) {
				t.Fatalf("STFTPower[%d][%d] = %v, want |X|^2 = %v", f, k, pow[f][k], want)
			}
		}
	}
}

// TestSTFTPureTone checks a single-bin cosine concentrates its energy in that
// bin and that DC/Nyquist come out (numerically) real.
func TestSTFTPureTone(t *testing.T) {
	const nfft = 64
	plan, _ := NewSTFTPlan(nfft)
	k0 := 5
	signal := make([]float64, nfft)
	for n := range signal {
		signal[n] = math.Cos(2 * math.Pi * float64(k0) * float64(n) / float64(nfft))
	}
	dst := [][]complex128{make([]complex128, plan.NumBins())}
	plan.STFT(dst, signal, nil, nfft, NoPad)

	mag := func(c complex128) float64 { return math.Hypot(real(c), imag(c)) }
	// Bin k0 should hold ~nfft/2; every other bin should be ~0.
	if got := mag(dst[0][k0]); math.Abs(got-float64(nfft)/2) > 1e-7 {
		t.Errorf("tone bin %d magnitude = %v, want ~%v", k0, got, float64(nfft)/2)
	}
	for k := range plan.NumBins() {
		if k == k0 {
			continue
		}
		if got := mag(dst[0][k]); got > 1e-7 {
			t.Errorf("non-tone bin %d magnitude = %v, want ~0", k, got)
		}
	}
	// DC and Nyquist bins of a real signal are real.
	if math.Abs(imag(dst[0][0])) > 1e-9 {
		t.Errorf("DC bin not real: %v", dst[0][0])
	}
	if math.Abs(imag(dst[0][plan.NumBins()-1])) > 1e-9 {
		t.Errorf("Nyquist bin not real: %v", dst[0][plan.NumBins()-1])
	}
}

// TestSTFTFraming checks frame counting and the no-padding (center=false)
// convention: frame f starts at f*hop.
func TestSTFTFraming(t *testing.T) {
	plan, _ := NewSTFTPlan(8)
	signal := make([]float64, 20)
	for i := range signal {
		signal[i] = float64(i)
	}
	hop := 4
	// frames at offsets 0,4,8,12 fit (need 8 samples): 12+8=20 ok, 16+8=24 no.
	wantFrames := 4
	if got := plan.NumFrames(len(signal), hop, NoPad); got != wantFrames {
		t.Fatalf("numFrames = %d, want %d", got, wantFrames)
	}
	dst := make([][]complex128, wantFrames)
	for f := range dst {
		dst[f] = make([]complex128, plan.NumBins())
	}
	if n := plan.STFT(dst, signal, nil, hop, NoPad); n != wantFrames {
		t.Fatalf("STFT frames = %d, want %d", n, wantFrames)
	}
	// DC bin of frame f is the sum of signal[f*hop : f*hop+8].
	for f := range wantFrames {
		var sum float64
		for i := range 8 {
			sum += signal[f*hop+i]
		}
		if math.Abs(real(dst[f][0])-sum) > 1e-9 {
			t.Errorf("frame %d DC = %v, want %v", f, real(dst[f][0]), sum)
		}
	}
}

// TestSTFTClamps verifies dst shorter than the frame count, and rows shorter than
// NumBins, are handled without panic.
func TestSTFTClamps(t *testing.T) {
	plan, _ := NewSTFTPlan(16)
	signal := testSignal(200)
	hop := 8
	full := plan.NumFrames(len(signal), hop, NoPad)

	// Fewer rows than frames: only len(dst) frames written.
	short := make([][]complex128, full-2)
	for f := range short {
		short[f] = make([]complex128, plan.NumBins())
	}
	if n := plan.STFT(short, signal, nil, hop, NoPad); n != full-2 {
		t.Errorf("clamped frames = %d, want %d", n, full-2)
	}

	// Rows shorter than NumBins: only the available bins written, no panic.
	rows := make([][]complex128, 1)
	rows[0] = make([]complex128, 3)
	if n := plan.STFT(rows, signal, nil, hop, NoPad); n != 1 {
		t.Errorf("partial-row frames = %d, want 1", n)
	}

	// Rows longer than NumBins: exactly NumBins bins written, the rest untouched,
	// and the written bins identical to a NumBins-wide row (same path).
	bins := plan.NumBins()
	const sentinel = complex(-7, 7)
	long := [][]complex128{make([]complex128, bins+3)}
	for k := range long[0] {
		long[0][k] = sentinel
	}
	exact := [][]complex128{make([]complex128, bins)}
	if n := plan.STFT(long, signal, nil, hop, NoPad); n != 1 {
		t.Errorf("long-row frames = %d, want 1", n)
	}
	plan.STFT(exact, signal, nil, hop, NoPad)
	for k := range bins {
		if long[0][k] != exact[0][k] {
			t.Fatalf("long row bin %d = %v, want %v", k, long[0][k], exact[0][k])
		}
	}
	for k := bins; k < len(long[0]); k++ {
		if long[0][k] != sentinel {
			t.Fatalf("long row bin %d beyond NumBins was written: %v", k, long[0][k])
		}
	}
	longPow := [][]float64{make([]float64, bins+3)}
	for k := range longPow[0] {
		longPow[0][k] = -7
	}
	if n := plan.STFTPower(longPow, signal, nil, hop, NoPad); n != 1 {
		t.Errorf("long-row power frames = %d, want 1", n)
	}
	exactPow := [][]float64{make([]float64, bins)}
	plan.STFTPower(exactPow, signal, nil, hop, NoPad)
	for k := range bins {
		if longPow[0][k] != exactPow[0][k] {
			t.Fatalf("long power row bin %d = %v, want %v", k, longPow[0][k], exactPow[0][k])
		}
	}
	for k := bins; k < len(longPow[0]); k++ {
		if longPow[0][k] != -7 {
			t.Fatalf("long power row bin %d beyond NumBins was written: %v", k, longPow[0][k])
		}
	}
}

// TestSTFTShortRowsMatchFull pins the two unravel paths against each other: a
// full-width row (NumBins) takes the vector RealFFTUnpack / RealFFTPower
// unravel, a shorter row keeps the per-bin scalar unravel. The bins a short row
// does write must agree with the same bins of the full row, for STFT and
// STFTPower, across row lengths on both sides of the split and across nfft sizes
// that exercise the vector kernels' full blocks and scalar tails. It also proves
// the full-width row really took the vector path: the plan's unpack scratch is
// poisoned with NaN before the call and the row's interior bins must be the
// bits that scratch holds afterwards (only RealFFTUnpack writes it).
func TestSTFTShortRowsMatchFull(t *testing.T) {
	signal := testSignal(3000)
	nan := math.NaN()
	for _, nfft := range []int{2, 4, 16, 64, 512} {
		plan, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatal(err)
		}
		window := hann(nfft)
		hop := max(nfft/4, 1)
		bins := plan.NumBins()
		half := bins - 1
		nf := plan.NumFrames(len(signal), hop, NoPad)

		full := make([][]complex128, nf)
		fullPow := make([][]float64, nf)
		for f := range nf {
			full[f] = make([]complex128, bins)
			fullPow[f] = make([]float64, bins)
		}
		for k := range plan.outRe {
			plan.outRe[k], plan.outIm[k] = nan, nan
		}
		plan.STFT(full, signal, window, hop, NoPad)
		// Positive control: the last full row's interior bins are exactly the
		// unpack scratch the vector path wrote (NaN would mean it never ran).
		for k := 1; k < half; k++ {
			if got, wr, wi := full[nf-1][k], plan.outRe[k], plan.outIm[k]; math.IsNaN(wr) || math.IsNaN(wi) ||
				math.Float64bits(real(got)) != math.Float64bits(wr) || math.Float64bits(imag(got)) != math.Float64bits(wi) {
				t.Fatalf("nfft=%d bin=%d: full row %v is not the vector unpack scratch (%v,%v)", nfft, k, got, wr, wi)
			}
		}
		// The rounding gap between the FMA vector unpack and the scalar unravel
		// scales with the packed spectrum's magnitude, not with the bin being
		// compared (a near-null bin is a cancellation of large terms), so the
		// tolerance is relative to the row's largest bin; 1e-12 is ~1e4 ulps of
		// float64 against a measured gap of a few ulps.
		rowMax := 0.0
		for k := range bins {
			rowMax = max(rowMax, math.Hypot(real(full[nf-1][k]), imag(full[nf-1][k])))
		}
		tol := 1e-12 * (1 + rowMax)
		closeTo := func(ctx string, got, want complex128) {
			t.Helper()
			if d := math.Hypot(real(got)-real(want), imag(got)-imag(want)); d > tol {
				t.Fatalf("%s: got %v want %v (|diff|=%g tol=%g)", ctx, got, want, d, tol)
			}
		}
		// The plan scratch still holds the last frame's half-size spectrum, so
		// every bin of the last full row, DC and Nyquist included, can be checked
		// against the scalar per-bin unravel the short-row path uses.
		for k := range bins {
			xr, xi := plan.unravelBin(k)
			closeTo(fmt.Sprintf("nfft=%d last frame bin=%d (vector vs scalar unravel)", nfft, k), full[nf-1][k], complex(xr, xi))
		}
		plan.STFTPower(fullPow, signal, window, hop, NoPad)
		powTol := 1e-12 * (1 + rowMax*rowMax)
		for k := range bins {
			xr, xi := plan.unravelBin(k)
			wp := xr*xr + xi*xi
			if d := math.Abs(fullPow[nf-1][k] - wp); d > powTol {
				t.Fatalf("nfft=%d last frame bin=%d: vector power %v, scalar power %v", nfft, k, fullPow[nf-1][k], wp)
			}
		}

		for _, rowLen := range []int{1, 2, bins / 2, bins - 1, bins} {
			short := make([][]complex128, nf)
			shortPow := make([][]float64, nf)
			for f := range nf {
				short[f] = make([]complex128, rowLen)
				shortPow[f] = make([]float64, rowLen)
			}
			plan.STFT(short, signal, window, hop, NoPad)
			plan.STFTPower(shortPow, signal, window, hop, NoPad)
			for f := range nf {
				for k := range rowLen {
					ctx := fmt.Sprintf("nfft=%d rowLen=%d frame=%d bin=%d", nfft, rowLen, f, k)
					closeTo(ctx, short[f][k], full[f][k])
					if d := math.Abs(shortPow[f][k] - fullPow[f][k]); d > powTol {
						t.Fatalf("%s: short-row power %v, full-row power %v", ctx, shortPow[f][k], fullPow[f][k])
					}
				}
			}
		}
	}
}

// TestSTFTWindowReuse pins that every public call applies exactly the window it
// was given: the plan splits the window into per-call scratch for the vector
// pack, so a stale split would leak a previous call's window into the next. One
// shared plan runs a sequence of calls that changes the window on EVERY call
// (Hann to ramp directly, to and from rectangular, a window longer than nfft)
// and rotates through STFT, STFTPowerInto and STFTPower, and each call is
// compared bit for bit against a plan created for that one call only (a fresh
// plan replaying the same sequence would carry the same stale split and hide
// the defect). A window longer than nfft must behave as its first nfft samples.
func TestSTFTWindowReuse(t *testing.T) {
	const nfft = 64
	signal := testSignal(2000)
	hop := 16
	hannWin := hann(nfft)
	ramp := make([]float64, nfft)
	for i := range ramp {
		ramp[i] = float64(i+1) / float64(nfft)
	}
	hannLong := append(append([]float64(nil), hannWin...), 9, 9, 9, 9, 9)
	shared, _ := NewSTFTPlan(nfft)
	nf := shared.NumFrames(len(signal), hop, PadReflect)
	bins := shared.NumBins()
	newSpec := func() [][]complex128 {
		spec := make([][]complex128, nf)
		for f := range spec {
			spec[f] = make([]complex128, bins)
		}
		return spec
	}
	newPow := func() [][]float64 {
		pow := make([][]float64, nf)
		for f := range pow {
			pow[f] = make([]float64, bins)
		}
		return pow
	}
	windows := [][]float64{hannWin, ramp, hannWin, ramp, hannWin, ramp, nil, hannLong, ramp}
	for i, window := range windows {
		fresh, _ := NewSTFTPlan(nfft)
		switch i % 3 {
		case 0:
			got, want := newSpec(), newSpec()
			shared.STFT(got, signal, window, hop, PadReflect)
			fresh.STFT(want, signal, window, hop, PadReflect)
			for f := range nf {
				for k := range bins {
					if got[f][k] != want[f][k] {
						t.Fatalf("call %d (STFT) frame %d bin %d: shared plan %v, fresh plan %v", i, f, k, got[f][k], want[f][k])
					}
				}
			}
		case 1:
			got, want := make([]float64, nf*bins), make([]float64, nf*bins)
			shared.STFTPowerInto(got, signal, window, hop, PadReflect)
			fresh.STFTPowerInto(want, signal, window, hop, PadReflect)
			for j := range got {
				if got[j] != want[j] {
					t.Fatalf("call %d (STFTPowerInto) flat index %d: shared plan %v, fresh plan %v", i, j, got[j], want[j])
				}
			}
		case 2:
			got, want := newPow(), newPow()
			shared.STFTPower(got, signal, window, hop, PadReflect)
			fresh.STFTPower(want, signal, window, hop, PadReflect)
			for f := range nf {
				for k := range bins {
					if got[f][k] != want[f][k] {
						t.Fatalf("call %d (STFTPower) frame %d bin %d: shared plan %v, fresh plan %v", i, f, k, got[f][k], want[f][k])
					}
				}
			}
		}
	}
	// A window longer than nfft is its first nfft samples: bit-identical to Hann.
	longPlan, _ := NewSTFTPlan(nfft)
	hannPlan, _ := NewSTFTPlan(nfft)
	got, want := newSpec(), newSpec()
	longPlan.STFT(got, signal, hannLong, hop, PadReflect)
	hannPlan.STFT(want, signal, hannWin, hop, PadReflect)
	for f := range nf {
		for k := range bins {
			if got[f][k] != want[f][k] {
				t.Fatalf("long window frame %d bin %d: %v, want Hann-prefix %v", f, k, got[f][k], want[f][k])
			}
		}
	}
}

func TestSTFTAllocFree(t *testing.T) {
	plan, _ := NewSTFTPlan(512)
	signal := testSignal(8192)
	window := hann(512)
	hop := 128
	nf := plan.NumFrames(len(signal), hop, NoPad)
	spec := make([][]complex128, nf)
	pow := make([][]float64, nf)
	for f := range spec {
		spec[f] = make([]complex128, plan.NumBins())
		pow[f] = make([]float64, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(spec, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("STFT allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(pow, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("STFTPower allocated %v times per run, want 0", a)
	}

	// Centered framing and the flat output must also be allocation-free.
	cf := plan.NumFrames(len(signal), hop, PadReflect)
	cpow := make([][]float64, cf)
	for f := range cpow {
		cpow[f] = make([]float64, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(cpow, signal, window, hop, PadReflect) }); a != 0 {
		t.Errorf("centered STFTPower allocated %v times per run, want 0", a)
	}
	flat := make([]float64, plan.NumFrames(len(signal), hop, PadZero)*plan.NumBins())
	if a := testing.AllocsPerRun(5, func() { plan.STFTPowerInto(flat, signal, window, hop, PadZero) }); a != 0 {
		t.Errorf("STFTPowerInto allocated %v times per run, want 0", a)
	}

	// The complex STFT with centered framing uses the same packFrameAt edge path
	// and must also be allocation-free.
	cspec := make([][]complex128, cf)
	for f := range cspec {
		cspec[f] = make([]complex128, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(cspec, signal, window, hop, PadReflect) }); a != 0 {
		t.Errorf("centered STFT allocated %v times per run, want 0", a)
	}

	// Rows shorter than NumBins take the scalar unravel branch, which must also be
	// allocation-free.
	sspec := make([][]complex128, nf)
	spow := make([][]float64, nf)
	for f := range sspec {
		sspec[f] = make([]complex128, 3)
		spow[f] = make([]float64, 3)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(sspec, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("short-row STFT allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(spow, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("short-row STFTPower allocated %v times per run, want 0", a)
	}
}

// FuzzSTFT is a differential fuzz target: every STFT bin must match a direct DFT
// of the windowed frame, across fuzzed signal contents, nfft, hop, and window
// choice. Inputs are bounded to [-1, 1] so the DFT bin magnitudes stay
// well-conditioned for the epsilon-scaled tolerance. Seeds run under plain
// `go test`; `go test -fuzz=FuzzSTFT` widens the space.
func FuzzSTFT(f *testing.F) {
	f.Add(make([]byte, 256), uint8(3), uint8(7), false, uint8(0))
	f.Add(make([]byte, 600), uint8(5), uint8(3), true, uint8(1))
	f.Add(make([]byte, 600), uint8(4), uint8(2), true, uint8(2))

	f.Fuzz(func(t *testing.T, raw []byte, nfftSel, hopSel uint8, useWin bool, padSel uint8) {
		// nfft in {4, 8, 16, 32, 64}; keep it small so the O(n^2) DFT is cheap.
		nfft := 1 << (2 + int(nfftSel)%5)
		samples := len(raw) / 8
		if samples < nfft {
			return
		}
		signal := make([]float64, samples)
		for i := range signal {
			signal[i] = float64(int64(binU64(raw[i*8:]))) / 9223372036854775808.0
		}
		plan, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatal(err)
		}
		var window []float64
		if useWin {
			window = hann(nfft)
		}
		hop := 1 + int(hopSel)%nfft
		pad := []PadMode{NoPad, PadZero, PadReflect}[int(padSel)%3]
		nf := plan.NumFrames(samples, hop, pad)
		if nf == 0 {
			return
		}
		dst := make([][]complex128, nf)
		for i := range dst {
			dst[i] = make([]complex128, plan.NumBins())
		}
		plan.STFT(dst, signal, window, hop, pad)

		// Compare against the independent reference for every pad mode.
		ref := stftRef(signal, window, nfft, hop, pad)
		if len(ref) != nf {
			t.Fatalf("nfft=%d hop=%d pad=%v: ref frames %d != NumFrames %d", nfft, hop, pad, len(ref), nf)
		}
		off := 0
		if pad != NoPad {
			off = nfft / 2
		}
		for fr := range nf {
			// scale = L1 norm of the windowed (padded) frame, for the tolerance.
			var scale float64
			base := fr*hop - off
			for i := range nfft {
				v := refSampleAt(signal, base+i, pad)
				if window != nil {
					v *= window[i]
				}
				scale += math.Abs(v)
			}
			for k := range plan.NumBins() {
				got, want := dst[fr][k], ref[fr][k]
				tol := 1e-9*scale + 1e-9
				if d := math.Hypot(real(got)-real(want), imag(got)-imag(want)); d > tol {
					t.Fatalf("nfft=%d hop=%d pad=%v frame=%d bin=%d: got %v want %v |diff|=%g", nfft, hop, pad, fr, k, got, want, d)
				}
			}
		}

		// The flat power output must equal |STFT|^2 bin-for-bin.
		bins := plan.NumBins()
		flat := make([]float64, nf*bins)
		plan.STFTPowerInto(flat, signal, window, hop, pad)
		for fr := range nf {
			for k := range bins {
				want := real(dst[fr][k])*real(dst[fr][k]) + imag(dst[fr][k])*imag(dst[fr][k])
				if d := math.Abs(flat[fr*bins+k] - want); d > 1e-9*(1+want)+1e-12 {
					t.Fatalf("nfft=%d pad=%v frame=%d bin=%d: flat power %v want %v", nfft, pad, fr, k, flat[fr*bins+k], want)
				}
			}
		}
	})
}

func binU64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func BenchmarkSTFT(b *testing.B) {
	const nfft = 1024
	plan, _ := NewSTFTPlan(nfft)
	window := hann(nfft)
	signal := testSignal(48000) // ~1s of 48 kHz audio
	hop := 256
	nf := plan.NumFrames(len(signal), hop, NoPad)
	dst := make([][]complex128, nf)
	for f := range dst {
		dst[f] = make([]complex128, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFT(dst, signal, window, hop, NoPad)
	}
}

func BenchmarkSTFTPower(b *testing.B) {
	const nfft = 1024
	plan, _ := NewSTFTPlan(nfft)
	window := hann(nfft)
	signal := testSignal(48000)
	hop := 256
	nf := plan.NumFrames(len(signal), hop, NoPad)
	dst := make([][]float64, nf)
	for f := range dst {
		dst[f] = make([]float64, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFTPower(dst, signal, window, hop, NoPad)
	}
}

func TestNumFrames(t *testing.T) {
	p, _ := NewSTFTPlan(8)
	cases := []struct {
		n, hop int
		pad    PadMode
		want   int
	}{
		{7, 4, NoPad, 0},       // shorter than nfft
		{8, 4, NoPad, 1},       // exactly one frame
		{16, 4, NoPad, 3},      // 1 + (16-8)/4
		{0, 4, PadZero, 0},     // empty signal
		{8, 4, PadZero, 3},     // 1 + 8/4
		{16, 4, PadReflect, 5}, // 1 + 16/4
		{10, 0, NoPad, 0},      // hop <= 0
	}
	for _, c := range cases {
		if got := p.NumFrames(c.n, c.hop, c.pad); got != c.want {
			t.Errorf("NumFrames(%d,%d,%v)=%d want %d", c.n, c.hop, c.pad, got, c.want)
		}
	}
}

func TestReflectIndex(t *testing.T) {
	// n=4: ...3 2 1 |0 1 2 3| 2 1 0 1... verified against numpy np.pad reflect.
	want := []int{3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1}
	for off, w := range want {
		idx := off - 3 // idx runs -3..7
		if got := reflectIndex(idx, 4); got != w {
			t.Errorf("reflectIndex(%d,4)=%d want %d", idx, got, w)
		}
	}
	if reflectIndex(-5, 1) != 0 || reflectIndex(3, 1) != 0 {
		t.Error("reflectIndex with n=1 must map every index to 0")
	}
}

// refReflectIndex reimplements numpy "reflect" index folding locally, kept
// deliberately independent of the production reflectIndex so the centered
// reference and the fuzz target would catch a regression in either one (rather
// than sharing the same mapping bug).
func refReflectIndex(idx, n int) int {
	if n == 1 {
		return 0
	}
	period := (n - 1) << 1
	m := idx % period
	if m < 0 {
		m += period
	}
	if m < n {
		return m
	}
	return period - m
}

// refSampleAt and stftRef independently re-implement centering, windowing, and
// the DFT (via dftBin), as a cross-check on the FFT-based centered STFT. They are
// deliberately a separate implementation from the package's sampleAt/packFrameAt
// so a bug in one does not mask a bug in the other.
func refSampleAt(signal []float64, idx int, pad PadMode) float64 {
	if idx >= 0 && idx < len(signal) {
		return signal[idx]
	}
	if pad == PadReflect {
		return signal[refReflectIndex(idx, len(signal))]
	}
	return 0
}

func stftRef(signal, window []float64, nfft, hop int, pad PadMode) [][]complex128 {
	off := 0
	if pad != NoPad {
		off = nfft / 2
	}
	var nf int
	switch {
	case pad == NoPad:
		if len(signal) >= nfft {
			nf = 1 + (len(signal)-nfft)/hop
		}
	case len(signal) > 0:
		nf = 1 + len(signal)/hop
	}
	out := make([][]complex128, nf)
	frame := make([]float64, nfft)
	for f := range out {
		base := f*hop - off
		for i := range frame {
			s := refSampleAt(signal, base+i, pad)
			if window != nil {
				s *= window[i]
			}
			frame[i] = s
		}
		row := make([]complex128, nfft/2+1)
		for k := range row {
			row[k] = dftBin(frame, k)
		}
		out[f] = row
	}
	return out
}

// TestSTFTCenteredAgainstRef gates the centered/padded paths: every bin of every
// frame must match the independent reference across nfft, hop, pad mode, and
// window choice.
func TestSTFTCenteredAgainstRef(t *testing.T) {
	for _, nfft := range []int{16, 64, 512, 1024} {
		for _, hop := range []int{nfft / 4, nfft / 2, nfft} {
			for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
				for _, useWin := range []bool{false, true} {
					p, _ := NewSTFTPlan(nfft)
					sig := testSignal(4*nfft + 7)
					var win []float64
					if useWin {
						win = hann(nfft)
					}
					nf := p.NumFrames(len(sig), hop, pad)
					ref := stftRef(sig, win, nfft, hop, pad)
					if len(ref) != nf {
						t.Fatalf("nfft=%d hop=%d pad=%v: ref frames %d != NumFrames %d", nfft, hop, pad, len(ref), nf)
					}
					dst := make([][]complex128, nf)
					for f := range dst {
						dst[f] = make([]complex128, p.NumBins())
					}
					if got := p.STFT(dst, sig, win, hop, pad); got != nf {
						t.Fatalf("nfft=%d hop=%d pad=%v: STFT wrote %d frames want %d", nfft, hop, pad, got, nf)
					}
					for f := range dst {
						for k := range dst[f] {
							ctx := fmt.Sprintf("nfft=%d hop=%d pad=%v win=%v f=%d k=%d", nfft, hop, pad, useWin, f, k)
							cmplxClose(t, ctx, dst[f][k], ref[f][k], float64(nfft))
						}
					}
				}
			}
		}
	}
}

// TestSTFTPowerInto checks the flat output equals the 2D STFTPower flattened, for
// every pad mode, and that a short flat dst floors to the frames that fit.
func TestSTFTPowerInto(t *testing.T) {
	for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
		nfft, hop := 256, 192
		p, _ := NewSTFTPlan(nfft)
		sig := testSignal(2000)
		win := hann(nfft)
		bins := p.NumBins()
		nf := p.NumFrames(len(sig), hop, pad)

		ref := make([][]float64, nf)
		for f := range ref {
			ref[f] = make([]float64, bins)
		}
		if got := p.STFTPower(ref, sig, win, hop, pad); got != nf {
			t.Fatalf("pad=%v: STFTPower wrote %d frames want %d", pad, got, nf)
		}
		flat := make([]float64, nf*bins)
		if got := p.STFTPowerInto(flat, sig, win, hop, pad); got != nf {
			t.Fatalf("pad=%v: STFTPowerInto wrote %d frames want %d", pad, got, nf)
		}
		for f := range ref {
			for k := range ref[f] {
				if d := math.Abs(flat[f*bins+k] - ref[f][k]); d > 1e-12 {
					t.Fatalf("pad=%v f=%d k=%d: flat=%g ref=%g", pad, f, k, flat[f*bins+k], ref[f][k])
				}
			}
		}
	}

	// A flat dst with room for fewer whole frames floors to what fits.
	p, _ := NewSTFTPlan(8)
	sig := testSignal(64)
	bins := p.NumBins()
	short := make([]float64, 2*bins+1) // 2 frames + 1 spare slot
	if got := p.STFTPowerInto(short, sig, nil, 4, NoPad); got != 2 {
		t.Fatalf("short flat dst: got %d frames want 2", got)
	}
}

//go:embed testdata/stft_librosa_golden.json
var librosaGoldenJSON []byte

// TestSTFTLibrosaParity pins the output convention against a golden vector
// generated by real librosa (see testdata/gen_stft_golden.py). This is the
// acceptance check that a model trained on librosa features accepts simd output.
// The golden is embedded (not read from disk) so the test runs from any working
// directory, including the cross-arch "copy the test binary and run it" flow. The
// signal and window are regenerated here with the same deterministic formulas the
// generator used, so only librosa's power output is pinned in the data file.
func TestSTFTLibrosaParity(t *testing.T) {
	var g struct {
		LibrosaVersion string `json:"librosa_version"`
		NFFT           int    `json:"nfft"`
		Hop            int    `json:"hop"`
		N              int    `json:"n"`
		Cases          []struct {
			GoPad  string    `json:"go_pad"`
			Frames int       `json:"frames"`
			Bins   int       `json:"bins"`
			Power  []float64 `json:"power"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(librosaGoldenJSON, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	t.Logf("golden generated by librosa %s (nfft=%d hop=%d)", g.LibrosaVersion, g.NFFT, g.Hop)
	signal := testSignal(g.N) // same formula the generator fed to librosa
	window := hann(g.NFFT)    // periodic Hann, same as the generator
	p, _ := NewSTFTPlan(g.NFFT)
	padOf := map[string]PadMode{"PadZero": PadZero, "PadReflect": PadReflect}
	for _, c := range g.Cases {
		pad, ok := padOf[c.GoPad]
		if !ok {
			t.Fatalf("unknown go_pad %q in golden", c.GoPad)
		}
		if nf := p.NumFrames(len(signal), g.Hop, pad); nf != c.Frames {
			t.Fatalf("%s: NumFrames=%d but librosa produced %d frames", c.GoPad, nf, c.Frames)
		}
		if c.Bins != p.NumBins() {
			t.Fatalf("%s: golden bins=%d but NumBins=%d", c.GoPad, c.Bins, p.NumBins())
		}
		flat := make([]float64, c.Frames*c.Bins)
		p.STFTPowerInto(flat, signal, window, g.Hop, pad)
		var maxRel float64
		for i := range flat {
			ref := c.Power[i]
			rel := math.Abs(flat[i]-ref) / (math.Abs(ref) + 1e-12)
			if rel > maxRel {
				maxRel = rel
			}
		}
		// librosa uses pocketfft and we use a radix-4 rfft, so the bins differ at
		// the float64 algorithm-noise level (~5e-8 relative; squaring to power
		// roughly doubles the amplitude error). A convention error (wrong
		// centering, window, or normalization) would be orders of magnitude
		// larger, so 1e-6 cleanly separates "matches librosa" from "wrong".
		if maxRel > 1e-6 {
			t.Errorf("%s: max relative error %g exceeds 1e-6 vs librosa", c.GoPad, maxRel)
		}
	}
}

// TestSTFTGuards covers the zero-frame and short-window (treated as rectangular)
// fallbacks shared by all three output methods.
func TestSTFTGuards(t *testing.T) {
	p, _ := NewSTFTPlan(16)
	bins := p.NumBins()

	// Signal shorter than nfft (NoPad) => zero frames, no writes, no panic.
	tiny := testSignal(8)
	if n := p.STFT(make([][]complex128, 4), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFT short signal: got %d frames want 0", n)
	}
	if n := p.STFTPower(make([][]float64, 4), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFTPower short signal: got %d frames want 0", n)
	}
	if n := p.STFTPowerInto(make([]float64, 4*bins), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFTPowerInto short signal: got %d frames want 0", n)
	}

	// A window shorter than nfft is treated as rectangular (identical to nil).
	sig := testSignal(200)
	hop := 8
	shortWin := hann(8) // len 8 < nfft 16
	nf := p.NumFrames(len(sig), hop, NoPad)

	specShort := make([][]complex128, nf)
	specNil := make([][]complex128, nf)
	for f := range specShort {
		specShort[f] = make([]complex128, bins)
		specNil[f] = make([]complex128, bins)
	}
	p.STFT(specShort, sig, shortWin, hop, NoPad)
	p.STFT(specNil, sig, nil, hop, NoPad)
	for f := range specShort {
		for k := range specShort[f] {
			if specShort[f][k] != specNil[f][k] {
				t.Fatalf("short window not treated as rectangular at f=%d k=%d", f, k)
			}
		}
	}

	powShort := make([][]float64, nf)
	for f := range powShort {
		powShort[f] = make([]float64, bins)
	}
	p.STFTPower(powShort, sig, shortWin, hop, NoPad)
	flatShort := make([]float64, nf*bins)
	p.STFTPowerInto(flatShort, sig, shortWin, hop, NoPad)
	for f := range powShort {
		for k := range powShort[f] {
			if powShort[f][k] != flatShort[f*bins+k] {
				t.Fatalf("STFTPower/STFTPowerInto short-window mismatch at f=%d k=%d", f, k)
			}
		}
	}
}

// TestSTFTStageTwiddles pins the per-stage contiguous twiddle layout fftHalf
// slices for ButterflyComplexStage4 (tw1/tw2) and the trailing
// ButterflyComplexStage (issues #205 and #243).
// Stage m in {2,4,...,half} must hold span = m/2 factors W_m^j = exp(-i*2*pi*j/m)
// at offset span-1, and the tables must total exactly half-1 entries.
func TestSTFTStageTwiddles(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 256, 1024} {
		p, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatalf("nfft=%d: NewSTFTPlan: %v", nfft, err)
		}
		half := nfft >> 1
		wantLen := max(half-1, 0)
		if len(p.stageTwRe) != wantLen || len(p.stageTwIm) != wantLen {
			t.Fatalf("nfft=%d: twiddle table len = (%d,%d), want %d",
				nfft, len(p.stageTwRe), len(p.stageTwIm), wantLen)
		}
		for m := 2; m <= half; m <<= 1 {
			span := m >> 1
			off := span - 1
			for j := range span {
				ang := 2 * math.Pi * float64(j) / float64(m)
				s, c := math.Sincos(ang)
				if d := math.Abs(p.stageTwRe[off+j] - c); d > 1e-15 {
					t.Errorf("nfft=%d m=%d j=%d: stageTwRe=%g want %g", nfft, m, j, p.stageTwRe[off+j], c)
				}
				if d := math.Abs(p.stageTwIm[off+j] - (-s)); d > 1e-15 {
					t.Errorf("nfft=%d m=%d j=%d: stageTwIm=%g want %g", nfft, m, j, p.stageTwIm[off+j], -s)
				}
			}
		}
	}
}

// TestSTFTStage4Tw3Twiddles pins the layout of the radix-4 core's third twiddle
// table, the one fftHalf cannot slice from the radix-2 tables: the radix-4 stage
// with span s in {1,4,16,...} (while 4*s <= half) keeps s factors
// w^(3j) = (cos(2*pi*3j/(4s)), -sin(2*pi*3j/(4s))) at offset (s-1)/3, and the
// table totals (4^stages - 1)/3 entries, zero when half < 4. A wrong offset,
// length or power would fail here, localized, rather than only as a spectrum
// mismatch downstream. Exact bits: NewSTFTPlan stores cos/-sin of this same
// float64 sincos.
func TestSTFTStage4Tw3Twiddles(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 32, 128, 256, 1024} {
		p, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatalf("nfft=%d: NewSTFTPlan: %v", nfft, err)
		}
		half := nfft >> 1
		wantLen := 0
		for s := 1; 4*s <= half; s *= 4 {
			wantLen += s
		}
		if len(p.stage4Tw3Re) != wantLen || len(p.stage4Tw3Im) != wantLen {
			t.Fatalf("nfft=%d: stage4Tw3 table len = (%d,%d), want %d",
				nfft, len(p.stage4Tw3Re), len(p.stage4Tw3Im), wantLen)
		}
		off := 0
		for s := 1; 4*s <= half; s *= 4 {
			if off != (s-1)/3 {
				t.Fatalf("nfft=%d s=%d: running offset %d, want (s-1)/3 = %d", nfft, s, off, (s-1)/3)
			}
			for j := range s {
				ang := 2 * math.Pi * float64(3*j) / float64(4*s)
				sin, cos := math.Sincos(ang)
				if math.Float64bits(p.stage4Tw3Re[off+j]) != math.Float64bits(cos) {
					t.Errorf("nfft=%d s=%d j=%d: stage4Tw3Re=%g want %g", nfft, s, j, p.stage4Tw3Re[off+j], cos)
				}
				if math.Float64bits(p.stage4Tw3Im[off+j]) != math.Float64bits(-sin) {
					t.Errorf("nfft=%d s=%d j=%d: stage4Tw3Im=%g want %g", nfft, s, j, p.stage4Tw3Im[off+j], -sin)
				}
			}
			off += s
		}
	}
}

// stftTol is the absolute tolerance for a float64 rfft-based value at the given
// magnitude scale: the transform error grows with log2(nfft) in units of the
// float64 epsilon, plus a small floor for near-zero results.
func stftTol(nfft int, scale float64) float64 {
	const eps64 = 2.220446049250313e-16
	logN := math.Log2(float64(nfft))
	return 8*logN*eps64*scale + 1e-11
}

// TestRFFTMatchesSTFT pins RFFT to the batched transform: the single-frame entry
// point on frame f of a NoPad STFT must reproduce that row exactly (same code
// path, so bit-for-bit).
func TestRFFTMatchesSTFT(t *testing.T) {
	const nfft, hop = 256, 64
	p, _ := NewSTFTPlan(nfft)
	signal := testSignal(2048)
	window := hann(nfft)
	frames := p.NumFrames(len(signal), hop, NoPad)
	spec := make([][]complex128, frames)
	for f := range spec {
		spec[f] = make([]complex128, p.NumBins())
	}
	p.STFT(spec, signal, window, hop, NoPad)
	row := make([]complex128, p.NumBins())
	for f := range frames {
		if n := p.RFFT(row, signal[f*hop:f*hop+nfft], window); n != p.NumBins() {
			t.Fatalf("frame %d: RFFT wrote %d bins, want %d", f, n, p.NumBins())
		}
		for k := range row {
			if row[k] != spec[f][k] {
				t.Fatalf("frame %d bin %d: RFFT %v != STFT %v", f, k, row[k], spec[f][k])
			}
		}
	}
	// Rectangular (nil window) frames must match the direct DFT.
	frame := signal[:nfft]
	p.RFFT(row, frame, nil)
	for k := range row {
		cmplxClose(t, fmt.Sprintf("rect bin %d", k), row[k], dftBin(frame, k), float64(nfft))
	}
}

// TestRFFTShortInputs covers the lenient edges: a frame shorter than nfft is
// zero-padded, a short window is treated as rectangular, and dst is clamped.
func TestRFFTShortInputs(t *testing.T) {
	const nfft = 64
	p, _ := NewSTFTPlan(nfft)
	signal := testSignal(nfft)
	full := make([]complex128, p.NumBins())
	short := make([]complex128, p.NumBins())

	// Short frame == zero-padded frame.
	padded := make([]float64, nfft)
	copy(padded, signal[:40])
	p.RFFT(full, padded, nil)
	p.RFFT(short, signal[:40], nil)
	for k := range full {
		if full[k] != short[k] {
			t.Fatalf("bin %d: short frame %v != zero-padded %v", k, short[k], full[k])
		}
	}

	// Short window == rectangular.
	p.RFFT(full, signal, nil)
	p.RFFT(short, signal, hann(nfft/2))
	for k := range full {
		if full[k] != short[k] {
			t.Fatalf("bin %d: short window %v != rectangular %v", k, short[k], full[k])
		}
	}

	// dst clamp and zero-length dst.
	if n := p.RFFT(make([]complex128, 5), signal, nil); n != 5 {
		t.Fatalf("RFFT into 5 bins wrote %d", n)
	}
	if n := p.RFFT(nil, signal, nil); n != 0 {
		t.Fatalf("RFFT into nil dst wrote %d", n)
	}
}

func TestRFFTAllocFree(t *testing.T) {
	p, _ := NewSTFTPlan(1024)
	frame := testSignal(1024)
	window := hann(1024)
	dst := make([]complex128, p.NumBins())
	if a := testing.AllocsPerRun(10, func() { p.RFFT(dst, frame, window) }); a != 0 {
		t.Errorf("RFFT allocated %v times per run, want 0", a)
	}
}

// naiveIRFFT is the reference inverse real DFT: x[n] = (1/N) * (Re X[0] +
// Re X[N/2] * (-1)^n + 2 * sum_{k=1}^{N/2-1} Re(X[k] e^{+2 pi i k n / N})),
// evaluated in float64. By construction it ignores the imaginary parts of the
// DC and Nyquist bins, which is the numpy.fft.irfft convention IRFFT follows.
func naiveIRFFT(spec []complex128, nfft int) []float64 {
	half := nfft / 2
	x := make([]float64, nfft)
	for n := range nfft {
		acc := real(spec[0])
		if n%2 == 0 {
			acc += real(spec[half])
		} else {
			acc -= real(spec[half])
		}
		for k := 1; k < half; k++ {
			ang := 2 * math.Pi * float64(k) * float64(n) / float64(nfft)
			s, c := math.Sincos(ang)
			acc += 2 * (real(spec[k])*c - imag(spec[k])*s)
		}
		x[n] = acc / float64(nfft)
	}
	return x
}

// randomSpectrum builds a deterministic complex half-spectrum with non-zero
// imaginary parts everywhere, including DC and Nyquist.
func randomSpectrum(bins int, seed float64) []complex128 {
	s := make([]complex128, bins)
	for k := range s {
		a := float64(k) + seed
		s[k] = complex(math.Sin(0.7*a)+0.3*math.Cos(1.9*a), math.Cos(0.4*a)-0.5*math.Sin(2.3*a))
	}
	return s
}

func TestIRFFTAgainstNaive(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 64, 256, 1024, 4096} {
		p, _ := NewSTFTPlan(nfft)
		spec := randomSpectrum(p.NumBins(), 0.5)
		want := naiveIRFFT(spec, nfft)
		got := make([]float64, nfft)
		if n := p.IRFFT(got, spec); n != nfft {
			t.Fatalf("nfft=%d: IRFFT wrote %d samples, want %d", nfft, n, nfft)
		}
		// Bin magnitudes are O(1), so |x| is O(1) after the 1/N scale; the
		// float64 transform error grows with log2(nfft).
		tol := stftTol(nfft, 2)
		for i := range got {
			if d := math.Abs(got[i] - want[i]); d > tol {
				t.Fatalf("nfft=%d sample %d: got %g want %g (|diff|=%g tol=%g)", nfft, i, got[i], want[i], d, tol)
			}
		}
	}
}

func TestRFFTIRFFTRoundTrip(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 64, 1024, 4096} {
		p, _ := NewSTFTPlan(nfft)
		x := testSignal(nfft)
		spec := make([]complex128, p.NumBins())
		y := make([]float64, nfft)
		p.RFFT(spec, x, nil)
		p.IRFFT(y, spec)
		tol := stftTol(nfft, 2)
		for i := range x {
			if d := math.Abs(y[i] - x[i]); d > tol {
				t.Fatalf("nfft=%d sample %d: round trip %g != %g (|diff|=%g tol=%g)", nfft, i, y[i], x[i], d, tol)
			}
		}
	}
}

// TestIRFFTShortInputs covers the lenient edges: missing bins are zero, dst is
// clamped, and an empty dst is a no-op.
func TestIRFFTShortInputs(t *testing.T) {
	const nfft = 32
	p, _ := NewSTFTPlan(nfft)
	spec := randomSpectrum(p.NumBins(), 1.5)

	// Missing bins == zero bins.
	zeroed := make([]complex128, len(spec))
	copy(zeroed, spec[:10])
	want := make([]float64, nfft)
	got := make([]float64, nfft)
	p.IRFFT(want, zeroed)
	p.IRFFT(got, spec[:10])
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("sample %d: short spec %g != zero-filled %g", i, got[i], want[i])
		}
	}

	// dst clamp: the first 7 samples equal the full transform's first 7.
	p.IRFFT(want, spec)
	part := make([]float64, 7)
	if n := p.IRFFT(part, spec); n != 7 {
		t.Fatalf("IRFFT into 7 samples wrote %d", n)
	}
	for i := range part {
		if part[i] != want[i] {
			t.Fatalf("sample %d: clamped %g != full %g", i, part[i], want[i])
		}
	}
	if n := p.IRFFT(nil, spec); n != 0 {
		t.Fatalf("IRFFT into nil dst wrote %d", n)
	}
}

func TestIRFFTAllocFree(t *testing.T) {
	p, _ := NewSTFTPlan(1024)
	spec := randomSpectrum(p.NumBins(), 2.5)
	dst := make([]float64, 1024)
	if a := testing.AllocsPerRun(10, func() { p.IRFFT(dst, spec) }); a != 0 {
		t.Errorf("IRFFT allocated %v times per run, want 0", a)
	}
}

// wolaNorm computes the squared-window overlap sum_f w^2[u - f*hop] at padded
// position u for the given frame count, the normalization ISTFT applies.
func wolaNorm(window []float64, nfft, hop, frames, u int) float64 {
	var norm float64
	for f := range frames {
		j := u - f*hop
		if j < 0 || j >= nfft {
			continue
		}
		w := 1.0
		if window != nil {
			w = window[j]
		}
		norm += w * w
	}
	return norm
}

// TestISTFTRoundTrip checks ISTFT(STFT(x)) == x for every pad mode, at hops
// nfft/4 and nfft/2, with Hann and rectangular windows. Samples whose squared
// window overlap is below 1e-3 (only the outermost NoPad edge samples under a
// Hann window, which ISTFT leaves unnormalized) are excluded from the check.
func TestISTFTRoundTrip(t *testing.T) {
	const nfft = 256
	x := testSignal(3000) // not a multiple of any hop, so the tail is partial
	for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
		for _, hop := range []int{nfft / 4, nfft / 2} {
			for _, window := range [][]float64{hann(nfft), nil} {
				p, _ := NewSTFTPlan(nfft)
				frames := p.NumFrames(len(x), hop, pad)
				spec := make([][]complex128, frames)
				for f := range spec {
					spec[f] = make([]complex128, p.NumBins())
				}
				p.STFT(spec, x, window, hop, pad)
				y := make([]float64, len(x))
				n := p.ISTFT(y, spec, window, hop, pad)
				var wantN int
				if pad == NoPad {
					wantN = min(len(x), (frames-1)*hop+nfft)
				} else {
					wantN = min(len(x), (frames-1)*hop)
				}
				if n != wantN {
					t.Fatalf("pad=%v hop=%d win=%v: wrote %d samples, want %d", pad, hop, window != nil, n, wantN)
				}
				off := 0
				if pad != NoPad {
					off = nfft / 2
				}
				tol := stftTol(nfft, 4)
				for i := range n {
					if wolaNorm(window, nfft, hop, frames, i+off) < 1e-3 {
						continue
					}
					if d := math.Abs(y[i] - x[i]); d > tol {
						t.Fatalf("pad=%v hop=%d win=%v sample %d: %g != %g (|diff|=%g tol=%g)", pad, hop, window != nil, i, y[i], x[i], d, tol)
					}
				}
			}
		}
	}
}

// TestISTFTGuards covers the degenerate inputs: no frames, bad hop, empty dst, a
// dst longer than the reconstructable length (clamped), and a short window
// treated as rectangular.
func TestISTFTGuards(t *testing.T) {
	const nfft, hop = 64, 16
	p, _ := NewSTFTPlan(nfft)
	x := testSignal(512)
	frames := p.NumFrames(len(x), hop, NoPad)
	spec := make([][]complex128, frames)
	for f := range spec {
		spec[f] = make([]complex128, p.NumBins())
	}
	p.STFT(spec, x, nil, hop, NoPad)

	if n := p.ISTFT(make([]float64, 10), nil, nil, hop, NoPad); n != 0 {
		t.Fatalf("no frames wrote %d", n)
	}
	if n := p.ISTFT(make([]float64, 10), spec, nil, 0, NoPad); n != 0 {
		t.Fatalf("hop 0 wrote %d", n)
	}
	if n := p.ISTFT(nil, spec, nil, hop, NoPad); n != 0 {
		t.Fatalf("nil dst wrote %d", n)
	}
	long := make([]float64, 10000)
	if n := p.ISTFT(long, spec, nil, hop, NoPad); n != (frames-1)*hop+nfft {
		t.Fatalf("long dst wrote %d, want %d", n, (frames-1)*hop+nfft)
	}
	a := make([]float64, 512)
	b := make([]float64, 512)
	p.ISTFT(a, spec, nil, hop, NoPad)
	p.ISTFT(b, spec, hann(nfft/2), hop, NoPad)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("sample %d: short window %g != rectangular %g", i, b[i], a[i])
		}
	}
}

func TestISTFTAllocFree(t *testing.T) {
	const nfft, hop = 512, 128
	p, _ := NewSTFTPlan(nfft)
	x := testSignal(8192)
	window := hann(nfft)
	frames := p.NumFrames(len(x), hop, PadZero)
	spec := make([][]complex128, frames)
	for f := range spec {
		spec[f] = make([]complex128, p.NumBins())
	}
	p.STFT(spec, x, window, hop, PadZero)
	y := make([]float64, len(x))
	if a := testing.AllocsPerRun(5, func() { p.ISTFT(y, spec, window, hop, PadZero) }); a != 0 {
		t.Errorf("ISTFT allocated %v times per run, want 0", a)
	}
}

//go:embed testdata/istft_librosa_golden.json
var istftLibrosaGoldenJSON []byte

// TestISTFTLibrosaParity pins ISTFT's normalization and centering against
// librosa.istft on a spectrum with a per-bin gain applied, so the test is not
// satisfied by the identity round trip alone. The forward spectrum comes from
// this package's own STFT (already pinned to librosa by TestSTFTLibrosaParity).
func TestISTFTLibrosaParity(t *testing.T) {
	var g struct {
		LibrosaVersion string  `json:"librosa_version"`
		NFFT           int     `json:"nfft"`
		Hop            int     `json:"hop"`
		N              int     `json:"n"`
		GainPeriod     float64 `json:"gain_period"`
		Cases          []struct {
			GoPad  string    `json:"go_pad"`
			Frames int       `json:"frames"`
			Y      []float64 `json:"y"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(istftLibrosaGoldenJSON, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	t.Logf("golden generated by librosa %s (nfft=%d hop=%d)", g.LibrosaVersion, g.NFFT, g.Hop)
	signal := testSignal(g.N)
	window := hann(g.NFFT)
	p, _ := NewSTFTPlan(g.NFFT)
	gain := make([]float64, p.NumBins())
	for k := range gain {
		gain[k] = 0.5 + 0.5*math.Cos(float64(k)/g.GainPeriod)
	}
	padOf := map[string]PadMode{"PadZero": PadZero, "PadReflect": PadReflect}
	for _, c := range g.Cases {
		pad := padOf[c.GoPad]
		frames := p.NumFrames(len(signal), g.Hop, pad)
		if frames != c.Frames {
			t.Fatalf("%s: NumFrames=%d but librosa produced %d", c.GoPad, frames, c.Frames)
		}
		spec := make([][]complex128, frames)
		for f := range spec {
			spec[f] = make([]complex128, p.NumBins())
		}
		p.STFT(spec, signal, window, g.Hop, pad)
		for f := range spec {
			for k := range spec[f] {
				spec[f][k] *= complex(gain[k], 0)
			}
		}
		y := make([]float64, g.N)
		if n := p.ISTFT(y, spec, window, g.Hop, pad); n != g.N {
			t.Fatalf("%s: ISTFT wrote %d samples, want %d", c.GoPad, n, g.N)
		}
		var peak float64
		for _, v := range c.Y {
			peak = max(peak, math.Abs(v))
		}
		var maxErr float64
		for i := range y {
			maxErr = max(maxErr, math.Abs(y[i]-c.Y[i]))
		}
		// float64 forward + inverse against librosa's float64: the error is at
		// the golden's rounding (~1e-9 of peak); 1e-6 keeps wide margin while any
		// convention slip (missing normalization, wrong trim) is far larger.
		if maxErr > 1e-6*peak {
			t.Errorf("%s: max abs error %g exceeds 1e-6 of peak %g vs librosa", c.GoPad, maxErr, peak)
		}
	}
}
