package f32

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

// dftBinF32 computes a single DFT bin X[k] = sum_n frame[n] * exp(-i 2pi k n / N)
// directly in float64, as an independent reference for the float32 FFT-based
// STFT. The frame samples are float32 but the reference math is float64 so the
// comparison measures the f32 transform's error against the true value.
func dftBinF32(frame []float32, k int) complex128 {
	n := len(frame)
	var re, im float64
	for t := range n {
		ang := -2 * math.Pi * float64(k) * float64(t) / float64(n)
		s, c := math.Sincos(ang)
		re += float64(frame[t]) * c
		im += float64(frame[t]) * s
	}
	return complex(re, im)
}

func hannF32(nfft int) []float32 {
	w := make([]float32, nfft)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(nfft)))
	}
	return w
}

// testSignalF32 builds a deterministic pseudo-random-ish real signal.
func testSignalF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(math.Sin(0.3*float64(i)) + 0.5*math.Cos(0.11*float64(i)+1) - 0.25*math.Sin(0.027*float64(i)))
	}
	return s
}

// stftTolF32 bounds the error of an nfft-point float32 rfft (radix-4 core with a
// trailing radix-2 stage, vector pack and unravel) against
// the true (float64) DFT: the relative error grows roughly with log2(nfft)*eps32,
// scaled by the sum of frame magnitudes that bounds the bin, plus a floor.
func stftTolF32(nfft int, scale float64) float64 {
	const eps32 = 1.1920928955078125e-07
	logN := math.Log2(float64(nfft))
	return 8*logN*eps32*scale + 1e-5
}

func cmplxCloseF32(t *testing.T, ctx string, got complex64, want complex128, tol float64) {
	t.Helper()
	if d := math.Hypot(float64(real(got))-real(want), float64(imag(got))-imag(want)); d > tol {
		t.Fatalf("%s: got %v want %v (|diff|=%g tol=%g)", ctx, got, want, d, tol)
	}
}

func TestNewSTFTPlanErrorsF32(t *testing.T) {
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

// TestSTFTAgainstDFTF32 is the core correctness gate: every bin of every frame
// must match a direct DFT of the windowed frame within the float32 tolerance,
// across nfft sizes, hops, and with or without a window.
func TestSTFTAgainstDFTF32(t *testing.T) {
	signal := testSignalF32(5000)
	// The size list spans both radix-4 schedule shapes: even log2(half) runs only
	// radix-4 stages (nfft 8/32/128 = 1/2/3 stages, no trailing), odd log2(half)
	// finishes with one trailing radix-2 stage (nfft 4/16/64/256/1024).
	for _, nfft := range []int{2, 4, 8, 16, 32, 64, 128, 256, 1024} {
		for _, useWin := range []bool{false, true} {
			plan, err := NewSTFTPlan(nfft)
			if err != nil {
				t.Fatal(err)
			}
			var window []float32
			if useWin {
				window = hannF32(nfft)
			}
			hop := max(nfft/2, 1)
			nf := plan.NumFrames(len(signal), hop, NoPad)
			dst := make([][]complex64, nf)
			for f := range dst {
				dst[f] = make([]complex64, plan.NumBins())
			}
			got := plan.STFT(dst, signal, window, hop, NoPad)
			if got != nf {
				t.Fatalf("nfft=%d: STFT wrote %d frames, want %d", nfft, got, nf)
			}

			frame := make([]float32, nfft)
			for f := range nf {
				base := f * hop
				var scale float64
				for i := range nfft {
					v := signal[base+i]
					if window != nil {
						v *= window[i]
					}
					frame[i] = v
					scale += math.Abs(float64(v))
				}
				tol := stftTolF32(nfft, scale)
				for k := range plan.NumBins() {
					want := dftBinF32(frame, k)
					ctx := fmt.Sprintf("nfft=%d win=%v frame=%d bin=%d", nfft, useWin, f, k)
					cmplxCloseF32(t, ctx, dst[f][k], want, tol)
				}
			}
		}
	}
}

// TestSTFTStageTwiddles pins the per-stage contiguous twiddle layout fftHalf
// slices for ButterflyComplexStage4 (tw1/tw2) and the trailing
// ButterflyComplexStage: the table holds max(half-1, 0)
// entries, and stage m in {2,4,...,half} keeps its span=m/2 factors
// W_m^j = (cos(2*pi*j/m), -sin(2*pi*j/m)) at offset span-1. An off-by-one in the
// offset or a wrong table length would put the wrong factor at [off+j] and fail
// here, localized, rather than only surfacing as a spectrum mismatch downstream.
// The f32 sibling of f64's TestSTFTStageTwiddles; the tolerance is float32-scale
// (the factors are stored as float32 rounded once from a float64 sincos).
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
				// NewSTFTPlan stores float32(c)/float32(-s) from this same float64
				// sincos, so the stored bits must match exactly (same machine, same
				// expression); an exact check catches any factor corruption a loose
				// tolerance would let pass.
				wantRe, wantIm := float32(c), float32(-s)
				if math.Float32bits(p.stageTwRe[off+j]) != math.Float32bits(wantRe) {
					t.Errorf("nfft=%d m=%d j=%d: stageTwRe=%g want %g", nfft, m, j, p.stageTwRe[off+j], wantRe)
				}
				if math.Float32bits(p.stageTwIm[off+j]) != math.Float32bits(wantIm) {
					t.Errorf("nfft=%d m=%d j=%d: stageTwIm=%g want %g", nfft, m, j, p.stageTwIm[off+j], wantIm)
				}
			}
		}
	}
}

// TestSTFTStage4Tw3TwiddlesF32 pins the layout of the radix-4 core's third
// twiddle table, the one fftHalf cannot slice from the radix-2 tables: the
// radix-4 stage with span s in {1,4,16,...} (while 4*s <= half) keeps s factors
// w^(3j) = (cos(2*pi*3j/(4s)), -sin(2*pi*3j/(4s))) at offset (s-1)/3, and the
// table totals (4^stages - 1)/3 entries, zero when half < 4. A wrong offset,
// length or power would fail here, localized, rather than only as a spectrum
// mismatch downstream. Exact bits, as in TestSTFTStageTwiddles: NewSTFTPlan
// stores float32(cos)/float32(-sin) of this same float64 sincos.
func TestSTFTStage4Tw3TwiddlesF32(t *testing.T) {
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
				wantRe, wantIm := float32(cos), float32(-sin)
				if math.Float32bits(p.stage4Tw3Re[off+j]) != math.Float32bits(wantRe) {
					t.Errorf("nfft=%d s=%d j=%d: stage4Tw3Re=%g want %g", nfft, s, j, p.stage4Tw3Re[off+j], wantRe)
				}
				if math.Float32bits(p.stage4Tw3Im[off+j]) != math.Float32bits(wantIm) {
					t.Errorf("nfft=%d s=%d j=%d: stage4Tw3Im=%g want %g", nfft, s, j, p.stage4Tw3Im[off+j], wantIm)
				}
			}
			off += s
		}
	}
}

// TestSTFTPowerMatchesSTFTF32 verifies STFTPower equals |STFT|^2 bin-for-bin
// (both derive from the same unravel, so the agreement is essentially exact).
func TestSTFTPowerMatchesSTFTF32(t *testing.T) {
	signal := testSignalF32(4096)
	plan, _ := NewSTFTPlan(512)
	window := hannF32(512)
	hop := 128
	nf := plan.NumFrames(len(signal), hop, NoPad)

	spec := make([][]complex64, nf)
	pow := make([][]float32, nf)
	for f := range spec {
		spec[f] = make([]complex64, plan.NumBins())
		pow[f] = make([]float32, plan.NumBins())
	}
	plan.STFT(spec, signal, window, hop, NoPad)
	plan.STFTPower(pow, signal, window, hop, NoPad)

	for f := range nf {
		for k := range plan.NumBins() {
			want := real(spec[f][k])*real(spec[f][k]) + imag(spec[f][k])*imag(spec[f][k])
			if d := math.Abs(float64(pow[f][k] - want)); d > 1e-6*(1+float64(want)) {
				t.Fatalf("STFTPower[%d][%d] = %v, want |X|^2 = %v", f, k, pow[f][k], want)
			}
		}
	}
}

// TestSTFTPureToneF32 checks a single-bin cosine concentrates its energy in that
// bin and that DC/Nyquist come out (numerically) real.
func TestSTFTPureToneF32(t *testing.T) {
	const nfft = 64
	plan, _ := NewSTFTPlan(nfft)
	k0 := 5
	signal := make([]float32, nfft)
	for n := range signal {
		signal[n] = float32(math.Cos(2 * math.Pi * float64(k0) * float64(n) / float64(nfft)))
	}
	dst := [][]complex64{make([]complex64, plan.NumBins())}
	plan.STFT(dst, signal, nil, nfft, NoPad)

	mag := func(c complex64) float64 { return math.Hypot(float64(real(c)), float64(imag(c))) }
	// Bin k0 should hold ~nfft/2; every other bin should be ~0.
	if got := mag(dst[0][k0]); math.Abs(got-float64(nfft)/2) > 1e-3 {
		t.Errorf("tone bin %d magnitude = %v, want ~%v", k0, got, float64(nfft)/2)
	}
	for k := range plan.NumBins() {
		if k == k0 {
			continue
		}
		if got := mag(dst[0][k]); got > 1e-3 {
			t.Errorf("non-tone bin %d magnitude = %v, want ~0", k, got)
		}
	}
	// DC and Nyquist bins of a real signal are real.
	if math.Abs(float64(imag(dst[0][0]))) > 1e-4 {
		t.Errorf("DC bin not real: %v", dst[0][0])
	}
	if math.Abs(float64(imag(dst[0][plan.NumBins()-1]))) > 1e-4 {
		t.Errorf("Nyquist bin not real: %v", dst[0][plan.NumBins()-1])
	}
}

// TestSTFTFramingF32 checks frame counting and the no-padding (center=false)
// convention: frame f starts at f*hop.
func TestSTFTFramingF32(t *testing.T) {
	plan, _ := NewSTFTPlan(8)
	signal := make([]float32, 20)
	for i := range signal {
		signal[i] = float32(i)
	}
	hop := 4
	// frames at offsets 0,4,8,12 fit (need 8 samples): 12+8=20 ok, 16+8=24 no.
	wantFrames := 4
	if got := plan.NumFrames(len(signal), hop, NoPad); got != wantFrames {
		t.Fatalf("numFrames = %d, want %d", got, wantFrames)
	}
	dst := make([][]complex64, wantFrames)
	for f := range dst {
		dst[f] = make([]complex64, plan.NumBins())
	}
	if n := plan.STFT(dst, signal, nil, hop, NoPad); n != wantFrames {
		t.Fatalf("STFT frames = %d, want %d", n, wantFrames)
	}
	// DC bin of frame f is the sum of signal[f*hop : f*hop+8].
	for f := range wantFrames {
		var sum float64
		for i := range 8 {
			sum += float64(signal[f*hop+i])
		}
		if math.Abs(float64(real(dst[f][0]))-sum) > 1e-4*(1+sum) {
			t.Errorf("frame %d DC = %v, want %v", f, real(dst[f][0]), sum)
		}
	}
}

// TestSTFTClampsF32 verifies dst shorter than the frame count, and rows shorter
// than NumBins, are handled without panic.
func TestSTFTClampsF32(t *testing.T) {
	plan, _ := NewSTFTPlan(16)
	signal := testSignalF32(200)
	hop := 8
	full := plan.NumFrames(len(signal), hop, NoPad)

	// Fewer rows than frames: only len(dst) frames written.
	short := make([][]complex64, full-2)
	for f := range short {
		short[f] = make([]complex64, plan.NumBins())
	}
	if n := plan.STFT(short, signal, nil, hop, NoPad); n != full-2 {
		t.Errorf("clamped frames = %d, want %d", n, full-2)
	}

	// Rows shorter than NumBins: only the available bins written, no panic.
	rows := make([][]complex64, 1)
	rows[0] = make([]complex64, 3)
	if n := plan.STFT(rows, signal, nil, hop, NoPad); n != 1 {
		t.Errorf("partial-row frames = %d, want 1", n)
	}

	// Rows longer than NumBins: exactly NumBins bins written, the rest untouched,
	// and the written bins identical to a NumBins-wide row (same path).
	bins := plan.NumBins()
	const sentinel = complex64(complex(-7, 7))
	long := [][]complex64{make([]complex64, bins+3)}
	for k := range long[0] {
		long[0][k] = sentinel
	}
	exact := [][]complex64{make([]complex64, bins)}
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
	longPow := [][]float32{make([]float32, bins+3)}
	for k := range longPow[0] {
		longPow[0][k] = -7
	}
	if n := plan.STFTPower(longPow, signal, nil, hop, NoPad); n != 1 {
		t.Errorf("long-row power frames = %d, want 1", n)
	}
	exactPow := [][]float32{make([]float32, bins)}
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

// TestSTFTShortRowsMatchFullF32 pins the two unravel paths against each other:
// a full-width row (NumBins) takes the vector RealFFTUnpack / RealFFTPower
// unravel, a shorter row keeps the per-bin scalar unravel. The bins a short row
// does write must agree with the same bins of the full row, for STFT and
// STFTPower, across row lengths on both sides of the split and across nfft sizes
// that exercise the vector kernels' full blocks and scalar tails. It also proves
// the full-width row really took the vector path: the plan's unpack scratch is
// poisoned with NaN before the call and the row's interior bins must be the
// bits that scratch holds afterwards (only RealFFTUnpack writes it).
func TestSTFTShortRowsMatchFullF32(t *testing.T) {
	signal := testSignalF32(3000)
	nan := float32(math.NaN())
	for _, nfft := range []int{2, 4, 16, 64, 512} {
		plan, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatal(err)
		}
		window := hannF32(nfft)
		hop := max(nfft/4, 1)
		bins := plan.NumBins()
		half := bins - 1
		nf := plan.NumFrames(len(signal), hop, NoPad)

		full := make([][]complex64, nf)
		fullPow := make([][]float32, nf)
		for f := range nf {
			full[f] = make([]complex64, bins)
			fullPow[f] = make([]float32, bins)
		}
		for k := range plan.outRe {
			plan.outRe[k], plan.outIm[k] = nan, nan
		}
		plan.STFT(full, signal, window, hop, NoPad)
		// Positive control: the last full row's interior bins are exactly the
		// unpack scratch the vector path wrote (NaN would mean it never ran).
		for k := 1; k < half; k++ {
			if got, wr, wi := full[nf-1][k], plan.outRe[k], plan.outIm[k]; math.IsNaN(float64(wr)) || math.IsNaN(float64(wi)) ||
				math.Float32bits(real(got)) != math.Float32bits(wr) || math.Float32bits(imag(got)) != math.Float32bits(wi) {
				t.Fatalf("nfft=%d bin=%d: full row %v is not the vector unpack scratch (%v,%v)", nfft, k, got, wr, wi)
			}
		}
		// The rounding gap between the FMA vector unpack and the scalar unravel
		// scales with the packed spectrum's magnitude, not with the bin being
		// compared (a near-null bin is a cancellation of large terms), so the
		// tolerance is relative to the row's largest bin.
		rowMax := 0.0
		for k := range bins {
			rowMax = max(rowMax, math.Hypot(float64(real(full[nf-1][k])), float64(imag(full[nf-1][k]))))
		}
		tol := 1e-5 * (1 + rowMax)
		// The plan scratch still holds the last frame's half-size spectrum, so
		// every bin of the last full row, DC and Nyquist included, can be checked
		// against the scalar per-bin unravel the short-row path uses.
		for k := range bins {
			xr, xi := plan.unravelBin(k)
			ctx := fmt.Sprintf("nfft=%d last frame bin=%d (vector vs scalar unravel)", nfft, k)
			cmplxCloseF32(t, ctx, full[nf-1][k], complex(float64(xr), float64(xi)), tol)
		}
		plan.STFTPower(fullPow, signal, window, hop, NoPad)
		powTol := 1e-5 * (1 + rowMax*rowMax)
		for k := range bins {
			xr, xi := plan.unravelBin(k)
			wp := xr*xr + xi*xi
			if d := math.Abs(float64(fullPow[nf-1][k] - wp)); d > powTol {
				t.Fatalf("nfft=%d last frame bin=%d: vector power %v, scalar power %v", nfft, k, fullPow[nf-1][k], wp)
			}
		}

		for _, rowLen := range []int{1, 2, bins / 2, bins - 1, bins} {
			short := make([][]complex64, nf)
			shortPow := make([][]float32, nf)
			for f := range nf {
				short[f] = make([]complex64, rowLen)
				shortPow[f] = make([]float32, rowLen)
			}
			plan.STFT(short, signal, window, hop, NoPad)
			plan.STFTPower(shortPow, signal, window, hop, NoPad)
			for f := range nf {
				for k := range rowLen {
					ctx := fmt.Sprintf("nfft=%d rowLen=%d frame=%d bin=%d", nfft, rowLen, f, k)
					cmplxCloseF32(t, ctx, short[f][k], complex128(full[f][k]), tol)
					if d := math.Abs(float64(shortPow[f][k] - fullPow[f][k])); d > powTol {
						t.Fatalf("%s: short-row power %v, full-row power %v", ctx, shortPow[f][k], fullPow[f][k])
					}
				}
			}
		}
	}
}

// TestSTFTWindowReuseF32 pins that every public call applies exactly the window
// it was given: the plan splits the window into per-call scratch for the vector
// pack, so a stale split would leak a previous call's window into the next. One
// shared plan runs a sequence of calls that changes the window on EVERY call
// (Hann to ramp directly, to and from rectangular, a window longer than nfft)
// and rotates through STFT, STFTPowerInto and STFTPower, and each call is
// compared bit for bit against a plan created for that one call only (a fresh
// plan replaying the same sequence would carry the same stale split and hide
// the defect). A window longer than nfft must behave as its first nfft samples.
func TestSTFTWindowReuseF32(t *testing.T) {
	const nfft = 64
	signal := testSignalF32(2000)
	hop := 16
	hann := hannF32(nfft)
	ramp := make([]float32, nfft)
	for i := range ramp {
		ramp[i] = float32(i+1) / float32(nfft)
	}
	hannLong := append(append([]float32(nil), hann...), 9, 9, 9, 9, 9)
	shared, _ := NewSTFTPlan(nfft)
	nf := shared.NumFrames(len(signal), hop, PadReflect)
	bins := shared.NumBins()
	newSpec := func() [][]complex64 {
		spec := make([][]complex64, nf)
		for f := range spec {
			spec[f] = make([]complex64, bins)
		}
		return spec
	}
	newPow := func() [][]float32 {
		pow := make([][]float32, nf)
		for f := range pow {
			pow[f] = make([]float32, bins)
		}
		return pow
	}
	windows := [][]float32{hann, ramp, hann, ramp, hann, ramp, nil, hannLong, ramp}
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
			got, want := make([]float32, nf*bins), make([]float32, nf*bins)
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
	hannPlan.STFT(want, signal, hann, hop, PadReflect)
	for f := range nf {
		for k := range bins {
			if got[f][k] != want[f][k] {
				t.Fatalf("long window frame %d bin %d: %v, want Hann-prefix %v", f, k, got[f][k], want[f][k])
			}
		}
	}
}

func TestSTFTAllocFreeF32(t *testing.T) {
	plan, _ := NewSTFTPlan(512)
	signal := testSignalF32(8192)
	window := hannF32(512)
	hop := 128
	nf := plan.NumFrames(len(signal), hop, NoPad)
	spec := make([][]complex64, nf)
	pow := make([][]float32, nf)
	for f := range spec {
		spec[f] = make([]complex64, plan.NumBins())
		pow[f] = make([]float32, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(spec, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("STFT allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(pow, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("STFTPower allocated %v times per run, want 0", a)
	}

	// Centered framing and the flat output must also be allocation-free.
	cf := plan.NumFrames(len(signal), hop, PadReflect)
	cpow := make([][]float32, cf)
	for f := range cpow {
		cpow[f] = make([]float32, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(cpow, signal, window, hop, PadReflect) }); a != 0 {
		t.Errorf("centered STFTPower allocated %v times per run, want 0", a)
	}
	flat := make([]float32, plan.NumFrames(len(signal), hop, PadZero)*plan.NumBins())
	if a := testing.AllocsPerRun(5, func() { plan.STFTPowerInto(flat, signal, window, hop, PadZero) }); a != 0 {
		t.Errorf("STFTPowerInto allocated %v times per run, want 0", a)
	}

	// The complex STFT with centered framing uses the same packFrameAt edge path
	// and must also be allocation-free.
	cspec := make([][]complex64, cf)
	for f := range cspec {
		cspec[f] = make([]complex64, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(cspec, signal, window, hop, PadReflect) }); a != 0 {
		t.Errorf("centered STFT allocated %v times per run, want 0", a)
	}

	// Rows shorter than NumBins take the scalar unravel branch, which must also be
	// allocation-free.
	sspec := make([][]complex64, nf)
	spow := make([][]float32, nf)
	for f := range sspec {
		sspec[f] = make([]complex64, 3)
		spow[f] = make([]float32, 3)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(sspec, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("short-row STFT allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(spow, signal, window, hop, NoPad) }); a != 0 {
		t.Errorf("short-row STFTPower allocated %v times per run, want 0", a)
	}
}

// FuzzSTFT is a differential fuzz target: every STFT bin must match a direct DFT
// of the windowed frame within the float32 tolerance, across fuzzed signal
// contents, nfft, hop, and window choice. Inputs are bounded to [-1, 1] (via
// f32sUnit) so the DFT bin magnitudes stay well-conditioned. Seeds run under
// plain `go test`; `go test -fuzz=FuzzSTFT` widens the space.
func FuzzSTFT(f *testing.F) {
	f.Add(make([]byte, 256), uint8(3), uint8(7), false, uint8(0))
	f.Add(make([]byte, 600), uint8(5), uint8(3), true, uint8(1))
	f.Add(make([]byte, 600), uint8(4), uint8(2), true, uint8(2))

	f.Fuzz(func(t *testing.T, raw []byte, nfftSel, hopSel uint8, useWin bool, padSel uint8) {
		// nfft in {4, 8, 16, 32, 64}; keep it small so the O(n^2) DFT is cheap.
		nfft := 1 << (2 + int(nfftSel)%5)
		signal := f32sUnit(raw)
		if len(signal) < nfft {
			return
		}
		plan, err := NewSTFTPlan(nfft)
		if err != nil {
			t.Fatal(err)
		}
		var window []float32
		if useWin {
			window = hannF32(nfft)
		}
		hop := 1 + int(hopSel)%nfft
		pad := []PadMode{NoPad, PadZero, PadReflect}[int(padSel)%3]
		nf := plan.NumFrames(len(signal), hop, pad)
		if nf == 0 {
			return
		}
		dst := make([][]complex64, nf)
		for i := range dst {
			dst[i] = make([]complex64, plan.NumBins())
		}
		plan.STFT(dst, signal, window, hop, pad)

		// Compare against the independent reference for every pad mode.
		ref := stftRefF32(signal, window, nfft, hop, pad)
		if len(ref) != nf {
			t.Fatalf("nfft=%d hop=%d pad=%v: ref frames %d != NumFrames %d", nfft, hop, pad, len(ref), nf)
		}
		off := 0
		if pad != NoPad {
			off = nfft / 2
		}
		frame := make([]float32, nfft)
		for fr := range nf {
			var scale float64
			base := fr*hop - off
			for i := range nfft {
				v := refSampleAtF32(signal, base+i, pad)
				if window != nil {
					v *= window[i]
				}
				frame[i] = v
				scale += math.Abs(float64(v))
			}
			tol := stftTolF32(nfft, scale)
			for k := range plan.NumBins() {
				got := dst[fr][k]
				want := ref[fr][k]
				if d := math.Hypot(float64(real(got))-real(want), float64(imag(got))-imag(want)); d > tol {
					t.Fatalf("nfft=%d hop=%d pad=%v frame=%d bin=%d: got %v want %v |diff|=%g tol=%g", nfft, hop, pad, fr, k, got, want, d, tol)
				}
			}
		}

		// The flat power output must equal |STFT|^2 bin-for-bin.
		bins := plan.NumBins()
		fl := make([]float32, nf*bins)
		plan.STFTPowerInto(fl, signal, window, hop, pad)
		for fr := range nf {
			for k := range bins {
				want := real(dst[fr][k])*real(dst[fr][k]) + imag(dst[fr][k])*imag(dst[fr][k])
				if d := math.Abs(float64(fl[fr*bins+k] - want)); d > 1e-6*(1+float64(want))+1e-9 {
					t.Fatalf("nfft=%d pad=%v frame=%d bin=%d: flat power %v want %v", nfft, pad, fr, k, fl[fr*bins+k], want)
				}
			}
		}
	})
}

func BenchmarkSTFT(b *testing.B) {
	const nfft = 1024
	plan, _ := NewSTFTPlan(nfft)
	window := hannF32(nfft)
	signal := testSignalF32(48000) // ~1s of 48 kHz audio
	hop := 256
	nf := plan.NumFrames(len(signal), hop, NoPad)
	dst := make([][]complex64, nf)
	for f := range dst {
		dst[f] = make([]complex64, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFT(dst, signal, window, hop, NoPad)
	}
}

func BenchmarkSTFTPower(b *testing.B) {
	const nfft = 1024
	plan, _ := NewSTFTPlan(nfft)
	window := hannF32(nfft)
	signal := testSignalF32(48000)
	hop := 256
	nf := plan.NumFrames(len(signal), hop, NoPad)
	dst := make([][]float32, nf)
	for f := range dst {
		dst[f] = make([]float32, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFTPower(dst, signal, window, hop, NoPad)
	}
}

func TestNumFramesF32(t *testing.T) {
	p, _ := NewSTFTPlan(8)
	cases := []struct {
		n, hop int
		pad    PadMode
		want   int
	}{
		{7, 4, NoPad, 0},
		{8, 4, NoPad, 1},
		{16, 4, NoPad, 3},
		{0, 4, PadZero, 0},
		{8, 4, PadZero, 3},
		{16, 4, PadReflect, 5},
		{10, 0, NoPad, 0},
	}
	for _, c := range cases {
		if got := p.NumFrames(c.n, c.hop, c.pad); got != c.want {
			t.Errorf("NumFrames(%d,%d,%v)=%d want %d", c.n, c.hop, c.pad, got, c.want)
		}
	}
}

func TestReflectIndexF32(t *testing.T) {
	// n=4: ...3 2 1 |0 1 2 3| 2 1 0 1... verified against numpy np.pad reflect.
	want := []int{3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1}
	for off, w := range want {
		idx := off - 3
		if got := reflectIndex(idx, 4); got != w {
			t.Errorf("reflectIndex(%d,4)=%d want %d", idx, got, w)
		}
	}
	if reflectIndex(-5, 1) != 0 || reflectIndex(3, 1) != 0 {
		t.Error("reflectIndex with n=1 must map every index to 0")
	}
}

// refSampleAtF32 and stftRefF32 independently re-implement centering, windowing,
// and the DFT (via dftBinF32, accumulated in float64), as a cross-check on the
// float32 FFT-based centered STFT. They are a separate implementation from the
// package's sampleAt/packFrameAt so a bug in one does not mask a bug in the other.
// refReflectIndexF32 reimplements numpy "reflect" index folding locally, kept
// deliberately independent of the production reflectIndex so the centered
// reference and the fuzz target would catch a regression in either one (rather
// than sharing the same mapping bug).
func refReflectIndexF32(idx, n int) int {
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

func refSampleAtF32(signal []float32, idx int, pad PadMode) float32 {
	if idx >= 0 && idx < len(signal) {
		return signal[idx]
	}
	if pad == PadReflect {
		return signal[refReflectIndexF32(idx, len(signal))]
	}
	return 0
}

func stftRefF32(signal, window []float32, nfft, hop int, pad PadMode) [][]complex128 {
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
	frame := make([]float32, nfft)
	for f := range out {
		base := f*hop - off
		for i := range frame {
			s := refSampleAtF32(signal, base+i, pad)
			if window != nil {
				s *= window[i]
			}
			frame[i] = s
		}
		row := make([]complex128, nfft/2+1)
		for k := range row {
			row[k] = dftBinF32(frame, k)
		}
		out[f] = row
	}
	return out
}

// windowedFrameL1F32 returns the L1 norm of the windowed, pad-aware frame f,
// which bounds the magnitude of its DFT bins for tolerance scaling.
func windowedFrameL1F32(signal, window []float32, nfft, hop, f, off int, pad PadMode) float64 {
	var scale float64
	base := f*hop - off
	for i := range nfft {
		v := refSampleAtF32(signal, base+i, pad)
		if window != nil {
			v *= window[i]
		}
		scale += math.Abs(float64(v))
	}
	return scale
}

// TestSTFTCenteredAgainstRefF32 gates the centered/padded paths: every bin of
// every frame must match the independent reference within the float32 tolerance.
func TestSTFTCenteredAgainstRefF32(t *testing.T) {
	for _, nfft := range []int{16, 64, 512, 1024} {
		for _, hop := range []int{nfft / 4, nfft / 2, nfft} {
			for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
				for _, useWin := range []bool{false, true} {
					p, _ := NewSTFTPlan(nfft)
					sig := testSignalF32(4*nfft + 7)
					var win []float32
					if useWin {
						win = hannF32(nfft)
					}
					nf := p.NumFrames(len(sig), hop, pad)
					ref := stftRefF32(sig, win, nfft, hop, pad)
					if len(ref) != nf {
						t.Fatalf("nfft=%d hop=%d pad=%v: ref frames %d != NumFrames %d", nfft, hop, pad, len(ref), nf)
					}
					dst := make([][]complex64, nf)
					for f := range dst {
						dst[f] = make([]complex64, p.NumBins())
					}
					if got := p.STFT(dst, sig, win, hop, pad); got != nf {
						t.Fatalf("nfft=%d hop=%d pad=%v: STFT wrote %d frames want %d", nfft, hop, pad, got, nf)
					}
					off := 0
					if pad != NoPad {
						off = nfft / 2
					}
					for f := range dst {
						tol := stftTolF32(nfft, windowedFrameL1F32(sig, win, nfft, hop, f, off, pad))
						for k := range dst[f] {
							ctx := fmt.Sprintf("nfft=%d hop=%d pad=%v win=%v f=%d k=%d", nfft, hop, pad, useWin, f, k)
							cmplxCloseF32(t, ctx, dst[f][k], ref[f][k], tol)
						}
					}
				}
			}
		}
	}
}

// TestSTFTPowerIntoF32 checks the flat output equals the 2D STFTPower flattened
// for every pad mode (same compute path, so essentially exact), and that a short
// flat dst floors to the frames that fit.
func TestSTFTPowerIntoF32(t *testing.T) {
	for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
		nfft, hop := 256, 192
		p, _ := NewSTFTPlan(nfft)
		sig := testSignalF32(2000)
		win := hannF32(nfft)
		bins := p.NumBins()
		nf := p.NumFrames(len(sig), hop, pad)

		ref := make([][]float32, nf)
		for f := range ref {
			ref[f] = make([]float32, bins)
		}
		if got := p.STFTPower(ref, sig, win, hop, pad); got != nf {
			t.Fatalf("pad=%v: STFTPower wrote %d frames want %d", pad, got, nf)
		}
		flat := make([]float32, nf*bins)
		if got := p.STFTPowerInto(flat, sig, win, hop, pad); got != nf {
			t.Fatalf("pad=%v: STFTPowerInto wrote %d frames want %d", pad, got, nf)
		}
		for f := range ref {
			for k := range ref[f] {
				if d := math.Abs(float64(flat[f*bins+k] - ref[f][k])); d > 1e-6*(1+float64(ref[f][k])) {
					t.Fatalf("pad=%v f=%d k=%d: flat=%g ref=%g", pad, f, k, flat[f*bins+k], ref[f][k])
				}
			}
		}
	}

	// A flat dst with room for fewer whole frames floors to what fits.
	p, _ := NewSTFTPlan(8)
	sig := testSignalF32(64)
	bins := p.NumBins()
	short := make([]float32, 2*bins+1) // 2 frames + 1 spare slot
	if got := p.STFTPowerInto(short, sig, nil, 4, NoPad); got != 2 {
		t.Fatalf("short flat dst: got %d frames want 2", got)
	}
}

//go:embed testdata/stft_librosa_golden.json
var librosaGoldenJSON []byte

// TestSTFTLibrosaParityF32 pins the float32 output convention against a golden
// vector generated by real librosa (float64). The golden is embedded so the test
// runs from any working directory (including the cross-arch copy-the-binary flow).
// The signal and window are regenerated with the same deterministic formulas the
// generator used; the tolerance is looser than f64 to absorb float32 accumulation.
func TestSTFTLibrosaParityF32(t *testing.T) {
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
	signal := testSignalF32(g.N)
	window := hannF32(g.NFFT)
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
		var peak float64
		for _, v := range c.Power {
			if v > peak {
				peak = v
			}
		}
		flat := make([]float32, c.Frames*c.Bins)
		p.STFTPowerInto(flat, signal, window, g.Hop, pad)
		var maxRel float64
		for i := range flat {
			ref := c.Power[i]
			// Floor the denominator at a small fraction of the peak power so
			// near-zero bins (where float32 noise dominates) do not blow up the
			// relative error.
			den := math.Abs(ref) + 1e-4*peak
			rel := math.Abs(float64(flat[i])-ref) / den
			if rel > maxRel {
				maxRel = rel
			}
		}
		// float32 rfft vs librosa's float64 pocketfft: the error is
		// dominated by float32 rounding (observed ~2e-6 relative after squaring to
		// power). The 1e-4 bound keeps ~50x margin while a convention error (wrong
		// centering, window, or normalization) would be orders of magnitude larger.
		if maxRel > 1e-4 {
			t.Errorf("%s: max relative error %g exceeds 1e-4 vs librosa", c.GoPad, maxRel)
		}
	}
}

// TestSTFTGuardsF32 covers the zero-frame and short-window (treated as
// rectangular) fallbacks shared by all three output methods.
func TestSTFTGuardsF32(t *testing.T) {
	p, _ := NewSTFTPlan(16)
	bins := p.NumBins()

	tiny := testSignalF32(8)
	if n := p.STFT(make([][]complex64, 4), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFT short signal: got %d frames want 0", n)
	}
	if n := p.STFTPower(make([][]float32, 4), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFTPower short signal: got %d frames want 0", n)
	}
	if n := p.STFTPowerInto(make([]float32, 4*bins), tiny, nil, 4, NoPad); n != 0 {
		t.Errorf("STFTPowerInto short signal: got %d frames want 0", n)
	}

	sig := testSignalF32(200)
	hop := 8
	shortWin := hannF32(8) // len 8 < nfft 16
	nf := p.NumFrames(len(sig), hop, NoPad)

	specShort := make([][]complex64, nf)
	specNil := make([][]complex64, nf)
	for f := range specShort {
		specShort[f] = make([]complex64, bins)
		specNil[f] = make([]complex64, bins)
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

	powShort := make([][]float32, nf)
	for f := range powShort {
		powShort[f] = make([]float32, bins)
	}
	p.STFTPower(powShort, sig, shortWin, hop, NoPad)
	flatShort := make([]float32, nf*bins)
	p.STFTPowerInto(flatShort, sig, shortWin, hop, NoPad)
	for f := range powShort {
		for k := range powShort[f] {
			if powShort[f][k] != flatShort[f*bins+k] {
				t.Fatalf("STFTPower/STFTPowerInto short-window mismatch at f=%d k=%d", f, k)
			}
		}
	}
}

// TestRFFTMatchesSTFTF32 pins RFFT to the batched transform: the single-frame
// entry point on frame f of a NoPad STFT must reproduce that row exactly (same
// code path, so bit-for-bit).
func TestRFFTMatchesSTFTF32(t *testing.T) {
	const nfft, hop = 256, 64
	p, _ := NewSTFTPlan(nfft)
	signal := testSignalF32(2048)
	window := hannF32(nfft)
	frames := p.NumFrames(len(signal), hop, NoPad)
	spec := make([][]complex64, frames)
	for f := range spec {
		spec[f] = make([]complex64, p.NumBins())
	}
	p.STFT(spec, signal, window, hop, NoPad)
	row := make([]complex64, p.NumBins())
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
		cmplxCloseF32(t, fmt.Sprintf("rect bin %d", k), row[k], dftBinF32(frame, k), stftTolF32(nfft, float64(nfft)))
	}
}

// TestRFFTShortInputsF32 covers the lenient edges: a frame shorter than nfft is
// zero-padded, a short window is treated as rectangular, and dst is clamped.
func TestRFFTShortInputsF32(t *testing.T) {
	const nfft = 64
	p, _ := NewSTFTPlan(nfft)
	signal := testSignalF32(nfft)
	full := make([]complex64, p.NumBins())
	short := make([]complex64, p.NumBins())

	// Short frame == zero-padded frame.
	padded := make([]float32, nfft)
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
	p.RFFT(short, signal, hannF32(nfft/2))
	for k := range full {
		if full[k] != short[k] {
			t.Fatalf("bin %d: short window %v != rectangular %v", k, short[k], full[k])
		}
	}

	// dst clamp and zero-length dst.
	if n := p.RFFT(make([]complex64, 5), signal, nil); n != 5 {
		t.Fatalf("RFFT into 5 bins wrote %d", n)
	}
	if n := p.RFFT(nil, signal, nil); n != 0 {
		t.Fatalf("RFFT into nil dst wrote %d", n)
	}
}

func TestRFFTAllocFreeF32(t *testing.T) {
	p, _ := NewSTFTPlan(1024)
	frame := testSignalF32(1024)
	window := hannF32(1024)
	dst := make([]complex64, p.NumBins())
	if a := testing.AllocsPerRun(10, func() { p.RFFT(dst, frame, window) }); a != 0 {
		t.Errorf("RFFT allocated %v times per run, want 0", a)
	}
}

// naiveIRFFTF32 is the reference inverse real DFT: x[n] = (1/N) * (Re X[0] +
// Re X[N/2] * (-1)^n + 2 * sum_{k=1}^{N/2-1} Re(X[k] e^{+2 pi i k n / N})),
// evaluated in float64. By construction it ignores the imaginary parts of the
// DC and Nyquist bins, which is the numpy.fft.irfft convention IRFFT follows.
func naiveIRFFTF32(spec []complex64, nfft int) []float64 {
	half := nfft / 2
	x := make([]float64, nfft)
	for n := range nfft {
		acc := float64(real(spec[0]))
		if n%2 == 0 {
			acc += float64(real(spec[half]))
		} else {
			acc -= float64(real(spec[half]))
		}
		for k := 1; k < half; k++ {
			ang := 2 * math.Pi * float64(k) * float64(n) / float64(nfft)
			s, c := math.Sincos(ang)
			acc += 2 * (float64(real(spec[k]))*c - float64(imag(spec[k]))*s)
		}
		x[n] = acc / float64(nfft)
	}
	return x
}

// randomSpectrumF32 builds a deterministic complex half-spectrum with non-zero
// imaginary parts everywhere, including DC and Nyquist.
func randomSpectrumF32(bins int, seed float64) []complex64 {
	s := make([]complex64, bins)
	for k := range s {
		a := float64(k) + seed
		s[k] = complex(float32(math.Sin(0.7*a)+0.3*math.Cos(1.9*a)), float32(math.Cos(0.4*a)-0.5*math.Sin(2.3*a)))
	}
	return s
}

func TestIRFFTAgainstNaiveF32(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 64, 256, 1024, 4096} {
		p, _ := NewSTFTPlan(nfft)
		spec := randomSpectrumF32(p.NumBins(), 0.5)
		want := naiveIRFFTF32(spec, nfft)
		got := make([]float32, nfft)
		if n := p.IRFFT(got, spec); n != nfft {
			t.Fatalf("nfft=%d: IRFFT wrote %d samples, want %d", nfft, n, nfft)
		}
		// Bin magnitudes are O(1), so |x| is O(1) after the 1/N scale; the
		// float32 transform error grows with log2(nfft).
		tol := stftTolF32(nfft, 2)
		for i := range got {
			if d := math.Abs(float64(got[i]) - want[i]); d > tol {
				t.Fatalf("nfft=%d sample %d: got %g want %g (|diff|=%g tol=%g)", nfft, i, got[i], want[i], d, tol)
			}
		}
	}
}

func TestRFFTIRFFTRoundTripF32(t *testing.T) {
	for _, nfft := range []int{2, 4, 8, 16, 64, 1024, 4096} {
		p, _ := NewSTFTPlan(nfft)
		x := testSignalF32(nfft)
		spec := make([]complex64, p.NumBins())
		y := make([]float32, nfft)
		p.RFFT(spec, x, nil)
		p.IRFFT(y, spec)
		tol := stftTolF32(nfft, 2)
		for i := range x {
			if d := math.Abs(float64(y[i] - x[i])); d > tol {
				t.Fatalf("nfft=%d sample %d: round trip %g != %g (|diff|=%g tol=%g)", nfft, i, y[i], x[i], d, tol)
			}
		}
	}
}

// TestIRFFTShortInputsF32 covers the lenient edges: missing bins are zero, dst
// is clamped, and an empty dst is a no-op.
func TestIRFFTShortInputsF32(t *testing.T) {
	const nfft = 32
	p, _ := NewSTFTPlan(nfft)
	spec := randomSpectrumF32(p.NumBins(), 1.5)

	// Missing bins == zero bins.
	zeroed := make([]complex64, len(spec))
	copy(zeroed, spec[:10])
	want := make([]float32, nfft)
	got := make([]float32, nfft)
	p.IRFFT(want, zeroed)
	p.IRFFT(got, spec[:10])
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("sample %d: short spec %g != zero-filled %g", i, got[i], want[i])
		}
	}

	// dst clamp: the first 7 samples equal the full transform's first 7.
	p.IRFFT(want, spec)
	part := make([]float32, 7)
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

func TestIRFFTAllocFreeF32(t *testing.T) {
	p, _ := NewSTFTPlan(1024)
	spec := randomSpectrumF32(p.NumBins(), 2.5)
	dst := make([]float32, 1024)
	if a := testing.AllocsPerRun(10, func() { p.IRFFT(dst, spec) }); a != 0 {
		t.Errorf("IRFFT allocated %v times per run, want 0", a)
	}
}

// wolaNormF32 computes the squared-window overlap sum_f w^2[u - f*hop] at padded
// position u for the given frame count, the normalization ISTFT applies.
func wolaNormF32(window []float32, nfft, hop, frames, u int) float64 {
	var norm float64
	for f := range frames {
		j := u - f*hop
		if j < 0 || j >= nfft {
			continue
		}
		w := 1.0
		if window != nil {
			w = float64(window[j])
		}
		norm += w * w
	}
	return norm
}

// TestISTFTRoundTripF32 checks ISTFT(STFT(x)) == x for every pad mode, at hops
// nfft/4 and nfft/2, with Hann and rectangular windows. Samples whose squared
// window overlap is below 1e-3 (only the outermost NoPad edge samples under a
// Hann window, which ISTFT leaves unnormalized) are excluded from the check.
func TestISTFTRoundTripF32(t *testing.T) {
	const nfft = 256
	x := testSignalF32(3000) // not a multiple of any hop, so the tail is partial
	for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
		for _, hop := range []int{nfft / 4, nfft / 2} {
			for _, window := range [][]float32{hannF32(nfft), nil} {
				p, _ := NewSTFTPlan(nfft)
				frames := p.NumFrames(len(x), hop, pad)
				spec := make([][]complex64, frames)
				for f := range spec {
					spec[f] = make([]complex64, p.NumBins())
				}
				p.STFT(spec, x, window, hop, pad)
				y := make([]float32, len(x))
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
				tol := stftTolF32(nfft, 4)
				for i := range n {
					if wolaNormF32(window, nfft, hop, frames, i+off) < 1e-3 {
						continue
					}
					if d := math.Abs(float64(y[i] - x[i])); d > tol {
						t.Fatalf("pad=%v hop=%d win=%v sample %d: %g != %g (|diff|=%g tol=%g)", pad, hop, window != nil, i, y[i], x[i], d, tol)
					}
				}
			}
		}
	}
}

// TestISTFTGuardsF32 covers the degenerate inputs: no frames, bad hop, empty
// dst, a dst longer than the reconstructable length (clamped), and a short
// window treated as rectangular.
func TestISTFTGuardsF32(t *testing.T) {
	const nfft, hop = 64, 16
	p, _ := NewSTFTPlan(nfft)
	x := testSignalF32(512)
	frames := p.NumFrames(len(x), hop, NoPad)
	spec := make([][]complex64, frames)
	for f := range spec {
		spec[f] = make([]complex64, p.NumBins())
	}
	p.STFT(spec, x, nil, hop, NoPad)

	if n := p.ISTFT(make([]float32, 10), nil, nil, hop, NoPad); n != 0 {
		t.Fatalf("no frames wrote %d", n)
	}
	if n := p.ISTFT(make([]float32, 10), spec, nil, 0, NoPad); n != 0 {
		t.Fatalf("hop 0 wrote %d", n)
	}
	if n := p.ISTFT(nil, spec, nil, hop, NoPad); n != 0 {
		t.Fatalf("nil dst wrote %d", n)
	}
	long := make([]float32, 10000)
	if n := p.ISTFT(long, spec, nil, hop, NoPad); n != (frames-1)*hop+nfft {
		t.Fatalf("long dst wrote %d, want %d", n, (frames-1)*hop+nfft)
	}
	a := make([]float32, 512)
	b := make([]float32, 512)
	p.ISTFT(a, spec, nil, hop, NoPad)
	p.ISTFT(b, spec, hannF32(nfft/2), hop, NoPad)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("sample %d: short window %g != rectangular %g", i, b[i], a[i])
		}
	}
}

func TestISTFTAllocFreeF32(t *testing.T) {
	const nfft, hop = 512, 128
	p, _ := NewSTFTPlan(nfft)
	x := testSignalF32(8192)
	window := hannF32(nfft)
	frames := p.NumFrames(len(x), hop, PadZero)
	spec := make([][]complex64, frames)
	for f := range spec {
		spec[f] = make([]complex64, p.NumBins())
	}
	p.STFT(spec, x, window, hop, PadZero)
	y := make([]float32, len(x))
	if a := testing.AllocsPerRun(5, func() { p.ISTFT(y, spec, window, hop, PadZero) }); a != 0 {
		t.Errorf("ISTFT allocated %v times per run, want 0", a)
	}
}
