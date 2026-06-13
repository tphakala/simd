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

// stftTolF32 bounds the error of an nfft-point float32 radix-2 transform against
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
	for _, nfft := range []int{2, 4, 8, 16, 64, 256, 1024} {
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
		// float32 radix-2 rfft vs librosa's float64 pocketfft: the error is
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
