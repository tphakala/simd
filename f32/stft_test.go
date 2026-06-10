package f32

import (
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
			nf := plan.numFrames(len(signal), hop)
			dst := make([][]complex64, nf)
			for f := range dst {
				dst[f] = make([]complex64, plan.NumBins())
			}
			got := plan.STFT(dst, signal, window, hop)
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
	nf := plan.numFrames(len(signal), hop)

	spec := make([][]complex64, nf)
	pow := make([][]float32, nf)
	for f := range spec {
		spec[f] = make([]complex64, plan.NumBins())
		pow[f] = make([]float32, plan.NumBins())
	}
	plan.STFT(spec, signal, window, hop)
	plan.STFTPower(pow, signal, window, hop)

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
	plan.STFT(dst, signal, nil, nfft)

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
	if got := plan.numFrames(len(signal), hop); got != wantFrames {
		t.Fatalf("numFrames = %d, want %d", got, wantFrames)
	}
	dst := make([][]complex64, wantFrames)
	for f := range dst {
		dst[f] = make([]complex64, plan.NumBins())
	}
	if n := plan.STFT(dst, signal, nil, hop); n != wantFrames {
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
	full := plan.numFrames(len(signal), hop)

	// Fewer rows than frames: only len(dst) frames written.
	short := make([][]complex64, full-2)
	for f := range short {
		short[f] = make([]complex64, plan.NumBins())
	}
	if n := plan.STFT(short, signal, nil, hop); n != full-2 {
		t.Errorf("clamped frames = %d, want %d", n, full-2)
	}

	// Rows shorter than NumBins: only the available bins written, no panic.
	rows := make([][]complex64, 1)
	rows[0] = make([]complex64, 3)
	if n := plan.STFT(rows, signal, nil, hop); n != 1 {
		t.Errorf("partial-row frames = %d, want 1", n)
	}
}

func TestSTFTAllocFreeF32(t *testing.T) {
	plan, _ := NewSTFTPlan(512)
	signal := testSignalF32(8192)
	window := hannF32(512)
	hop := 128
	nf := plan.numFrames(len(signal), hop)
	spec := make([][]complex64, nf)
	pow := make([][]float32, nf)
	for f := range spec {
		spec[f] = make([]complex64, plan.NumBins())
		pow[f] = make([]float32, plan.NumBins())
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFT(spec, signal, window, hop) }); a != 0 {
		t.Errorf("STFT allocated %v times per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() { plan.STFTPower(pow, signal, window, hop) }); a != 0 {
		t.Errorf("STFTPower allocated %v times per run, want 0", a)
	}
}

// FuzzSTFT is a differential fuzz target: every STFT bin must match a direct DFT
// of the windowed frame within the float32 tolerance, across fuzzed signal
// contents, nfft, hop, and window choice. Inputs are bounded to [-1, 1] (via
// f32sUnit) so the DFT bin magnitudes stay well-conditioned. Seeds run under
// plain `go test`; `go test -fuzz=FuzzSTFT` widens the space.
func FuzzSTFT(f *testing.F) {
	f.Add(make([]byte, 256), uint8(3), uint8(7), false)
	f.Add(make([]byte, 600), uint8(5), uint8(3), true)

	f.Fuzz(func(t *testing.T, raw []byte, nfftSel, hopSel uint8, useWin bool) {
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
		nf := plan.numFrames(len(signal), hop)
		if nf == 0 {
			return
		}
		dst := make([][]complex64, nf)
		for i := range dst {
			dst[i] = make([]complex64, plan.NumBins())
		}
		plan.STFT(dst, signal, window, hop)

		frame := make([]float32, nfft)
		for fr := range nf {
			base := fr * hop
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
				got := dst[fr][k]
				if d := math.Hypot(float64(real(got))-real(want), float64(imag(got))-imag(want)); d > tol {
					t.Fatalf("nfft=%d hop=%d frame=%d bin=%d: got %v want %v |diff|=%g tol=%g", nfft, hop, fr, k, got, want, d, tol)
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
	nf := plan.numFrames(len(signal), hop)
	dst := make([][]complex64, nf)
	for f := range dst {
		dst[f] = make([]complex64, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFT(dst, signal, window, hop)
	}
}

func BenchmarkSTFTPower(b *testing.B) {
	const nfft = 1024
	plan, _ := NewSTFTPlan(nfft)
	window := hannF32(nfft)
	signal := testSignalF32(48000)
	hop := 256
	nf := plan.numFrames(len(signal), hop)
	dst := make([][]float32, nf)
	for f := range dst {
		dst[f] = make([]float32, plan.NumBins())
	}
	b.ReportAllocs()
	for b.Loop() {
		plan.STFTPower(dst, signal, window, hop)
	}
}
