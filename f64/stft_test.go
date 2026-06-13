package f64

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
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
	for _, nfft := range []int{2, 4, 8, 16, 64, 256, 1024} {
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

// refSampleAt and stftRef independently re-implement centering, windowing, and
// the DFT (via dftBin), as a cross-check on the FFT-based centered STFT. They are
// deliberately a separate implementation from the package's sampleAt/packFrameAt
// so a bug in one does not mask a bug in the other.
func refSampleAt(signal []float64, idx int, pad PadMode) float64 {
	if idx >= 0 && idx < len(signal) {
		return signal[idx]
	}
	if pad == PadReflect {
		return signal[reflectIndex(idx, len(signal))]
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

// TestSTFTLibrosaParity pins the output convention against a golden vector
// generated by real librosa (see testdata/gen_stft_golden.py). This is the
// acceptance check that a model trained on librosa features accepts simd output.
func TestSTFTLibrosaParity(t *testing.T) {
	data, err := os.ReadFile("../testdata/stft_librosa_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		LibrosaVersion string    `json:"librosa_version"`
		NFFT           int       `json:"nfft"`
		Hop            int       `json:"hop"`
		Signal         []float64 `json:"signal"`
		Window         []float64 `json:"window"`
		Cases          []struct {
			GoPad  string    `json:"go_pad"`
			Frames int       `json:"frames"`
			Bins   int       `json:"bins"`
			Power  []float64 `json:"power"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	t.Logf("golden generated by librosa %s (nfft=%d hop=%d)", g.LibrosaVersion, g.NFFT, g.Hop)
	p, _ := NewSTFTPlan(g.NFFT)
	padOf := map[string]PadMode{"PadZero": PadZero, "PadReflect": PadReflect}
	for _, c := range g.Cases {
		pad, ok := padOf[c.GoPad]
		if !ok {
			t.Fatalf("unknown go_pad %q in golden", c.GoPad)
		}
		if nf := p.NumFrames(len(g.Signal), g.Hop, pad); nf != c.Frames {
			t.Fatalf("%s: NumFrames=%d but librosa produced %d frames", c.GoPad, nf, c.Frames)
		}
		if c.Bins != p.NumBins() {
			t.Fatalf("%s: golden bins=%d but NumBins=%d", c.GoPad, c.Bins, p.NumBins())
		}
		flat := make([]float64, c.Frames*c.Bins)
		p.STFTPowerInto(flat, g.Signal, g.Window, g.Hop, pad)
		var maxRel float64
		for i := range flat {
			ref := c.Power[i]
			rel := math.Abs(flat[i]-ref) / (math.Abs(ref) + 1e-12)
			if rel > maxRel {
				maxRel = rel
			}
		}
		// librosa uses pocketfft and we use a radix-2 rfft, so the bins differ at
		// the float64 algorithm-noise level (~5e-8 relative; squaring to power
		// roughly doubles the amplitude error). A convention error (wrong
		// centering, window, or normalization) would be orders of magnitude
		// larger, so 1e-6 cleanly separates "matches librosa" from "wrong".
		if maxRel > 1e-6 {
			t.Errorf("%s: max relative error %g exceeds 1e-6 vs librosa", c.GoPad, maxRel)
		}
	}
}
