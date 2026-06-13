package f64

import (
	"errors"
	"math"
)

// This file implements a fused, real-input Short-Time Fourier Transform. The
// transform is the missing middle of a spectral feature pipeline: the library
// already covers windowing inputs, the post-FFT power spectrum (c128.AbsSq), mel
// projection (DotProductBatch), and PCEN/log-mel normalization (Exp/Mul/Log),
// but not the FFT itself.
//
// Design (matches the rest of the library's batched primitives such as
// DotProductBatch / ConvolveValidMulti):
//
//   - Real input via a half-length complex FFT (rfft): an N-point real transform
//     is computed as an N/2-point complex FFT plus an O(N) unravel, ~2x cheaper
//     than a full complex FFT and producing the Hermitian half-spectrum
//     (N/2+1 bins) that librosa/scipy return.
//   - Batched framing: one call emits every hop-spaced frame, with the twiddle
//     tables and bit-reversal plan resident in the STFTPlan, so there is no
//     per-frame setup or dispatch.
//   - Fused window + (optional) power: the analysis window is applied while the
//     frame is packed into the FFT input, and STFTPower emits |X|^2 directly
//     without materializing the complex bins.
//
// This first cut is a correct scalar radix-2 transform (power-of-two nfft only);
// vectorizing the inner butterfly is a separate, profile-gated change. See #108.

// ErrSTFT* describe invalid STFTPlan configurations.
var (
	// ErrNotPowerOfTwo is returned when nfft is not a power of two >= 2.
	ErrNotPowerOfTwo = errors.New("f64: STFT nfft must be a power of two >= 2")
)

// rfftHalf is the 1/2 factor in the real-FFT even/odd half-spectrum split.
const rfftHalf = 0.5

// PadMode selects the STFT framing/centering convention.
//
//   - NoPad: center=false. Frame f is signal[f*hop : f*hop+nfft] with no
//     padding (the original convention; matches librosa stft(center=False)).
//   - PadZero: center=true with nfft/2 zero (constant) padding on each side.
//     This matches librosa's modern default (pad_mode="constant" since 0.8.0).
//   - PadReflect: center=true with nfft/2 reflect padding on each side (numpy
//     "reflect" semantics, where edge samples are not repeated; this was
//     librosa's pre-0.8.0 default pad_mode).
//
// Padding implies centering: the first centered frame is centered on sample 0.
// The pad mode is always explicit because librosa's default pad_mode has changed
// across versions, and getting centering subtly wrong shifts every frame.
type PadMode int

// Pad modes for STFT framing; see PadMode.
const (
	NoPad PadMode = iota
	PadZero
	PadReflect
)

// STFTPlan holds the resident twiddle tables, bit-reversal permutation, and
// transform scratch for a fixed nfft. Build one with NewSTFTPlan and reuse it
// across many STFT/STFTPower calls to stay allocation-free.
//
// A plan holds per-transform scratch, so its methods are NOT safe for concurrent
// use on the same plan; use one plan per goroutine (plans are cheap to create
// and the underlying tables are small). Distinct plans share no state.
type STFTPlan struct {
	nfft int // transform size (power of two)
	half int // nfft / 2: size of the packed complex FFT

	bitrev []int // bit-reversal permutation for the size-half FFT

	// Twiddles for the size-half radix-2 FFT: twRe[t] = cos(2*pi*t/half),
	// twIm[t] = -sin(2*pi*t/half) for t in [0, half/2).
	twRe, twIm []float64

	// Unravel twiddles W_N^k = exp(-i*2*pi*k/nfft) for k in [0, half], used to
	// recombine the even/odd half-spectra into the real-input spectrum.
	unRe, unIm []float64

	// Per-transform scratch (the packed complex frame, FFT'd in place).
	re, im []float64
}

// NumBins returns the number of output bins per frame, nfft/2 + 1 (the Hermitian
// half-spectrum, DC through Nyquist).
func (p *STFTPlan) NumBins() int { return p.half + 1 }

// NFFT returns the transform size the plan was built for.
func (p *STFTPlan) NFFT() int { return p.nfft }

// NewSTFTPlan builds a reusable plan for nfft-point real-input STFTs. nfft must
// be a power of two and at least 2; otherwise ErrNotPowerOfTwo is returned.
func NewSTFTPlan(nfft int) (*STFTPlan, error) {
	if nfft < 2 || nfft&(nfft-1) != 0 {
		return nil, ErrNotPowerOfTwo
	}
	half := nfft >> 1

	p := &STFTPlan{
		nfft:   nfft,
		half:   half,
		bitrev: make([]int, half),
		twRe:   make([]float64, max(half>>1, 1)),
		twIm:   make([]float64, max(half>>1, 1)),
		unRe:   make([]float64, half+1),
		unIm:   make([]float64, half+1),
		re:     make([]float64, half),
		im:     make([]float64, half),
	}

	// Bit-reversal permutation for a size-half FFT.
	logHalf := 0
	for (1 << logHalf) < half {
		logHalf++
	}
	for i := range p.bitrev {
		r := 0
		for b := range logHalf {
			r |= ((i >> b) & 1) << (logHalf - 1 - b)
		}
		p.bitrev[i] = r
	}

	// Size-half FFT twiddles.
	for t := range p.twRe {
		ang := 2 * math.Pi * float64(t) / float64(half)
		s, c := math.Sincos(ang)
		p.twRe[t] = c
		p.twIm[t] = -s
	}

	// Real-input unravel twiddles W_N^k.
	for k := 0; k <= half; k++ {
		ang := 2 * math.Pi * float64(k) / float64(nfft)
		s, c := math.Sincos(ang)
		p.unRe[k] = c
		p.unIm[k] = -s
	}

	return p, nil
}

// fftHalf runs an in-place size-half radix-2 decimation-in-time complex FFT on
// the plan's scratch (p.re, p.im), using the resident bit-reversal and twiddles.
func (p *STFTPlan) fftHalf() {
	re, im := p.re, p.im
	// Bit-reversal reorder.
	for i, j := range p.bitrev {
		if j > i {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	// Butterfly stages.
	for m := 2; m <= p.half; m <<= 1 {
		halfM := m >> 1
		step := p.half / m // twiddle stride into twRe/twIm
		for k := 0; k < p.half; k += m {
			for j := range halfM {
				idx := j * step
				wr, wi := p.twRe[idx], p.twIm[idx]
				a := k + j
				b := a + halfM
				vr := wr*re[b] - wi*im[b]
				vi := wr*im[b] + wi*re[b]
				re[b] = re[a] - vr
				im[b] = im[a] - vi
				re[a] += vr
				im[a] += vi
			}
		}
	}
}

// packFrame loads frame f (signal[f*hop : f*hop+nfft]) into the scratch as half
// complex samples c[j] = x[2j] + i*x[2j+1], applying the window during the pack.
// window may be nil (rectangular). The caller guarantees the frame fits.
func (p *STFTPlan) packFrame(signal, window []float64, base int) {
	re, im := p.re, p.im
	if window == nil {
		for j := range p.half {
			re[j] = signal[base+2*j]
			im[j] = signal[base+2*j+1]
		}
		return
	}
	for j := range p.half {
		re[j] = signal[base+2*j] * window[2*j]
		im[j] = signal[base+2*j+1] * window[2*j+1]
	}
}

// NumFrames reports how many frames a call with the given signal length, hop,
// and pad mode will write, so callers can size dst (or a flat STFTPowerInto
// buffer) exactly.
//
//	NoPad:              1 + (signalLen-nfft)/hop, or 0 if signalLen < nfft
//	PadZero/PadReflect: 1 + signalLen/hop,        or 0 if signalLen <= 0
//
// The centered count (1 + signalLen/hop for even nfft) matches librosa's
// stft(center=True) framing.
func (p *STFTPlan) NumFrames(signalLen, hop int, pad PadMode) int {
	if hop <= 0 {
		return 0
	}
	if pad == NoPad {
		if signalLen < p.nfft {
			return 0
		}
		return 1 + (signalLen-p.nfft)/hop
	}
	if signalLen <= 0 {
		return 0
	}
	return 1 + signalLen/hop
}

// reflectIndex maps an out-of-range index into [0,n) using numpy "reflect"
// semantics (edge samples are not repeated), folding with period 2*(n-1) so it
// is correct for arbitrary pad widths. n must be >= 1.
func reflectIndex(idx, n int) int {
	if n == 1 {
		return 0
	}
	period := (n - 1) << 1 // 2*(n-1): the period of the reflection
	m := idx % period
	if m < 0 {
		m += period
	}
	if m < n {
		return m
	}
	return period - m
}

// sampleAt reads signal[idx], substituting the pad value when idx is out of
// range. NoPad never reaches the out-of-range branch (callers keep NoPad frames
// in bounds); out-of-range with anything but PadReflect yields zero.
func sampleAt(signal []float64, idx int, pad PadMode) float64 {
	if idx >= 0 && idx < len(signal) {
		return signal[idx]
	}
	if pad == PadReflect {
		return signal[reflectIndex(idx, len(signal))]
	}
	return 0
}

// packFrameAt packs the frame whose first sample is at source index base (which
// may be negative for centered frames) into the scratch, applying the window and
// pad mode. A frame fully inside the signal uses the fast packFrame path; only
// edge frames pay for the bounds-aware sampleAt reads, so the common interior
// frame is unaffected.
func (p *STFTPlan) packFrameAt(signal, window []float64, base int, pad PadMode) {
	if base >= 0 && base+p.nfft <= len(signal) {
		p.packFrame(signal, window, base)
		return
	}
	re, im := p.re, p.im
	for j := range p.half {
		s0 := sampleAt(signal, base+2*j, pad)
		s1 := sampleAt(signal, base+2*j+1, pad)
		if window == nil {
			re[j], im[j] = s0, s1
		} else {
			re[j], im[j] = s0*window[2*j], s1*window[2*j+1]
		}
	}
}

// unravelBin computes the real-input spectrum bin X[k] (k in [0, half]) from the
// half-length complex FFT result currently in p.re/p.im, returning (re, im).
func (p *STFTPlan) unravelBin(k int) (re, im float64) {
	// k runs 0..half inclusive; the half-size spectrum C wraps at p.half,
	// so both k == 0 and k == p.half read C[0]. Branch instead of modulo
	// to keep integer division off this per-bin path.
	ck, cm := 0, 0
	if k > 0 && k < p.half {
		ck, cm = k, p.half-k
	}
	ckr, cki := p.re[ck], p.im[ck]
	cmr, cmi := p.re[cm], p.im[cm]

	// Even/odd half-spectra: E = 0.5*(C[k] + conj(C[half-k])),
	// O = -0.5i*(C[k] - conj(C[half-k])).
	er := rfftHalf * (ckr + cmr)
	ei := rfftHalf * (cki - cmi)
	or := rfftHalf * (cki + cmi)
	oi := -rfftHalf * (ckr - cmr)

	// X[k] = E + W_N^k * O.
	wr, wi := p.unRe[k], p.unIm[k]
	re = er + (wr*or - wi*oi)
	im = ei + (wr*oi + wi*or)
	return re, im
}

// STFT computes the real-input STFT of signal and writes one Hermitian
// half-spectrum (NumBins complex128 values) per frame into dst. Frame f is
// signal[f*hop : f*hop+nfft], windowed by window when non-nil (which must have
// length nfft). This is the center=false / no-padding convention (matching
// librosa stft(..., center=False)); pre-pad the signal yourself for centered
// frames.
//
// It writes min(len(dst), NumFrames) frames and, per frame, min(len(dst[f]),
// NumBins) bins, and returns the number of frames written. It is allocation-free
// and reuses the plan scratch.
func (p *STFTPlan) STFT(dst [][]complex128, signal, window []float64, hop int, pad PadMode) int {
	frames := min(p.NumFrames(len(signal), hop, pad), len(dst))
	if frames == 0 {
		return 0
	}
	if window != nil && len(window) < p.nfft {
		// Treat a short window as rectangular rather than panicking, matching the
		// library's lenient public-API style.
		window = nil
	}
	off := 0
	if pad != NoPad {
		off = p.half // center: first sample of frame f is at f*hop - nfft/2
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		row := dst[f]
		nb := min(len(row), bins)
		for k := range nb {
			xr, xi := p.unravelBin(k)
			row[k] = complex(xr, xi)
		}
	}
	return frames
}

// STFTPower computes the real-input STFT power spectrum |X|^2 directly, skipping
// materialization of the complex bins. dst, signal, window, hop, and pad follow
// the same conventions as STFT. Returns the number of frames written.
// Allocation-free.
func (p *STFTPlan) STFTPower(dst [][]float64, signal, window []float64, hop int, pad PadMode) int {
	frames := min(p.NumFrames(len(signal), hop, pad), len(dst))
	if frames == 0 {
		return 0
	}
	if window != nil && len(window) < p.nfft {
		window = nil
	}
	off := 0
	if pad != NoPad {
		off = p.half
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		row := dst[f]
		nb := min(len(row), bins)
		for k := range nb {
			xr, xi := p.unravelBin(k)
			row[k] = xr*xr + xi*xi
		}
	}
	return frames
}

// STFTPowerInto computes the real-input STFT power spectrum |X|^2 frame by frame
// into a single flat buffer, frame-contiguous with stride NumBins(): the bins of
// frame f occupy dst[f*NumBins : (f+1)*NumBins], ready to pass as the vec argument
// to DotProductBatch for a mel-filterbank projection. signal, window, hop, and
// pad follow the same conventions as STFTPower. It writes
// min(NumFrames, len(dst)/NumBins) whole frames and returns that frame count.
// Allocation-free.
func (p *STFTPlan) STFTPowerInto(dst, signal, window []float64, hop int, pad PadMode) int {
	bins := p.NumBins()
	frames := p.NumFrames(len(signal), hop, pad)
	if fit := len(dst) / bins; fit < frames {
		frames = fit
	}
	if frames == 0 {
		return 0
	}
	if window != nil && len(window) < p.nfft {
		window = nil
	}
	off := 0
	if pad != NoPad {
		off = p.half
	}
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		base := f * bins
		for k := range bins {
			xr, xi := p.unravelBin(k)
			dst[base+k] = xr*xr + xi*xi
		}
	}
	return frames
}
