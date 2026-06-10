package f32

import (
	"errors"
	"math"
)

// This file implements a fused, real-input Short-Time Fourier Transform for
// float32, mirroring the f64 STFTPlan (see f64/stft.go). The transform is the
// missing middle of a spectral feature pipeline: the library already covers
// windowing inputs, the post-FFT power spectrum (c64.AbsSq), mel projection
// (DotProductBatch), and PCEN/log-mel normalization (Exp/Mul/Log), but not the
// FFT itself.
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
// The transform runs in float32 to match the rest of the f32 package; the
// twiddle and unravel tables are computed in float64 and rounded once to float32
// so the resident constants carry full precision. This first cut is a correct
// scalar radix-2 transform (power-of-two nfft only); vectorizing the inner
// butterfly is a separate, profile-gated change. See #108.

// ErrSTFT* describe invalid STFTPlan configurations.
var (
	// ErrNotPowerOfTwo is returned when nfft is not a power of two >= 2.
	ErrNotPowerOfTwo = errors.New("f32: STFT nfft must be a power of two >= 2")
)

// rfftHalf is the 1/2 factor in the real-FFT even/odd half-spectrum split.
const rfftHalf = 0.5

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
	twRe, twIm []float32

	// Unravel twiddles W_N^k = exp(-i*2*pi*k/nfft) for k in [0, half], used to
	// recombine the even/odd half-spectra into the real-input spectrum.
	unRe, unIm []float32

	// Per-transform scratch (the packed complex frame, FFT'd in place).
	re, im []float32
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
		twRe:   make([]float32, max(half>>1, 1)),
		twIm:   make([]float32, max(half>>1, 1)),
		unRe:   make([]float32, half+1),
		unIm:   make([]float32, half+1),
		re:     make([]float32, half),
		im:     make([]float32, half),
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

	// Size-half FFT twiddles (computed in float64, stored as float32).
	for t := range p.twRe {
		ang := 2 * math.Pi * float64(t) / float64(half)
		s, c := math.Sincos(ang)
		p.twRe[t] = float32(c)
		p.twIm[t] = float32(-s)
	}

	// Real-input unravel twiddles W_N^k.
	for k := 0; k <= half; k++ {
		ang := 2 * math.Pi * float64(k) / float64(nfft)
		s, c := math.Sincos(ang)
		p.unRe[k] = float32(c)
		p.unIm[k] = float32(-s)
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

// packFrame loads frame f (signal[base : base+nfft]) into the scratch as half
// complex samples c[j] = x[2j] + i*x[2j+1], applying the window during the pack.
// window may be nil (rectangular). The caller guarantees the frame fits.
func (p *STFTPlan) packFrame(signal, window []float32, base int) {
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

// numFrames returns how many full, non-centered frames of nfft samples fit in
// signal at the given hop.
func (p *STFTPlan) numFrames(signalLen, hop int) int {
	if signalLen < p.nfft || hop <= 0 {
		return 0
	}
	return 1 + (signalLen-p.nfft)/hop
}

// unravelBin computes the real-input spectrum bin X[k] (k in [0, half]) from the
// half-length complex FFT result currently in p.re/p.im, returning (re, im).
func (p *STFTPlan) unravelBin(k int) (re, im float32) {
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
// half-spectrum (NumBins complex64 values) per frame into dst. Frame f is
// signal[f*hop : f*hop+nfft], windowed by window when non-nil (which must have
// length nfft). This is the center=false / no-padding convention (matching
// librosa stft(..., center=False)); pre-pad the signal yourself for centered
// frames.
//
// It writes min(len(dst), numFrames) frames and, per frame, min(len(dst[f]),
// NumBins) bins, and returns the number of frames written. It is allocation-free
// and reuses the plan scratch.
func (p *STFTPlan) STFT(dst [][]complex64, signal, window []float32, hop int) int {
	frames := min(p.numFrames(len(signal), hop), len(dst))
	if frames == 0 {
		return 0
	}
	if window != nil && len(window) < p.nfft {
		// Treat a short window as rectangular rather than panicking, matching the
		// library's lenient public-API style.
		window = nil
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrame(signal, window, f*hop)
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
// materialization of the complex bins. dst, signal, window, and hop follow the
// same conventions as STFT. Returns the number of frames written. Allocation-free.
func (p *STFTPlan) STFTPower(dst [][]float32, signal, window []float32, hop int) int {
	frames := min(p.numFrames(len(signal), hop), len(dst))
	if frames == 0 {
		return 0
	}
	if window != nil && len(window) < p.nfft {
		window = nil
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrame(signal, window, f*hop)
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
