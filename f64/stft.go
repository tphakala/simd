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
// The transform is a radix-2 rfft (power-of-two nfft only); its butterfly stages
// run through ButterflyComplexStage, so they take the AVX+FMA / NEON vector paths
// where the span and block count justify them and fall back to scalar Go
// otherwise. See #108 and #205.

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

	// Per-stage contiguous twiddles for the size-half radix-2 FFT, so each stage
	// can be driven through ButterflyComplexStage (which reads its twiddles
	// contiguous in j over [0, span)). Stage m in {2,4,...,half} uses span = m/2
	// factors W_m^j = exp(-i*2*pi*j/m) for j in [0, span); the stage with span s
	// occupies stageTwRe[s-1 : 2*s-1], and the tables total half-1 entries.
	stageTwRe, stageTwIm []float64

	// Extra twiddle for the radix-4 FFT core: the w^(3j) power ButterflyComplexStage4
	// needs beyond the two it can slice from stageTwRe/stageTwIm (tw1 = w^(2j) is the
	// span-s radix-2 table, tw2 = w^j is the first s entries of the span-2s table).
	// The radix-4 stage with span s (s in {1,4,16,...}) occupies stage4Tw3Re[(s-1)/3 :
	// (s-1)/3 + s]; the offset (s-1)/3 is exact because s is a power of four. Empty
	// when half < 4 (no radix-4 stage runs).
	stage4Tw3Re, stage4Tw3Im []float64

	// Unravel twiddles W_N^k = exp(-i*2*pi*k/nfft) for k in [0, half], used to
	// recombine the even/odd half-spectra into the real-input spectrum.
	unRe, unIm []float64

	// Per-transform scratch (the packed complex frame, FFT'd in place).
	re, im []float64

	// Unravel scratch for STFT: RealFFTUnpack writes the split-complex bins
	// X[1..half-1] here before they are interleaved into the caller's complex128
	// row. Length half; index 0 is unused (DC and Nyquist are computed directly).
	outRe, outIm []float64

	// Window halves for the vector pack: winRe holds the even window samples
	// w[2j] and winIm the odd samples w[2j+1], split by prepareWindow once per
	// call so packFrame can apply the window with two in-place Mul calls.
	winRe, winIm []float64
}

// NumBins returns the number of output bins per frame, nfft/2 + 1 (the Hermitian
// half-spectrum, DC through Nyquist).
func (p *STFTPlan) NumBins() int { return p.half + 1 }

// NFFT returns the transform size the plan was built for.
func (p *STFTPlan) NFFT() int { return p.nfft }

// stage4Tw3Power is the twiddle power of the radix-4 stage's third factor: tw3 =
// w^(3j) (tw1 = w^(2j) and tw2 = w^(1j) are sliced from the radix-2 tables).
const stage4Tw3Power = 3

// NewSTFTPlan builds a reusable plan for nfft-point real-input STFTs. nfft must
// be a power of two and at least 2; otherwise ErrNotPowerOfTwo is returned.
func NewSTFTPlan(nfft int) (*STFTPlan, error) {
	if nfft < 2 || nfft&(nfft-1) != 0 {
		return nil, ErrNotPowerOfTwo
	}
	half := nfft >> 1

	// Size the radix-4 tw3 table: the radix-4 core runs stages at spans 1, 4, 16, ...
	// while 4*span <= half, and stage span s holds s entries, so they sum to
	// (4^numStages - 1)/3. Zero when half < 4.
	stage4Tw3Len := 0
	for s := 1; butterflyStage4Radix*s <= half; s *= butterflyStage4Radix {
		stage4Tw3Len += s
	}

	p := &STFTPlan{
		nfft:        nfft,
		half:        half,
		bitrev:      make([]int, half),
		stageTwRe:   make([]float64, max(half-1, 0)),
		stageTwIm:   make([]float64, max(half-1, 0)),
		stage4Tw3Re: make([]float64, stage4Tw3Len),
		stage4Tw3Im: make([]float64, stage4Tw3Len),
		unRe:        make([]float64, half+1),
		unIm:        make([]float64, half+1),
		re:          make([]float64, half),
		im:          make([]float64, half),
		outRe:       make([]float64, half),
		outIm:       make([]float64, half),
		winRe:       make([]float64, half),
		winIm:       make([]float64, half),
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

	// Per-stage contiguous FFT twiddles: stage m in {2,4,...,half} writes its
	// span = m/2 factors W_m^j = exp(-i*2*pi*j/m) starting at offset span-1.
	for m := 2; m <= half; m <<= 1 {
		span := m >> 1
		off := span - 1
		for j := range span {
			ang := 2 * math.Pi * float64(j) / float64(m)
			s, c := math.Sincos(ang)
			p.stageTwRe[off+j] = c
			p.stageTwIm[off+j] = -s
		}
	}

	// Radix-4 tw3 = w^(3j) with w = exp(-i*2*pi/(4*span)), for each radix-4 stage
	// span s in {1,4,16,...}. tw1 = w^(2j) and tw2 = w^j are the span-s and span-2s
	// radix-2 tables above, sliced in fftHalf; only w^(3j) is not already present.
	// The offset (s-1)/(radix4-1) is the exact running sum of the earlier stage
	// lengths because each s is a power of butterflyStage4Radix.
	for s := 1; butterflyStage4Radix*s <= half; s *= butterflyStage4Radix {
		off := (s - 1) / (butterflyStage4Radix - 1)
		for j := range s {
			ang := 2 * math.Pi * float64(stage4Tw3Power*j) / float64(butterflyStage4Radix*s)
			sin, cos := math.Sincos(ang)
			p.stage4Tw3Re[off+j] = cos
			p.stage4Tw3Im[off+j] = -sin
		}
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

// fftHalf runs an in-place size-half decimation-in-time complex FFT on the plan's
// scratch (p.re, p.im), using the resident bit-reversal and per-stage twiddles.
// The core is radix-4: each ButterflyComplexStage4 call advances two radix-2
// stages at once, with at most one trailing ButterflyComplexStage when half is
// not a power of four, so the butterflies take the AVX+FMA / NEON vector paths
// where the span and block count justify them.
func (p *STFTPlan) fftHalf() {
	re, im := p.re, p.im
	// Bit-reversal reorder.
	for i, j := range p.bitrev {
		if j > i {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	// Butterfly stages via the radix-4 core: a radix-4 stage at span s advances the
	// transform two radix-2 stages at once (span s then span 2s), so it runs spans
	// 1, 4, 16, ... while a full radix-4 block fits (4*s <= half). tw1 = w^(2j) is the
	// span-s radix-2 table at stageTw[s-1 : 2s-1], tw2 = w^j is the first s entries of
	// the span-2s table at stageTw[2s-1 : 3s-1], and tw3 = w^(3j) is the dedicated
	// stage4Tw3 table at [(s-1)/3 : (s-1)/3 + s]. A single trailing radix-2 stage
	// finishes the transform when half is not a power of four (odd log2(half)).
	s := 1
	for butterflyStage4Radix*s <= p.half {
		o1 := s - 1                                // span-s radix-2 table
		o2 := butterflyStageRadix*s - 1            // span-2s radix-2 table, first s taken
		o3 := (s - 1) / (butterflyStage4Radix - 1) // dedicated w^(3j) table
		ButterflyComplexStage4(re, im, s,
			p.stageTwRe[o1:o1+s], p.stageTwIm[o1:o1+s],
			p.stageTwRe[o2:o2+s], p.stageTwIm[o2:o2+s],
			p.stage4Tw3Re[o3:o3+s], p.stage4Tw3Im[o3:o3+s])
		s *= butterflyStage4Radix
	}
	if s < p.half {
		off := s - 1
		ButterflyComplexStage(re, im, s, p.stageTwRe[off:off+s], p.stageTwIm[off:off+s])
	}
}

// prepareWindow splits window (nil for rectangular, else length >= nfft) into its
// even and odd samples, winRe and winIm, so the interior-frame pack can apply it
// with two vector multiplies on the packed halves instead of a scalar
// multiply-and-deinterleave per sample. It runs once per STFT call, since the
// window is fixed across the call's frames, and is not needed for a nil window.
func (p *STFTPlan) prepareWindow(window []float64) {
	if window == nil {
		return
	}
	Deinterleave2(p.winRe, p.winIm, window[:p.nfft])
}

// packFrame loads frame f (signal[base : base+nfft]) into the scratch as half
// complex samples c[j] = x[2j] + i*x[2j+1]: a vector deinterleave of the frame,
// then, when windowed, an in-place vector multiply of each half by the window
// half prepareWindow split out (the same single-rounded products as a scalar
// x[n]*w[n] pack). window may be nil (rectangular). The caller guarantees the
// frame fits and that prepareWindow has run for this window.
func (p *STFTPlan) packFrame(signal, window []float64, base int) {
	re, im := p.re, p.im
	Deinterleave2(re, im, signal[base:base+p.nfft])
	if window == nil {
		return
	}
	Mul(re, re, p.winRe)
	Mul(im, im, p.winIm)
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

// unravelRow writes the real-input spectrum X[0..half] of the FFT'd frame in
// p.re/p.im into row as complex128 bins. A full-width row (len >= NumBins) takes
// the vector path: RealFFTUnpack computes X[1..half-1] into the plan's split
// scratch in one SIMD pass (its twiddle W[k] at index k-1 is exactly unRe/unIm
// shifted by one, since W_N^k = exp(-i*pi*k/half)), and DC and Nyquist, both
// real, come straight from C[0]. A shorter row keeps the per-bin scalar unravel,
// since the vector kernel writes every interior bin.
func (p *STFTPlan) unravelRow(row []complex128) {
	half := p.half
	if len(row) <= half {
		for k := range row {
			xr, xi := p.unravelBin(k)
			row[k] = complex(xr, xi)
		}
		return
	}
	row = row[:half+1]
	RealFFTUnpack(p.outRe, p.outIm, p.re, p.im, p.unRe[1:], p.unIm[1:])
	c0r, c0i := p.re[0], p.im[0]
	row[0] = complex(c0r+c0i, 0)
	outRe, outIm := p.outRe[:half], p.outIm[:half]
	for k := 1; k < half; k++ {
		row[k] = complex(outRe[k], outIm[k])
	}
	row[half] = complex(c0r-c0i, 0)
}

// unravelPowerRow is the power-spectrum counterpart of unravelRow: it writes
// |X[k]|^2 for k in [0, half] into row. A full-width row takes RealFFTPower,
// which fuses the unpack and the magnitude-squared into one SIMD pass over the
// interior bins and writes them in place; DC and Nyquist are squared directly.
// A shorter row keeps the per-bin scalar unravel.
func (p *STFTPlan) unravelPowerRow(row []float64) {
	half := p.half
	if len(row) <= half {
		for k := range row {
			xr, xi := p.unravelBin(k)
			row[k] = xr*xr + xi*xi
		}
		return
	}
	row = row[:half+1]
	RealFFTPower(row[:half], p.re, p.im, p.unRe[1:], p.unIm[1:])
	c0r, c0i := p.re[0], p.im[0]
	dc := c0r + c0i
	ny := c0r - c0i
	row[0] = dc * dc
	row[half] = ny * ny
}

// STFT computes the real-input STFT of signal and writes one Hermitian
// half-spectrum (NumBins complex128 values) per frame into dst. window, when
// non-nil, must have length nfft. The pad argument selects the framing
// convention: NoPad applies no padding (frame f is signal[f*hop : f*hop+nfft],
// matching librosa stft(..., center=False)); PadZero and PadReflect center each
// frame with nfft/2 of zero or reflect padding per side, matching librosa
// center=True. See PadMode and NumFrames.
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
	p.prepareWindow(window)
	off := 0
	if pad != NoPad {
		off = p.half // center: first sample of frame f is at f*hop - nfft/2
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		row := dst[f]
		p.unravelRow(row[:min(len(row), bins)])
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
	p.prepareWindow(window)
	off := 0
	if pad != NoPad {
		off = p.half
	}
	bins := p.NumBins()
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		row := dst[f]
		p.unravelPowerRow(row[:min(len(row), bins)])
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
	p.prepareWindow(window)
	off := 0
	if pad != NoPad {
		off = p.half
	}
	for f := range frames {
		p.packFrameAt(signal, window, f*hop-off, pad)
		p.fftHalf()
		// Each frame's stride is a full-width row, so it takes the vector unravel.
		p.unravelPowerRow(dst[f*bins : (f+1)*bins])
	}
	return frames
}
