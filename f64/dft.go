package f64

import (
	"errors"
	"math"
	"math/bits"
)

// ErrDFTCapacity is returned by NewDFTPlan when capacity is not in [1, 1<<28].
var ErrDFTCapacity = errors.New("f64: DFT capacity must be in [1, 1<<28]")

// maxDFTCapacity is the largest capacity NewDFTPlan accepts. It keeps the inner
// power-of-two size nextPow2(2*capacity-1) within maxFFTSize.
const maxDFTCapacity = maxFFTSize / 2

// DFTPlan is a complex DFT of any length n in [1, Capacity()], in place on split
// real and imaginary slices. A power-of-two n runs the radix-4 core of FFTPlan
// directly and gives exactly the bits FFTPlan gives for that n. Any other n uses
// Bluestein's chirp-z algorithm, X[k] = c[k] * sum_j (x[j]*c[j]) * conj(c[k-j])
// with c[t] = exp(-i*pi*t^2/n), evaluated as a circular convolution through
// power-of-two FFTs of size M = nextpow2(2n-1). Forward computes
// X[k] = sum_j x[j]*exp(-2*pi*i*j*k/n) and Inverse the conjugate transform
// x[j] = sum_k X[k]*exp(+2*pi*i*j*k/n), unscaled: Inverse(Forward(x)) equals n*x,
// where numpy.fft.ifft divides by n.
//
// The result for a given n depends only on n and the CPU tier: not on Capacity()
// and not on the lengths transformed before. Results are tolerance-stable, not
// bit-stable, across CPU tiers (see FFTPlan).
//
// The first call with a non-power-of-two n that differs from the previous one
// retargets the plan: n math.Sincos calls for the chirp plus one size-M FFT of
// the convolution kernel, into preallocated scratch, with no allocation.
// BenchmarkDFTPlanRetarget on an Apple M5 Pro measures about 10 us at n = 1000,
// 0.8 ms at 32767 and 1.5 ms at 32769, which is 0.6 to 0.9 of a Forward call at
// the same n. A power-of-two n does not retarget and leaves the target as it was. A
// caller that alternates between two non-power-of-two lengths retargets on every
// call (2.9 ms per call for 32767 and 32769 in BenchmarkDFTPlanForwardRetarget,
// against 1.1 ms and 2.4 ms for a Forward call at a settled target) and should
// use one plan per length.
//
// The chirp phase t^2 mod 2n is reduced in 64-bit integer arithmetic before it is
// scaled by pi/n, so it is exact for every allowed n, on 32-bit targets too.
//
// A DFTPlan is NOT safe for concurrent use: it holds scratch and the current
// target. Use one plan per goroutine. re and im must not overlap each other.
//
// Memory, with Mc = nextpow2(2*capacity-1), is exactly core(Mc) + 16*capacity +
// 32*Mc bytes, where core(Mc) = 4*Mc + 16*(Mc-1) + 16*(4^q-1)/3 and
// q = floor(log2(Mc)/2) are the FFTPlan tables: 2*capacity float64 of chirp and
// 4*Mc float64 of kernel and work buffers on top of them. That is about
// 57.3*Mc + 16*capacity bytes when log2(Mc) is even and 54.7*Mc + 16*capacity when
// it is odd: 1,004,880 bytes at capacity 4097, 1,922,384 at 8193 and 7,689,552 at
// 32769. Building a plan costs the table construction of an FFTPlan for Mc, about
// 4*Mc/3 math.Sincos calls plus O(Mc) table fill; BenchmarkNewDFTPlan on an Apple
// M5 Pro measures 0.12 ms at capacity 4097 and 0.66 ms at 32769.
type DFTPlan struct {
	core     fftCore // tables for Mc = nextPow2(2*capacity-1)
	capacity int

	chirpRe, chirpIm []float64 // len capacity: c[t] = exp(-i*pi*t^2/n) for the current n
	kerRe, kerIm     []float64 // len Mc: FFT_M(conj c, wrapped) / M for the current n
	workRe, workIm   []float64 // len Mc: per-call convolution buffer

	// n and m are the current Bluestein target and its inner size; both are 0
	// until the first non-power-of-two call.
	n, m int
}

// NewDFTPlan builds a plan for transforms of every length in [1, capacity].
// capacity must be in [1, 1<<28]; otherwise ErrDFTCapacity is returned. No length
// is targeted until the first non-power-of-two Forward.
func NewDFTPlan(capacity int) (*DFTPlan, error) {
	if capacity < 1 || capacity > maxDFTCapacity {
		return nil, ErrDFTCapacity
	}
	mc := nextPow2(2*capacity - 1)
	return &DFTPlan{
		core:     newFFTCore(mc),
		capacity: capacity,
		chirpRe:  make([]float64, capacity),
		chirpIm:  make([]float64, capacity),
		kerRe:    make([]float64, mc),
		kerIm:    make([]float64, mc),
		workRe:   make([]float64, mc),
		workIm:   make([]float64, mc),
	}, nil
}

// Capacity returns the largest transform length the plan serves.
func (p *DFTPlan) Capacity() int { return p.capacity }

// Forward computes the DFT of (re, im) in place, X[k] = sum_j x[j]*exp(-2*pi*i*j*k/n)
// with n = len(re), and returns n. It returns 0 and leaves both slices untouched
// when n is 0, len(im) != n, or n > Capacity(). It does not panic on such input,
// so a caller that ignores the return value gets untransformed data. re and im
// must not overlap. Allocation-free, including a retarget.
func (p *DFTPlan) Forward(re, im []float64) int {
	n := len(re)
	if n == 0 || len(im) != n || n > p.capacity {
		return 0
	}
	if n&(n-1) == 0 {
		p.core.transform(re, im)
		return n
	}
	if n != p.n {
		p.retarget(n)
	}
	m := p.m
	cr, ci := p.chirpRe[:n], p.chirpIm[:n]
	wr, wi := p.workRe[:m], p.workIm[:m]

	// a[j] = x[j]*c[j], zero padded to m. All of the input is consumed here, so the
	// output can be written back into re and im at the end.
	complexMul(wr[:n], wi[:n], re, im, cr, ci)
	clear(wr[n:])
	clear(wi[n:])
	p.core.transform(wr, wi)
	complexMul(wr, wi, wr, wi, p.kerRe[:m], p.kerIm[:m])
	// Unscaled inverse by the swap identity (see FFTPlan.Inverse). The 1/m is
	// folded into the kernel, so the first n entries are the linear convolution
	// because m >= 2n-1.
	p.core.transform(wi, wr)
	complexMul(re, im, wr[:n], wi[:n], cr, ci)
	return n
}

// Inverse computes the unscaled inverse DFT of (re, im) in place,
// x[j] = sum_k X[k]*exp(+2*pi*i*j*k/n) with n = len(re), and returns n; divide by
// n for the normalized inverse. The length rules, the zero return, the retarget
// behavior and the no-overlap rule are those of Forward.
func (p *DFTPlan) Inverse(re, im []float64) int { return p.Forward(im, re) }

// retarget points the plan at Bluestein length n, which must be at least 3 and
// not a power of two. It rewrites chirp[:n] and ker[:m] completely, so the plan
// state is a function of n alone.
func (p *DFTPlan) retarget(n int) {
	m := nextPow2(2*n - 1)
	p.n, p.m = n, m

	kr, ki := p.kerRe[:m], p.kerIm[:m]
	clear(kr)
	clear(ki)

	// c[t] = exp(-i*pi*t^2/n). t^2 mod 2n is reduced exactly before the angle is
	// formed; t < 2^28 keeps t^2 inside uint64. The kernel is conj(c[t]) at
	// indices t and m-t, so the circular convolution over m points sees
	// conj(c[k-j]) for every k-j in (-n, n).
	period := uint64(n) + uint64(n)
	for t := range n {
		r := uint64(t) * uint64(t) % period
		s, c := math.Sincos(math.Pi * float64(r) / float64(n))
		p.chirpRe[t], p.chirpIm[t] = c, -s
		kr[t], ki[t] = c, s
		if t > 0 {
			kr[m-t], ki[m-t] = c, s
		}
	}
	p.core.transform(kr, ki)
	// 1/m is exact because m is a power of two.
	Scale(kr, kr, 1/float64(m))
	Scale(ki, ki, 1/float64(m))
}

// complexMul sets dst = a*b elementwise on split complex slices of equal length.
// dst may be a exactly (element for element); it must not partly overlap a or b.
func complexMul(dstRe, dstIm, aRe, aIm, bRe, bIm []float64) {
	n := len(dstRe)
	dstIm, aRe, aIm, bRe, bIm = dstIm[:n], aRe[:n], aIm[:n], bRe[:n], bIm[:n]
	for i := range n {
		ar, ai := aRe[i], aIm[i]
		br, bi := bRe[i], bIm[i]
		dstRe[i] = ar*br - ai*bi
		dstIm[i] = ar*bi + ai*br
	}
}

// nextPow2 returns the smallest power of two that is at least x, for x >= 1.
func nextPow2(x int) int { return 1 << bits.Len(uint(x-1)) }
