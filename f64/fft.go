package f64

import (
	"math"
	"math/bits"
)

// This file holds the power-of-two complex FFT core that STFTPlan runs on its
// packed half-length frame: a bit-reversal table, per-stage twiddles and a radix-4
// driver over ButterflyComplexStage4 with at most one trailing radix-2
// ButterflyComplexStage. The tables are prefixes that do not depend on the
// transform size, so one core built for maxN serves every power of two n <= maxN.

// maxFFTSize is the largest power-of-two transform size the core accepts. It keeps
// the table-building and driver loops inside a 32-bit int: the radix-2 stage loop's
// m <<= 1 runs to 2*maxN and the radix-4 loops evaluate 4*s past maxN, so at 1<<30
// they would wrap on 32-bit targets and never end. It also keeps the int32 bit
// reversal table and the 32-bit bits.Reverse32 index inside range.
const maxFFTSize = 1 << 29

// uint32Bits is the width of the bits.Reverse32 index the bit-reversal table is
// built from.
const uint32Bits = 32

// stage4Tw3Power is the twiddle power of the radix-4 stage's third factor: tw3 =
// w^(3j) (tw1 = w^(2j) and tw2 = w^(1j) are sliced from the radix-2 tables).
const stage4Tw3Power = 3

// fftCore holds the resident tables of a decimation-in-time complex FFT for every
// power-of-two size up to maxN. It is read-only after newFFTCore, so transform
// keeps no scratch and may run concurrently on distinct buffers.
type fftCore struct {
	maxN   int // largest transform size (power of two, >= 1)
	logMax int // log2(maxN)

	// Bit-reversal permutation for a size-maxN FFT. The permutation for a smaller
	// size n = maxN >> sh is bitrev[i] >> sh for i in [0, n).
	bitrev []int32

	// Per-stage contiguous twiddles, laid out per radix-2 stage: stage m in
	// {2,4,...,maxN} uses span = m/2 factors W_m^j = exp(-i*2*pi*j/m) for j in
	// [0, span); the stage with span s occupies stageTwRe[s-1 : 2*s-1], and the
	// tables total maxN-1 entries. transform slices them as the tw1/tw2 inputs of
	// ButterflyComplexStage4 and as the twiddles of the trailing
	// ButterflyComplexStage (both read contiguous j over [0, span)).
	stageTwRe, stageTwIm []float64

	// Extra twiddle for the radix-4 core: the w^(3j) power ButterflyComplexStage4
	// needs beyond the two it can slice from stageTwRe/stageTwIm (tw1 = w^(2j) is
	// the span-s radix-2 table, tw2 = w^j is the first s entries of the span-2s
	// table). The radix-4 stage with span s (s in {1,4,16,...}) occupies
	// stage4Tw3Re[(s-1)/3 : (s-1)/3 + s]; the offset (s-1)/3 is exact because s is
	// a power of four. Empty when maxN < 4 (no radix-4 stage runs).
	stage4Tw3Re, stage4Tw3Im []float64
}

// newFFTCore builds the tables for sizes up to maxN. maxN must be a power of two
// in [1, maxFFTSize].
func newFFTCore(maxN int) fftCore {
	// Size the radix-4 tw3 table: the radix-4 core runs stages at spans 1, 4, 16, ...
	// while 4*span <= maxN, and stage span s holds s entries, so they sum to
	// (4^numStages - 1)/3. Zero when maxN < 4.
	stage4Tw3Len := 0
	for s := 1; butterflyStage4Radix*s <= maxN; s *= butterflyStage4Radix {
		stage4Tw3Len += s
	}

	logMax := bits.TrailingZeros(uint(maxN))
	c := fftCore{
		maxN:        maxN,
		logMax:      logMax,
		bitrev:      make([]int32, maxN),
		stageTwRe:   make([]float64, max(maxN-1, 0)),
		stageTwIm:   make([]float64, max(maxN-1, 0)),
		stage4Tw3Re: make([]float64, stage4Tw3Len),
		stage4Tw3Im: make([]float64, stage4Tw3Len),
	}

	// Bit-reversal permutation for a size-maxN FFT (the single entry 0 when maxN
	// is 1).
	if logMax > 0 {
		for i := range c.bitrev {
			c.bitrev[i] = int32(bits.Reverse32(uint32(i)) >> (uint32Bits - logMax))
		}
	}

	// Per-stage contiguous FFT twiddles: stage m in {2,4,...,maxN} writes its
	// span = m/2 factors W_m^j = exp(-i*2*pi*j/m) starting at offset span-1.
	for m := 2; m <= maxN; m <<= 1 {
		span := m >> 1
		off := span - 1
		for j := range span {
			ang := 2 * math.Pi * float64(j) / float64(m)
			s, cos := math.Sincos(ang)
			c.stageTwRe[off+j] = cos
			c.stageTwIm[off+j] = -s
		}
	}

	// Radix-4 tw3 = w^(3j) with w = exp(-i*2*pi/(4*span)), for each radix-4 stage
	// span s in {1,4,16,...}. tw1 = w^(2j) and tw2 = w^j are the span-s and span-2s
	// radix-2 tables above, sliced in transform; only w^(3j) is not already present.
	// The offset (s-1)/(radix4-1) is the exact running sum of the earlier stage
	// lengths because each s is a power of butterflyStage4Radix.
	for s := 1; butterflyStage4Radix*s <= maxN; s *= butterflyStage4Radix {
		off := (s - 1) / (butterflyStage4Radix - 1)
		for j := range s {
			ang := 2 * math.Pi * float64(stage4Tw3Power*j) / float64(butterflyStage4Radix*s)
			sin, cos := math.Sincos(ang)
			c.stage4Tw3Re[off+j] = cos
			c.stage4Tw3Im[off+j] = -sin
		}
	}
	return c
}

// transform runs an in-place decimation-in-time complex FFT on (re, im). Both
// must have the same length n, a power of two no larger than c.maxN; the core
// does not check this. The core is radix-4: each ButterflyComplexStage4 call
// advances two radix-2 stages at once, with at most one trailing
// ButterflyComplexStage when n is not a power of four, so the butterflies take
// the AVX+FMA / NEON vector paths where the span and block count justify them.
func (c *fftCore) transform(re, im []float64) {
	n := len(re)
	sh := c.logMax - bits.TrailingZeros(uint(n))
	// Bit-reversal reorder.
	for i, r := range c.bitrev[:n] {
		if j := int(r >> sh); j > i {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	// Butterfly stages via the radix-4 core: a radix-4 stage at span s advances the
	// transform two radix-2 stages at once (span s then span 2s), so it runs spans
	// 1, 4, 16, ... while a full radix-4 block fits (4*s <= n). tw1 = w^(2j) is the
	// span-s radix-2 table at stageTw[s-1 : 2s-1], tw2 = w^j is the first s entries of
	// the span-2s table at stageTw[2s-1 : 3s-1], and tw3 = w^(3j) is the dedicated
	// stage4Tw3 table at [(s-1)/3 : (s-1)/3 + s]. A single trailing radix-2 stage
	// finishes the transform when n is not a power of four (odd log2(n)).
	s := 1
	for butterflyStage4Radix*s <= n {
		o1 := s - 1                                // span-s radix-2 table
		o2 := butterflyStageRadix*s - 1            // span-2s radix-2 table, first s taken
		o3 := (s - 1) / (butterflyStage4Radix - 1) // dedicated w^(3j) table
		ButterflyComplexStage4(re, im, s,
			c.stageTwRe[o1:o1+s], c.stageTwIm[o1:o1+s],
			c.stageTwRe[o2:o2+s], c.stageTwIm[o2:o2+s],
			c.stage4Tw3Re[o3:o3+s], c.stage4Tw3Im[o3:o3+s])
		s *= butterflyStage4Radix
	}
	if s < n {
		off := s - 1
		ButterflyComplexStage(re, im, s, c.stageTwRe[off:off+s], c.stageTwIm[off:off+s])
	}
}
