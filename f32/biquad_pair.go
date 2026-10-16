//go:build amd64 || arm64

package f32

const (
	// biquadPairSections is the most sections a pair kernel runs: two channels
	// times two sections fill the four lanes of one register row.
	biquadPairSections = 2
	// biquadPairChannels is the number of channels one pair kernel holds.
	biquadPairChannels = 2
)

// biquadPairMinWork is the smallest m*sections, m being the common length of the
// two channels of a pair, for which the pair kernel runs; below it the pair goes
// through biquadCascadeChannel per channel. The value comes from
// BenchmarkBiquadCascadePairShort, which compares the pair kernel forced on with
// two production BiquadCascade calls (not with Go), on a Kaby Lake core and on a
// Cortex-A76 core. At m*sections = 8 the two calls were faster on both hosts, by
// 1% to 24%. At 16 and at every larger point of the sweep (up to m = 256) the
// pair kernel won on both: the two calls took 0.9% to 22.5% longer at 16, and 1.7x
// to 1.9x longer at m = 256. The crossover sits between the same two sweep points
// on both hosts, so one constant serves both architectures.
const biquadPairMinWork = 16

// packPair transposes the coefficients of a 1- or 2-section cascade and the
// states of two channels into the pair layout: lane 2k+ch of row r holds section
// k of channel ch. Coefficients are shared, so both channel lanes of a section
// get the same value. The output-lane selector alternates sel0 and sel1, so the
// output select moves the last section of channel 0 to lane 0 and of channel 1
// to lane 1. Lanes at or above 2*sections stay zero.
func (l *biquadLanes) packPair(coeffs, st0, st1 []float32, sel0, sel1 uint32) {
	for i := 0; i < len(l.sel); i += biquadPairChannels {
		l.sel[i] = sel0
		l.sel[i+1] = sel1
	}
	for k := range len(coeffs) / biquadCoeffs {
		lane := biquadPairChannels * k
		for r := range biquadCoeffs {
			l.co[r*biquadMaxLanes+lane] = coeffs[biquadCoeffs*k+r]
			l.co[r*biquadMaxLanes+lane+1] = coeffs[biquadCoeffs*k+r]
		}
		for r := range biquadStateLen {
			l.st[r*biquadMaxLanes+lane] = st0[biquadStateLen*k+r]
			l.st[r*biquadMaxLanes+lane+1] = st1[biquadStateLen*k+r]
		}
	}
}

// unpackPair writes the state image back into the per-section states of the two
// channels of a pair.
func (l *biquadLanes) unpackPair(st0, st1 []float32) {
	for k := range len(st0) / biquadStateLen {
		lane := biquadPairChannels * k
		for r := range biquadStateLen {
			st0[biquadStateLen*k+r] = l.st[r*biquadMaxLanes+lane]
			st1[biquadStateLen*k+r] = l.st[r*biquadMaxLanes+lane+1]
		}
	}
}

func biquadCascadeN32(dst, src [][]float32, coeffs []float32, state [][]float32) {
	biquadCascadeNBlocks(dst, src, coeffs, state, biquadKernelLanes(), biquadMinWork, biquadPairMinWork)
}

// biquadCascadeNBlocks runs the channels of a multichannel cascade. For a
// cascade of at most biquadPairSections sections on a host with a kernel (lanes
// > 0), channels (0,1), (2,3), ... run in pairs through biquadCascadePair; an odd
// last channel, and every channel of a longer cascade or of a host without a
// kernel, runs through biquadCascadeBlocks as BiquadCascade would. lanes,
// minWork and pairMinWork are as in biquadCascadeBlocks and biquadCascadePair.
func biquadCascadeNBlocks(dst, src [][]float32, coeffs []float32, state [][]float32, lanes, minWork, pairMinWork int) {
	c := 0
	if lanes > 0 && len(coeffs)/biquadCoeffs <= biquadPairSections {
		for ; c+1 < len(dst); c += biquadPairChannels {
			biquadCascadePair(dst, src, coeffs, state, c, lanes, minWork, pairMinWork)
		}
	}
	for ; c < len(dst); c++ {
		biquadCascadeChannel(dst[c], src[c], coeffs, state[c], lanes, minWork)
	}
}

// biquadCascadeChannel runs one channel through biquadCascadeBlocks over the
// common prefix of dst and src, skipping an empty one: biquadCascadeBlocks needs
// len(dst) == len(src) >= 1.
func biquadCascadeChannel(dst, src, coeffs, state []float32, lanes, minWork int) {
	if n := min(len(dst), len(src)); n > 0 {
		biquadCascadeBlocks(dst[:n], src[:n], coeffs, state, lanes, minWork)
	}
}
