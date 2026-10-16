//go:build amd64 || arm64

package f32

// biquadMaxLanes is the lane count of the widest BiquadCascade kernel: one
// section per lane, in rows of eight float32 (a YMM register, or two Q
// registers).
const biquadMaxLanes = 8

// biquadNarrowLanes is the lane count of the 4-section kernels (one XMM or Q
// register per row). A block of up to this many sections uses them, and an
// AVX-only host runs every cascade in passes of this size.
const biquadNarrowLanes = 4

// biquadSkew is the lane pipeline's skew in samples: lane k runs sample t-2k at
// step t, so a kernel needs at least biquadSkew*(sections-1) samples to fill the
// pipeline.
const biquadSkew = 2

// biquadLanes is the transposed coefficient and state image a kernel works on.
// Row r holds one value per section (lane = section; the pair kernels use lane =
// 2*section + channel), rows are biquadMaxLanes float32 apart for every kernel
// width. A zero value leaves the unused lanes at zero coefficients and state,
// which the kernels never unpack.
type biquadLanes struct {
	co  [biquadCoeffs * biquadMaxLanes]float32
	st  [biquadStateLen * biquadMaxLanes]float32
	sel [biquadMaxLanes]uint32 // output-lane selector: pack sets every word alike, packPair alternates two values
}

// pack transposes the per-section coeffs and state of one block into lanes and
// sets every word of the output-lane selector to sel.
func (l *biquadLanes) pack(coeffs, state []float32, sel uint32) {
	for i := range l.sel {
		l.sel[i] = sel
	}
	for k := range len(coeffs) / biquadCoeffs {
		for r := range biquadCoeffs {
			l.co[r*biquadMaxLanes+k] = coeffs[biquadCoeffs*k+r]
		}
		for r := range biquadStateLen {
			l.st[r*biquadMaxLanes+k] = state[biquadStateLen*k+r]
		}
	}
}

// unpack writes the state image back into the per-section state of one block.
func (l *biquadLanes) unpack(state []float32) {
	for k := range len(state) / biquadStateLen {
		for r := range biquadStateLen {
			state[biquadStateLen*k+r] = l.st[r*biquadMaxLanes+k]
		}
	}
}

// biquadMinWork is the smallest n*sections for which a block runs on a kernel.
// Below it the per-call cost of packing the lanes and of filling and draining
// the pipeline outweighs the kernel's gain over the Go path. The value comes
// from BenchmarkBiquadCascadeShort on a Kaby Lake core and on a Cortex-A76: on
// both, at n*sections = 64 the kernel beat Go for 2 and 4 sections; at 32 it
// lost for 1, 2 and 4.
const biquadMinWork = 64

// biquadKernelOK reports whether a block of the given section count runs on a
// kernel for n samples. The kernel precondition is n >= 1 and n >= biquadSkew *
// (sections-1), so that fill and drain never overlap; minWork is the performance
// floor on n*sections, below which the Go path is faster. Tests pass minWork = 0
// to reach the kernels with the shortest legal inputs.
func biquadKernelOK(n, sections, minWork int) bool {
	return n >= 1 && n >= biquadSkew*(sections-1) && n*sections >= minWork
}
