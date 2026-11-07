package f64

import "math/bits"

// VecMat multiplies the row vector x by a strided row-major matrix m:
//
//	dst[c] = sum over r in [0, len(x)) of x[r] * m[r*stride+c], for c in [0, len(dst))
//
// x has R = len(x) entries, dst has C = len(dst) entries, and row r of the
// matrix starts at m[r*stride]. The matrix is stored so that each output is a
// column: the elements that feed one output are stride apart and the elements
// that feed neighbouring outputs are adjacent, which is what the SIMD kernels
// vectorize over. stride may exceed C, so an output window narrower than the
// stored rows (the first n of a table of fixed width) is a valid call.
//
// Each column is accumulated from +0.0 in ascending r, and every product rounds
// to float64 before it is added. The multiply and the add are never fused, and
// the order is never changed, so every non-NaN output is bit-identical to the
// scalar loop
//
//	sum := 0.0
//	for r := range x { sum += float64(x[r] * m[r*stride+c]) }
//
// on every path (amd64 AVX, arm64 NEON, and the pure-Go fallback). A NaN output
// appears exactly where the scalar sum produces NaN; its bit pattern is
// unspecified and can differ between architectures. Because every column starts
// at +0.0, all -0.0 products sum to +0.0, and len(x) == 0 gives +0.0 in every
// column. [DotProduct] and [MulAdd] fuse on some
// tiers and are not bit-identical to this.
//
// dst is fully overwritten. VecMat panics before writing anything if
// stride < len(dst), or if m is shorter than (len(x)-1)*stride+len(dst). With
// len(dst) == 0 it returns immediately and examines neither stride nor m.
//
// dst must not overlap x or m. x and m are only read and may overlap each
// other. VecMat does not allocate.
func VecMat(dst, x, m []float64, stride int) {
	c := len(dst)
	if c == 0 {
		return // nothing to write; stride and m are not examined
	}
	if stride < c {
		panic("simd: VecMat stride is less than len(dst)")
	}
	r := len(x)
	if r == 0 {
		clear(dst) // the empty sum, +0.0; m may be nil
		return
	}
	hi, lo := bits.Mul(uint(r-1), uint(stride))
	if len(m) < c || hi != 0 || lo > uint(len(m)-c) {
		panic("simd: VecMat matrix is shorter than (len(x)-1)*stride+len(dst)")
	}
	vecMat(dst, x, m, stride)
}
