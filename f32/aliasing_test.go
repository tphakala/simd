package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the f32 exact-overlay contract (issue #221).
//
// For every operation documented as supporting an exact in-place overlay, this
// runs the operation twice at each length in aliastest.Sizes: once into a
// separate destination, once with the destination physically overlaid on an
// input, and asserts the two results are bit-identical. forTiers forces each
// amd64 function-pointer dispatch tier and both the Go and NEON arm64 paths. A
// few f32 ops (CopySign, AbsPow34, and the transcendentals) dispatch through an
// inline CPU-feature branch the harness cannot rebind; the size sweep still runs
// their Go and native-SIMD kernels, and TestAliasingDirectKernels (amd64)
// exercises the CopySign and AbsPow34 SSE and Go kernels that an AVX host would
// otherwise never reach.
//
// It asserts nothing about how a non-overlapping op corrupts a shifted overlay:
// that pattern is undefined and varies with kernel width and length.

func aliasEqF32(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }

// hashF32 is a deterministic pseudo-random value in [0,1) from an index.
func hashF32(i int) float32 {
	u := uint32(i)*2654435761 + 1013904223
	return float32(u) / float32(1<<32)
}

// genF32 spreads values over [-4,4), including negatives and near-zero.
func genF32(i int) float32 { return hashF32(i)*8 - 4 }

// genF32Pos spreads values over (0,8] for domain-restricted ops (log, sqrt, pow).
func genF32Pos(i int) float32 { return hashF32(i)*8 + 0.03125 }

// Scalar parameters for the scalar-taking element-wise ops. The exact values are
// irrelevant to the overlay property; they only need to exercise the kernel.
const (
	aliasScaleK  = 1.5
	aliasAddK    = 0.75
	aliasClampLo = -1.25
	aliasClampHi = 2.5
	aliasScaleC  = 0.5
	aliasPowExp  = 0.75
)

func f32AliasCases() []aliastest.Case {
	return []aliastest.Case{
		// Element-wise unary maps (dst may overlay a exactly).
		aliastest.UnaryCase("Abs", aliasEqF32, genF32, Abs),
		aliastest.UnaryCase("Neg", aliasEqF32, genF32, Neg),
		aliastest.UnaryCase("Round", aliasEqF32, genF32, Round),
		aliastest.UnaryCase("Reciprocal", aliasEqF32, genF32, Reciprocal),
		aliastest.UnaryCase("ReLU", aliasEqF32, genF32, ReLU),
		aliastest.UnaryCase("Sigmoid", aliasEqF32, genF32, Sigmoid),
		aliastest.UnaryCase("Tanh", aliasEqF32, genF32, Tanh),
		aliastest.UnaryCase("Exp", aliasEqF32, genF32, Exp),
		aliastest.UnaryCase("Sqrt", aliasEqF32, genF32Pos, Sqrt),
		aliastest.UnaryCase("Log", aliasEqF32, genF32Pos, Log),
		aliastest.UnaryCase("Log2", aliasEqF32, genF32Pos, Log2),
		aliastest.UnaryCase("Log10", aliasEqF32, genF32Pos, Log10),
		aliastest.UnaryCase("AbsPow34", aliasEqF32, genF32, AbsPow34),
		aliastest.UnaryCase("CumulativeSum", aliasEqF32, genF32, CumulativeSum),
		aliastest.UnaryCase("Normalize", aliasEqF32, genF32, Normalize),
		aliastest.UnaryCase("Reverse", aliasEqF32, genF32, Reverse),

		// Element-wise unary maps with a scalar parameter.
		aliastest.UnaryCase("Scale", aliasEqF32, genF32, func(dst, a []float32) { Scale(dst, a, aliasScaleK) }),
		aliastest.UnaryCase("AddScalar", aliasEqF32, genF32, func(dst, a []float32) { AddScalar(dst, a, aliasAddK) }),
		aliastest.UnaryCase("SubFromScalar", aliasEqF32, genF32, func(dst, a []float32) { SubFromScalar(dst, a, aliasAddK) }),
		aliastest.UnaryCase("Clamp", aliasEqF32, genF32, func(dst, a []float32) { Clamp(dst, a, aliasClampLo, aliasClampHi) }),
		aliastest.UnaryCase("ClampScale", aliasEqF32, genF32, func(dst, a []float32) { ClampScale(dst, a, aliasClampLo, aliasClampHi, aliasScaleC) }),
		aliastest.UnaryCase("Pow", aliasEqF32, genF32Pos, func(dst, a []float32) { Pow(dst, a, aliasPowExp) }),

		// Element-wise binary maps (dst may overlay a, b, or both exactly).
		aliastest.BinaryCase("Add", aliasEqF32, genF32, Add),
		aliastest.BinaryCase("Sub", aliasEqF32, genF32, Sub),
		aliastest.BinaryCase("Mul", aliasEqF32, genF32, Mul),
		aliastest.BinaryCase("Div", aliasEqF32, genF32, Div),
		aliastest.BinaryCase("PowElem", aliasEqF32, genF32Pos, PowElem),
		aliastest.BinaryCase("CopySign", aliasEqF32, genF32, CopySign),
		aliastest.BinaryCase("AbsSqComplex", aliasEqF32, genF32, AbsSqComplex),

		// Fused multiply-add (dst may overlay a, b, or c exactly).
		aliastest.TernaryCase("FMA", aliasEqF32, genF32, FMA),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, f32AliasCases())
		t.Run("MulComplex", func(t *testing.T) { sweepSplitComplex(t, MulComplex) })
		t.Run("MulConjComplex", func(t *testing.T) { sweepSplitComplex(t, MulConjComplex) })
		t.Run("AddScaled", sweepAddScaled)
	})
}

// sweepAddScaled checks AddScaled's documented in-place overlay: s may equal dst
// exactly (the self-scaling dst[i] += alpha*dst[i]). AddScaled does not fit the
// generic Unary/Binary shapes because dst is a read-modify-write accumulator, so
// it gets a bespoke check. AddScaled is function-pointer dispatched, so forTiers
// forces its tiers.
func sweepAddScaled(t *testing.T) {
	t.Helper()
	const alpha = 1.5
	for _, n := range aliastest.Sizes {
		data := make([]float32, n)
		for i := range data {
			data[i] = genF32(i)
		}
		want := append([]float32(nil), data...)
		AddScaled(want, alpha, append([]float32(nil), data...))
		got := append([]float32(nil), data...)
		AddScaled(got, alpha, got)
		aliastest.Report(t, n, "AddScaled s==dst", aliasEqF32, want, got)
	}
}

func splitInputs(n int) (aRe, aIm, bRe, bIm []float32) {
	aRe = make([]float32, n)
	aIm = make([]float32, n)
	bRe = make([]float32, n)
	bIm = make([]float32, n)
	for i := range aRe {
		aRe[i] = genF32(i)
		aIm[i] = genF32(i + 11)
		bRe[i] = genF32(i + 101)
		bIm[i] = genF32(i + 211)
	}
	return
}

// sweepSplitComplex checks the two per-operand in-place overlays claimed for the
// split-format complex products: the result may overwrite a (dstRe==aRe,
// dstIm==aIm) or b (dstRe==bRe, dstIm==bIm). The two output components must stay
// distinct, so that overlay is never claimed and never tested.
func sweepSplitComplex(t *testing.T, op func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)) {
	t.Helper()
	for _, n := range aliastest.Sizes {
		aRe, aIm, bRe, bIm := splitInputs(n)
		wantRe := make([]float32, n)
		wantIm := make([]float32, n)
		op(wantRe, wantIm, aRe, aIm, bRe, bIm)

		// Overwrite a in place.
		gRe := append([]float32(nil), aRe...)
		gIm := append([]float32(nil), aIm...)
		op(gRe, gIm, gRe, gIm, bRe, bIm)
		aliastest.Report(t, n, "dstRe=aRe,dstIm=aIm", aliasEqF32, wantRe, gRe)
		aliastest.Report(t, n, "dstRe=aRe,dstIm=aIm", aliasEqF32, wantIm, gIm)

		// Overwrite b in place.
		gRe = append([]float32(nil), bRe...)
		gIm = append([]float32(nil), bIm...)
		op(gRe, gIm, aRe, aIm, gRe, gIm)
		aliastest.Report(t, n, "dstRe=bRe,dstIm=bIm", aliasEqF32, wantRe, gRe)
		aliastest.Report(t, n, "dstRe=bRe,dstIm=bIm", aliasEqF32, wantIm, gIm)
	}
}
