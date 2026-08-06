package f16

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the f16 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then compared bit-for-bit under both the Go and SIMD
// kernels. Float16 is a uint16 alias, so the comparison is exact on the raw bits
// (a deterministic NaN or Inf from Div/Sqrt/Reciprocal/Exp reproduces identically
// between the two runs). The reductions (Sum, Min, Max, Mean, DotProduct) write no
// output slice, and the cross-type conversions (ToFloat32Slice, FromFloat32Slice)
// cannot alias in safe Go, so neither is swept here. It asserts nothing about how a
// shifted overlay corrupts, which is undefined.

func aliasEqF16(x, y Float16) bool { return x == y }

// aliasGenF16 spreads finite half-precision values over roughly [-4, 4], including
// zero and both signs, so the arithmetic and the saturating/branching ops
// (ReLU, Clamp, Sqrt) all see representative inputs.
func aliasGenF16(i int) Float16 {
	u := uint32(i)*2654435761 + 1013904223
	v := float32(int32(u%2000)-1000) / 250.0 //nolint:gosec // deliberate wrap into [-4,4]
	return FromFloat32(v)
}

// Scalar parameters for the scalar-taking ops. Their exact values do not affect
// the overlay property.
var (
	aliasScaleF16   = FromFloat32(0.75)
	aliasAddScalF16 = FromFloat32(1.25)
	aliasClampLoF16 = FromFloat32(-1.5)
	aliasClampHiF16 = FromFloat32(1.5)
)

func f16AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.BinaryCase("Add", aliasEqF16, aliasGenF16, Add),
		aliastest.BinaryCase("Sub", aliasEqF16, aliasGenF16, Sub),
		aliastest.BinaryCase("Mul", aliasEqF16, aliasGenF16, Mul),
		aliastest.BinaryCase("Div", aliasEqF16, aliasGenF16, Div),
		aliastest.TernaryCase("FMA", aliasEqF16, aliasGenF16, FMA),
		aliastest.UnaryCase("Abs", aliasEqF16, aliasGenF16, Abs),
		aliastest.UnaryCase("Neg", aliasEqF16, aliasGenF16, Neg),
		aliastest.UnaryCase("ReLU", aliasEqF16, aliasGenF16, ReLU),
		aliastest.UnaryCase("Sigmoid", aliasEqF16, aliasGenF16, Sigmoid),
		aliastest.UnaryCase("Sqrt", aliasEqF16, aliasGenF16, Sqrt),
		aliastest.UnaryCase("Reciprocal", aliasEqF16, aliasGenF16, Reciprocal),
		aliastest.UnaryCase("Exp", aliasEqF16, aliasGenF16, Exp),
		aliastest.UnaryCase("Scale", aliasEqF16, aliasGenF16, func(dst, a []Float16) { Scale(dst, a, aliasScaleF16) }),
		aliastest.UnaryCase("AddScalar", aliasEqF16, aliasGenF16, func(dst, a []Float16) { AddScalar(dst, a, aliasAddScalF16) }),
		aliastest.UnaryCase("Clamp", aliasEqF16, aliasGenF16, func(dst, a []Float16) { Clamp(dst, a, aliasClampLoF16, aliasClampHiF16) }),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, f16AliasCases())
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under both the Go and SIMD kernels.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, f16AliasCases())
	})
}
