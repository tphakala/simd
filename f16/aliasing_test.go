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

// aliasBuildF16 fills a length-n slice from the shared generator.
func aliasBuildF16(n int) []Float16 {
	s := make([]Float16, n)
	for i := range s {
		s[i] = aliasGenF16(i)
	}
	return s
}

// addScaledAliasCase covers AddScaled's dst==s overlay. AddScaled is a
// read-modify-write accumulator (dst += alpha*s), which the write-only Unary
// helper does not model, so it uses the exported aliastest.Report: the reference
// run accumulates into a copy of s from a distinct s, the overlay run accumulates
// into s aliased as both dst and source.
func addScaledAliasCase() aliastest.Case {
	alpha := FromFloat32(0.5)
	return aliastest.Case{
		Name: "AddScaled",
		Check: func(t *testing.T, n int) {
			t.Helper()
			s := aliasBuildF16(n)
			want := aliasBuildF16(n)
			AddScaled(want, alpha, s)
			got := aliasBuildF16(n)
			AddScaled(got, alpha, got)
			aliastest.Report(t, n, "dst=s", aliasEqF16, want, got)
		},
		Alloc: func(t *testing.T) {
			t.Helper()
			s := aliasBuildF16(64)
			aliastest.ZeroAlloc(t, "AddScaled dst=s", func() { AddScaled(s, alpha, s) })
		},
	}
}

// accumulateAddAliasCase covers AccumulateAdd's dst==src overlay at offset 0.
// AccumulateAdd is also an accumulator (dst[offset:] += src), so it uses Report
// the same way, pinning the exact dst==src overlay.
func accumulateAddAliasCase() aliastest.Case {
	return aliastest.Case{
		Name: "AccumulateAdd",
		Check: func(t *testing.T, n int) {
			t.Helper()
			src := aliasBuildF16(n)
			want := aliasBuildF16(n)
			AccumulateAdd(want, src, 0)
			got := aliasBuildF16(n)
			AccumulateAdd(got, got, 0)
			aliastest.Report(t, n, "dst=src", aliasEqF16, want, got)
		},
		Alloc: func(t *testing.T) {
			t.Helper()
			s := aliasBuildF16(64)
			aliastest.ZeroAlloc(t, "AccumulateAdd dst=src", func() { AccumulateAdd(s, s, 0) })
		},
	}
}

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
		aliastest.UnaryCase("Tanh", aliasEqF16, aliasGenF16, Tanh),
		aliastest.UnaryCase("Normalize", aliasEqF16, aliasGenF16, Normalize),
		aliastest.UnaryCase("CumulativeSum", aliasEqF16, aliasGenF16, CumulativeSum),
		aliastest.UnaryCase("Scale", aliasEqF16, aliasGenF16, func(dst, a []Float16) { Scale(dst, a, aliasScaleF16) }),
		aliastest.UnaryCase("AddScalar", aliasEqF16, aliasGenF16, func(dst, a []Float16) { AddScalar(dst, a, aliasAddScalF16) }),
		aliastest.UnaryCase("Clamp", aliasEqF16, aliasGenF16, func(dst, a []Float16) { Clamp(dst, a, aliasClampLoF16, aliasClampHiF16) }),
		aliastest.UnaryCase("ClampScale", aliasEqF16, aliasGenF16, func(dst, a []Float16) { ClampScale(dst, a, aliasClampLoF16, aliasClampHiF16, aliasScaleF16) }),
		addScaledAliasCase(),
		accumulateAddAliasCase(),
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
