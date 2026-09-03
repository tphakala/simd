package i32

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the i32 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then compared under both the Go and SIMD kernels. The
// length-changing shuffles (Interleave2, Deinterleave2), the valid convolutions
// (FIRValidQ15 and FIRSymValidQ15, whose dst is shorter than x), the in-place pair transform
// (Butterfly) and the reductions (Sum, MaxAbs, MinMax, SumSqShiftedQ31) do not take an
// element-for-element dst overlay and are not swept here; see the package doc. It
// asserts nothing about how a shifted overlay (dst offset from an input)
// corrupts, which is undefined.

func aliasEqI32(x, y int32) bool { return x == y }

// aliasGenI32 spreads values over the full int32 range, including MinInt32 (the
// wrapping edge for Abs and the Q31 saturating cases).
func aliasGenI32(i int) int32 {
	u := uint32(i)*2654435761 + 1013904223
	return int32(u) //nolint:gosec // deliberate wrap to cover the full int32 range
}

// Scalar parameters for the scalar-taking ops. Their exact values do not affect
// the overlay property.
const (
	aliasScaleKI32              = int32(0x2000_0000) // 0.25 in Q31
	aliasScaleKI16              = int16(0x4000)      // 0.5 in Q15
	aliasGainG                  = int32(0x4000_0000) // 0.5 in Q31
	aliasGainPre, aliasGainPost = 9, 12
)

// aliasSignAt is a deterministic sign pattern for NegWhereNeg, alternating so both
// the negated and pass-through branches are exercised at every length.
func aliasSignAt(i int) float32 {
	if uint32(i)*2654435761>>31 == 1 {
		return -1
	}
	return 1
}

// negWhereNegAliasCase covers NegWhereNeg's dst==mag overlay. The generic helpers
// model a single element type, but NegWhereNeg mixes int32 (dst, mag) with float32
// (sign), so it builds the check with the exported aliastest.Report. sign is a
// distinct float32 slice that cannot alias the int32 dst.
func negWhereNegAliasCase() aliastest.Case {
	return aliastest.Case{
		Name: "NegWhereNeg",
		Check: func(t *testing.T, n int) {
			t.Helper()
			mag := make([]int32, n)
			sign := make([]float32, n)
			for i := range mag {
				mag[i] = aliasGenI32(i)
				sign[i] = aliasSignAt(i)
			}
			want := make([]int32, n)
			NegWhereNeg(want, mag, sign)
			got := make([]int32, n)
			copy(got, mag)
			NegWhereNeg(got, got, sign)
			aliastest.Report(t, n, "dst=mag", aliasEqI32, want, got)
		},
		Alloc: func(t *testing.T) {
			t.Helper()
			const n = 64
			mag := make([]int32, n)
			sign := make([]float32, n)
			for i := range mag {
				mag[i] = aliasGenI32(i)
				sign[i] = aliasSignAt(i)
			}
			aliastest.ZeroAlloc(t, "NegWhereNeg dst=mag", func() { NegWhereNeg(mag, mag, sign) })
		},
	}
}

func i32AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.BinaryCase("Add", aliasEqI32, aliasGenI32, Add),
		aliastest.BinaryCase("Sub", aliasEqI32, aliasGenI32, Sub),
		aliastest.UnaryCase("Abs", aliasEqI32, aliasGenI32, Abs),
		aliastest.UnaryCase("ScaleQ31", aliasEqI32, aliasGenI32, func(dst, a []int32) { ScaleQ31(dst, a, aliasScaleKI32) }),
		aliastest.UnaryCase("ScaleQ15", aliasEqI32, aliasGenI32, func(dst, a []int32) { ScaleQ15(dst, a, aliasScaleKI16) }),
		aliastest.UnaryCase("GainQ31", aliasEqI32, aliasGenI32, func(dst, a []int32) { GainQ31(dst, a, aliasGainG, aliasGainPre, aliasGainPost) }),
		negWhereNegAliasCase(),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, i32AliasCases())
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under both the Go and SIMD kernels.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, i32AliasCases())
	})
}
