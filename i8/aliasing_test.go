package i8

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the i8 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then compared under both the Go and SIMD kernels. The
// cross-type conversions (Quantize/Dequantize/Requantize/ToInt16/ToInt32) cannot
// alias in safe Go and are not swept here. It asserts nothing about the
// corruption pattern of a non-overlapping op, which is undefined.

func aliasEqI8(x, y int8) bool { return x == y }

// aliasGenI8 spreads values over the full int8 range, including -128 (the
// saturating edge for Abs/Neg).
func aliasGenI8(i int) int8 {
	u := uint32(i)*2654435761 + 1013904223
	return int8(u >> 24) //nolint:gosec // deliberate wrap to cover [-128,127]
}

// Scalar parameters for the scalar-taking ops. Their exact values do not affect
// the overlay property.
const (
	aliasClampLoI8 = int8(-100)
	aliasClampHiI8 = int8(100)
	aliasScalarI8  = int8(50)
)

func i8AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.UnaryCase("Abs", aliasEqI8, aliasGenI8, Abs),
		aliastest.UnaryCase("Neg", aliasEqI8, aliasGenI8, Neg),
		aliastest.BinaryCase("AddSaturate", aliasEqI8, aliasGenI8, AddSaturate),
		aliastest.BinaryCase("SubSaturate", aliasEqI8, aliasGenI8, SubSaturate),
		aliastest.BinaryCase("AbsDiff", aliasEqI8, aliasGenI8, AbsDiff),
		aliastest.BinaryCase("Max", aliasEqI8, aliasGenI8, Max),
		aliastest.BinaryCase("Min", aliasEqI8, aliasGenI8, Min),
		aliastest.UnaryCase("AddScalarSaturate", aliasEqI8, aliasGenI8, func(dst, a []int8) { AddScalarSaturate(dst, a, aliasScalarI8) }),
		aliastest.UnaryCase("SubScalarSaturate", aliasEqI8, aliasGenI8, func(dst, a []int8) { SubScalarSaturate(dst, a, aliasScalarI8) }),
		aliastest.UnaryCase("Clamp", aliasEqI8, aliasGenI8, func(dst, a []int8) { Clamp(dst, a, aliasClampLoI8, aliasClampHiI8) }),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, i8AliasCases())
	})
}
