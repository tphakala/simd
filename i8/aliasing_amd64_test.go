//go:build amd64

package i8

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the pure-Go reference and the AVX2
// kernels by flipping the package hasAVX2 gate (the i8 kernels dispatch on that
// var, not a function-pointer table). Forcing it off runs the Go path at every
// length, not just the sub-block tail.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasAVX2, "AVX2", run)
}
