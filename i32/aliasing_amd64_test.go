//go:build amd64

package i32

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the pure-Go reference and the AVX2
// kernels by flipping the package hasAVX2 gate (every swept op dispatches on that
// var). Forcing it off runs the Go path at every length, not just the sub-block
// tail.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasAVX2, "AVX2", run)
}
