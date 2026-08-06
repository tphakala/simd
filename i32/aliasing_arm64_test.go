//go:build arm64

package i32

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the pure-Go reference and the NEON
// kernels by flipping the package hasNEON gate. Forcing it off runs the Go path
// at every length, not just the sub-block tail.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasNEON, "NEON", run)
}
