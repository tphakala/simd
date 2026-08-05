//go:build arm64

package c64

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the Go and NEON arm64 paths by
// flipping the package hasNEON gate.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasNEON, "NEON", run)
}
