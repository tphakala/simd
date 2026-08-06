//go:build amd64

package f16

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the pure-Go reference and the F16C
// kernels by flipping the package hasF16C gate. Most f16 arithmetic is pure Go on
// amd64 (the F16C tier accelerates the float16<->float32 conversions the ops build
// on), so for those ops both settings run the same kernel; the sweep still covers
// every op's overlay on the path it actually takes.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasF16C, "F16C", run)
}
