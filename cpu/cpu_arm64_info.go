//go:build arm64

package cpu

// cpuInfo is shared by the Linux and darwin arm64 builds: the two files differ
// only in their init() feature detection, but report the tier identically.
func cpuInfo() string {
	// Report the tier the library actually runs (NEON), not SVE: there are no SVE
	// kernels yet, so an SVE-capable host still executes the NEON path. SVE/SVE2 is
	// annotated as detected-but-unused so the capability is still visible.
	var base string
	switch {
	case ARM64.NEON && ARM64.FP16:
		base = "ARM64 NEON+FP16"
	case ARM64.NEON:
		base = "ARM64 NEON"
	default:
		return "ARM64 (no SIMD)"
	}
	switch {
	case ARM64.SVE2:
		return base + " (SVE2 detected, unused)"
	case ARM64.SVE:
		return base + " (SVE detected, unused)"
	default:
		return base
	}
}
