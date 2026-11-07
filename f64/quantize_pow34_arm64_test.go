//go:build arm64

package f64

// quantPow34Kernels lists the kernels the host can run, forced by calling each
// one directly (the public API only exercises the best tier).
func quantPow34Kernels() []quantPow34Kernel {
	ks := []quantPow34Kernel{{"go", quantizePow34BandsGo}}
	if hasNEON {
		ks = append(ks, quantPow34Kernel{"neon", quantizePow34BandsNEON})
	}
	return ks
}
