//go:build !amd64 && !arm64

package f64

// quantPow34Kernels lists the kernels available on this architecture.
func quantPow34Kernels() []quantPow34Kernel {
	return []quantPow34Kernel{{"go", quantizePow34BandsGo}}
}
