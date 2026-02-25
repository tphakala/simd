//go:build !darwin || !arm64 || !cgo

package accelerate

func Enabled() bool {
	return false
}

func Sin64(_, _ []float64) bool {
	return false
}

func Cos64(_, _ []float64) bool {
	return false
}

func SinCos64(_, _, _ []float64) bool {
	return false
}
