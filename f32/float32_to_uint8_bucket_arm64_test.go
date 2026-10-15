//go:build arm64

package f32

import "testing"

// bucketNEONFunc adapts float32ToUint8BucketNEON to bucketFunc. The kernel needs
// len(dst) == len(src) >= 16, so shorter inputs go to the Go fallback, as the
// dispatcher does.
func bucketNEONFunc(dst []uint8, src []float32, scale, offset float32, buckets int) {
	n := min(len(dst), len(src))
	if n < bucketBlock {
		float32ToUint8BucketGo(dst, src, scale, offset, float32(buckets-1))
		return
	}
	float32ToUint8BucketNEON(dst[:n], src[:n], scale, offset, float32(buckets-1))
}

// TestFloat32ToUint8BucketNEONDirect runs the suite against the NEON kernel
// itself, so it stays covered when SIMD_DISABLE sends the public path to Go.
func TestFloat32ToUint8BucketNEONDirect(t *testing.T) {
	runBucketSuite(t, bucketNEONFunc)
}
