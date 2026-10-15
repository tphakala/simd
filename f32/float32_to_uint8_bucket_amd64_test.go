//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// bucketAVX2Func adapts float32ToUint8BucketAVX2 to bucketFunc. The kernel needs
// len(dst) == len(src) >= 16, so shorter inputs go to the Go fallback, as the
// dispatcher does.
func bucketAVX2Func(dst []uint8, src []float32, scale, offset float32, buckets int) {
	n := min(len(dst), len(src))
	if n < bucketBlock {
		float32ToUint8BucketGo(dst, src, scale, offset, float32(buckets-1))
		return
	}
	float32ToUint8BucketAVX2(dst[:n], src[:n], scale, offset, float32(buckets-1))
}

// TestFloat32ToUint8BucketAVX2Direct runs the suite against the AVX2 kernel
// itself. The public path picks its kernel from cpu.X86.AVX2 directly, so the
// tier-forcing helper does not steer it; this test is what keeps the kernel
// covered when SIMD_DISABLE sends the public path to Go. It skips without AVX2
// so the sub-AVX2 legs never execute a VEX.256 integer op.
func TestFloat32ToUint8BucketAVX2Direct(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	runBucketSuite(t, bucketAVX2Func)
}
