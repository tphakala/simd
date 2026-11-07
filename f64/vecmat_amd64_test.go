//go:build amd64

package f64

import "github.com/tphakala/simd/cpu"

// vecMatPaths lists the VecMat implementations that can be called directly on
// this host. VecMat dispatches without the function-pointer table, so the
// kernels are reached by name; the AVX-512, AVX+FMA and AVX-without-FMA tiers
// all run the same AVX kernels.
func vecMatPaths() []vecMatPath {
	paths := []vecMatPath{{"Go", vecMatGo, 1}}
	if cpu.X86.AVX {
		paths = append(paths, vecMatPath{"AVX", vecMatAVX, vecMatBlock4})
	}
	return paths
}
