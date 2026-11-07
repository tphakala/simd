//go:build arm64

package f64

// vecMatPaths lists the VecMat implementations that can be called directly on
// this host.
func vecMatPaths() []vecMatPath {
	paths := []vecMatPath{{"Go", vecMatGo, 1}}
	if hasNEON {
		paths = append(paths, vecMatPath{"NEON", vecMatNEON, vecMatBlock2})
	}
	return paths
}
