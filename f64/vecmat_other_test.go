//go:build !amd64 && !arm64

package f64

// vecMatPaths lists the VecMat implementations that can be called directly on
// this host.
func vecMatPaths() []vecMatPath {
	return []vecMatPath{{"Go", vecMatGo, 1}}
}
