//go:build !amd64 && !arm64

package crc

// checksum16 uses the scalar slice-by-16 loop on architectures without a
// carry-less-multiply instruction.
func checksum16(p []byte) uint16 { return checksum16Go(p) }
