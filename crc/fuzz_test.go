package crc

import "testing"

// FuzzChecksum16 is a differential fuzz target: the dispatched Checksum16 (which
// folds with PCLMULQDQ/PMULL when available) must agree bit-for-bit with the
// pure-Go scalar reference for every input length. The classic bug class here is
// the tail/remainder handling around the minFoldBytes = 64 folding threshold, so
// the seeds bracket 63/64/65 and 127/128 in addition to the empty and tiny
// inputs. Seeds run under plain `go test`; `go test -fuzz=FuzzChecksum16` widens
// the length and byte-pattern space.
func FuzzChecksum16(f *testing.F) {
	// Lengths bracketing the fold threshold and the 16-byte fold stride.
	for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 79, 80, 127, 128, 129, 256} {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(i*31 + 7)
		}
		f.Add(b)
	}
	f.Add([]byte("The quick brown fox jumps over the lazy dog"))

	f.Fuzz(func(t *testing.T, p []byte) {
		got := Checksum16(p)
		want := checksum16Go(p)
		if got != want {
			t.Fatalf("Checksum16(len=%d) = 0x%04x, reference = 0x%04x", len(p), got, want)
		}
	})
}
