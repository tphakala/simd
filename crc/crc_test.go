package crc

import (
	"math/rand"
	"strconv"
	"testing"
)

// refChecksum16 is an independent byte-at-a-time CRC-16 reference (FLAC poly
// 0x8005, init 0, MSB-first, no reflection) used as the oracle for parity. It
// is deliberately the simplest possible implementation.
func refChecksum16(p []byte) uint16 {
	const poly = 0x8005
	var c uint16
	for _, b := range p {
		c ^= uint16(b) << 8
		for range 8 {
			if c&0x8000 != 0 {
				c = (c << 1) ^ poly
			} else {
				c <<= 1
			}
		}
	}
	return c
}

// TestChecksum16CheckVector pins the standard CRC-16/UMTS check value.
func TestChecksum16CheckVector(t *testing.T) {
	if got := Checksum16([]byte("123456789")); got != 0xFEE8 {
		t.Fatalf("Checksum16(\"123456789\") = %#04x, want 0xFEE8", got)
	}
}

// TestChecksum16ParityScalarReference checks Checksum16 against the independent
// byte-at-a-time reference across lengths that straddle the 16-byte fold stride,
// the fold threshold, and odd tails.
func TestChecksum16ParityScalarReference(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for n := 0; n <= 600; n++ {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(r.Intn(256))
		}
		if got, want := Checksum16(buf), refChecksum16(buf); got != want {
			t.Fatalf("n=%d: Checksum16=%#04x want %#04x", n, got, want)
		}
	}
}

// TestChecksum16ParityLargeBuffers exercises the fold path on multi-KB buffers
// with assorted tail lengths.
func TestChecksum16ParityLargeBuffers(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for _, n := range []int{1024, 4096, 4097, 4111, 8192, 16384, 16385, 65535} {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(r.Intn(256))
		}
		if got, want := Checksum16(buf), refChecksum16(buf); got != want {
			t.Fatalf("n=%d: Checksum16=%#04x want %#04x", n, got, want)
		}
	}
}

// TestChecksum16AllByteValues guards against a folded byte being mistreated by
// a buffer that walks every byte value at the fold boundary.
func TestChecksum16AllByteValues(t *testing.T) {
	buf := make([]byte, 256)
	for i := range buf {
		buf[i] = byte(i)
	}
	for n := 0; n <= 256; n++ {
		if got, want := Checksum16(buf[:n]), refChecksum16(buf[:n]); got != want {
			t.Fatalf("n=%d: Checksum16=%#04x want %#04x", n, got, want)
		}
	}
}

// TestChecksum16ZeroAlloc asserts the hot path allocates nothing.
func TestChecksum16ZeroAlloc(t *testing.T) {
	buf := make([]byte, 16384)
	for i := range buf {
		buf[i] = byte(i)
	}
	if n := testing.AllocsPerRun(100, func() { _ = Checksum16(buf) }); n != 0 {
		t.Fatalf("Checksum16 allocated %v times per run, want 0", n)
	}
}

func BenchmarkChecksum16(b *testing.B) {
	buf := make([]byte, 16384)
	for i := range buf {
		buf[i] = byte(i)
	}
	b.SetBytes(int64(len(buf)))
	for b.Loop() {
		_ = Checksum16(buf)
	}
}

// BenchmarkChecksum16Scalar measures the slice-by-16 fallback directly, so the
// carry-less-multiply speedup can be read against the scalar baseline.
func BenchmarkChecksum16Scalar(b *testing.B) {
	buf := make([]byte, 16384)
	for i := range buf {
		buf[i] = byte(i)
	}
	b.SetBytes(int64(len(buf)))
	for b.Loop() {
		_ = checksum16Go(buf)
	}
}

func BenchmarkChecksum16Sizes(b *testing.B) {
	for _, n := range []int{32, 48, 64, 96, 128, 256, 1024, 4096, 8192} {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(i)
		}
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = Checksum16(buf)
			}
		})
	}
}
