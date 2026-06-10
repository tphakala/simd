//go:build arm64

package f32

import "testing"

// buildStreams returns nc planar streams of length n, each value unique to its
// (channel, frame) via chanVal so a misplaced lane is always caught.
func buildStreams(nc, n int) [][]float32 {
	srcs := make([][]float32, nc)
	for c := range srcs {
		srcs[c] = make([]float32, n)
		for i := range srcs[c] {
			srcs[c][i] = chanVal(c, i)
		}
	}
	return srcs
}

// TestInterleave68NEONKernels calls the N=6 and N=8 NEON kernels directly
// (bypassing the dispatch threshold) and checks interleave against the scalar
// oracle and the deinterleave round-trip, across the 4-frame main loop and every
// frame remainder (n mod 4 = 0..3).
func TestInterleave68NEONKernels(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	lengths := []int{4, 5, 6, 7, 8, 9, 11, 16, 17, 19, 64, 100, 255}
	for _, n := range lengths {
		// N=6 interleave vs oracle.
		s6 := buildStreams(6, n)
		got6 := make([]float32, n*6)
		interleave6NEON(got6, s6[0], s6[1], s6[2], s6[3], s6[4], s6[5], n)
		want6 := make([]float32, n*6)
		interleaveNRef(want6, s6, n)
		for i := range want6 {
			if got6[i] != want6[i] {
				t.Fatalf("interleave6NEON n=%d i=%d got %v want %v", n, i, got6[i], want6[i])
			}
		}
		// N=6 deinterleave round-trip.
		back6 := make([][]float32, 6)
		for c := range back6 {
			back6[c] = make([]float32, n)
		}
		deinterleave6NEON(back6[0], back6[1], back6[2], back6[3], back6[4], back6[5], got6, n)
		for c := range back6 {
			for i := range n {
				if back6[c][i] != s6[c][i] {
					t.Fatalf("deinterleave6NEON n=%d c=%d i=%d got %v want %v", n, c, i, back6[c][i], s6[c][i])
				}
			}
		}

		// N=8 interleave vs oracle.
		s8 := buildStreams(8, n)
		got8 := make([]float32, n*8)
		interleave8NEON(got8, s8[0], s8[1], s8[2], s8[3], s8[4], s8[5], s8[6], s8[7], n)
		want8 := make([]float32, n*8)
		interleaveNRef(want8, s8, n)
		for i := range want8 {
			if got8[i] != want8[i] {
				t.Fatalf("interleave8NEON n=%d i=%d got %v want %v", n, i, got8[i], want8[i])
			}
		}
		// N=8 deinterleave round-trip.
		back8 := make([][]float32, 8)
		for c := range back8 {
			back8[c] = make([]float32, n)
		}
		deinterleave8NEON(back8[0], back8[1], back8[2], back8[3], back8[4], back8[5], back8[6], back8[7], got8, n)
		for c := range back8 {
			for i := range n {
				if back8[c][i] != s8[c][i] {
					t.Fatalf("deinterleave8NEON n=%d c=%d i=%d got %v want %v", n, c, i, back8[c][i], s8[c][i])
				}
			}
		}
	}
}

// TestInterleave68AllocFreeARM64 confirms the N=6/N=8 public dispatch stays
// allocation-free through the NEON kernels.
func TestInterleave68AllocFreeARM64(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	const n = 256
	for _, nc := range []int{6, 8} {
		srcs := buildStreams(nc, n)
		dst := make([]float32, n*nc)
		if a := testing.AllocsPerRun(50, func() { InterleaveN(dst, srcs) }); a != 0 {
			t.Errorf("InterleaveN N=%d allocated %v times per run, want 0", nc, a)
		}
		outs := make([][]float32, nc)
		for c := range outs {
			outs[c] = make([]float32, n)
		}
		if a := testing.AllocsPerRun(50, func() { DeinterleaveN(outs, dst) }); a != 0 {
			t.Errorf("DeinterleaveN N=%d allocated %v times per run, want 0", nc, a)
		}
	}
}
