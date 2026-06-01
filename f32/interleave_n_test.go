package f32

import (
	"fmt"
	"testing"
)

// interleaveNRef is the independent scalar oracle for InterleaveN:
// dst[i*nc+c] = srcs[c][i]. It is intentionally the most literal possible
// transcription of the contract so it cannot share a bug with the production
// generic path (interleaveNGo).
func interleaveNRef(dst []float32, srcs [][]float32, n int) {
	nc := len(srcs)
	for i := range n {
		for c := range nc {
			dst[i*nc+c] = srcs[c][i]
		}
	}
}

// deinterleaveNRef is the independent scalar oracle for DeinterleaveN:
// dsts[c][i] = src[i*nc+c].
func deinterleaveNRef(dsts [][]float32, src []float32, n int) {
	nc := len(dsts)
	for i := range n {
		for c := range nc {
			dsts[c][i] = src[i*nc+c]
		}
	}
}

// chanVal returns a value unique to (channel, frame) so any lane that lands in
// the wrong slot is caught: a transpose bug cannot accidentally produce a
// matching value.
func chanVal(c, i int) float32 { return float32(c*100000 + i) }

// interleaveNFrameCounts spans the SIMD block boundaries of every kernel
// (NEON ST3/ST4 and AMD64 4x4/8x8 process 4 or 8 frames per iteration) plus
// their +-1 tails and some larger sizes.
var interleaveNFrameCounts = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 255, 1000}

// interleaveNStreamCounts covers the SIMD-specialized N and unspecialized N
// (1, 5) that must fall back to the generic path.
var interleaveNStreamCounts = []int{1, 2, 3, 4, 5, 6, 8}

func TestInterleaveN_Parity(t *testing.T) {
	const pad = 5 // extra dst frames that must stay untouched
	const sentinel = float32(-7777)
	for _, nc := range interleaveNStreamCounts {
		for _, n := range interleaveNFrameCounts {
			t.Run(fmt.Sprintf("N=%d/n=%d", nc, n), func(t *testing.T) {
				srcs := make([][]float32, nc)
				for c := range srcs {
					srcs[c] = make([]float32, n)
					for i := range srcs[c] {
						srcs[c][i] = chanVal(c, i)
					}
				}
				dst := make([]float32, (n+pad)*nc)
				for i := range dst {
					dst[i] = sentinel
				}
				InterleaveN(dst, srcs)

				want := make([]float32, (n+pad)*nc)
				for i := range want {
					want[i] = sentinel
				}
				interleaveNRef(want, srcs, n)

				for i := range dst {
					if dst[i] != want[i] {
						t.Fatalf("dst[%d] = %v, want %v (n=%d nc=%d)", i, dst[i], want[i], n, nc)
					}
				}
			})
		}
	}
}

func TestDeinterleaveN_Parity(t *testing.T) {
	const pad = 5
	const sentinel = float32(-7777)
	for _, nc := range interleaveNStreamCounts {
		for _, n := range interleaveNFrameCounts {
			t.Run(fmt.Sprintf("N=%d/n=%d", nc, n), func(t *testing.T) {
				src := make([]float32, n*nc)
				for i := range src {
					src[i] = float32(i + 1)
				}
				dsts := make([][]float32, nc)
				want := make([][]float32, nc)
				for c := range dsts {
					dsts[c] = make([]float32, n+pad)
					want[c] = make([]float32, n+pad)
					for i := range dsts[c] {
						dsts[c][i] = sentinel
						want[c][i] = sentinel
					}
				}
				DeinterleaveN(dsts, src)
				deinterleaveNRef(want, src, n)

				for c := range dsts {
					for i := range dsts[c] {
						if dsts[c][i] != want[c][i] {
							t.Fatalf("dsts[%d][%d] = %v, want %v (n=%d nc=%d)", c, i, dsts[c][i], want[c][i], n, nc)
						}
					}
				}
			})
		}
	}
}

// TestInterleaveN_MatchesInterleave2 pins the N==2 path to the existing
// Interleave2 result exactly (the contract requires bit-identical output).
func TestInterleaveN_MatchesInterleave2(t *testing.T) {
	for _, n := range interleaveNFrameCounts {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range n {
			a[i] = float32(i) * 1.5
			b[i] = float32(i)*-2.25 + 0.5
		}
		got := make([]float32, n*2)
		want := make([]float32, n*2)
		InterleaveN(got, [][]float32{a, b})
		Interleave2(want, a, b)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d: InterleaveN[%d]=%v, Interleave2=%v", n, i, got[i], want[i])
			}
		}
	}
}

// TestDeinterleaveN_MatchesDeinterleave2 pins the N==2 path to Deinterleave2.
func TestDeinterleaveN_MatchesDeinterleave2(t *testing.T) {
	for _, n := range interleaveNFrameCounts {
		src := make([]float32, n*2)
		for i := range src {
			src[i] = float32(i)*0.75 - 3
		}
		gotA := make([]float32, n)
		gotB := make([]float32, n)
		wantA := make([]float32, n)
		wantB := make([]float32, n)
		DeinterleaveN([][]float32{gotA, gotB}, src)
		Deinterleave2(wantA, wantB, src)
		for i := range n {
			if gotA[i] != wantA[i] || gotB[i] != wantB[i] {
				t.Fatalf("n=%d: i=%d got (%v,%v) want (%v,%v)", n, i, gotA[i], gotB[i], wantA[i], wantB[i])
			}
		}
	}
}

// TestInterleaveN_RaggedClamp checks that the frame count clamps to the shortest
// source and to len(dst)/N, and that nothing past the clamped region is written.
func TestInterleaveN_RaggedClamp(t *testing.T) {
	srcs := [][]float32{
		{1, 2, 3, 4, 5},
		{10, 20, 30}, // shortest -> clamps n to 3
		{100, 200, 300, 400},
	}
	const sentinel = float32(-1)
	dst := make([]float32, 20)
	for i := range dst {
		dst[i] = sentinel
	}
	InterleaveN(dst, srcs)
	// n clamps to 3, nc=3 -> 9 written.
	want := []float32{1, 10, 100, 2, 20, 200, 3, 30, 300}
	for i, w := range want {
		if dst[i] != w {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], w)
		}
	}
	for i := len(want); i < len(dst); i++ {
		if dst[i] != sentinel {
			t.Fatalf("dst[%d]=%v modified past clamp", i, dst[i])
		}
	}
}

// TestInterleaveN_ShortDst clamps on len(dst)/N when dst is the limiting factor.
func TestInterleaveN_ShortDst(t *testing.T) {
	srcs := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	dst := make([]float32, 7) // 7/3 = 2 frames
	InterleaveN(dst, srcs)
	want := []float32{1, 5, 9, 2, 6, 10, 0} // last element untouched (was 0)
	for i, w := range want {
		if dst[i] != w {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], w)
		}
	}
}

// TestDeinterleaveN_RaggedClamp mirrors the clamp behavior for the split path.
func TestDeinterleaveN_RaggedClamp(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	const sentinel = float32(-1)
	dsts := [][]float32{
		make([]float32, 5),
		make([]float32, 2), // shortest -> clamps n to 2
		make([]float32, 4),
	}
	for c := range dsts {
		for i := range dsts[c] {
			dsts[c][i] = sentinel
		}
	}
	DeinterleaveN(dsts, src)
	// n = min(12/3, 5, 2, 4) = 2
	wantPrefix := [][]float32{{1, 4}, {2, 5}, {3, 6}}
	for c := range wantPrefix {
		for i := range wantPrefix[c] {
			if dsts[c][i] != wantPrefix[c][i] {
				t.Fatalf("dsts[%d][%d]=%v want %v", c, i, dsts[c][i], wantPrefix[c][i])
			}
		}
		for i := len(wantPrefix[c]); i < len(dsts[c]); i++ {
			if dsts[c][i] != sentinel {
				t.Fatalf("dsts[%d][%d]=%v modified past clamp", c, i, dsts[c][i])
			}
		}
	}
}

// TestInterleaveN_RoundTrip interleaves then deinterleaves and expects identity.
func TestInterleaveN_RoundTrip(t *testing.T) {
	for _, nc := range interleaveNStreamCounts {
		for _, n := range []int{1, 3, 4, 8, 17, 100} {
			srcs := make([][]float32, nc)
			for c := range srcs {
				srcs[c] = make([]float32, n)
				for i := range srcs[c] {
					srcs[c][i] = chanVal(c, i)
				}
			}
			inter := make([]float32, n*nc)
			InterleaveN(inter, srcs)
			out := make([][]float32, nc)
			for c := range out {
				out[c] = make([]float32, n)
			}
			DeinterleaveN(out, inter)
			for c := range nc {
				for i := range n {
					if out[c][i] != srcs[c][i] {
						t.Fatalf("nc=%d n=%d round-trip out[%d][%d]=%v want %v", nc, n, c, i, out[c][i], srcs[c][i])
					}
				}
			}
		}
	}
}

// TestInterleaveN_Empty exercises the no-op guards (empty srcs/dsts, zero frames).
func TestInterleaveN_Empty(_ *testing.T) {
	InterleaveN(nil, nil)
	InterleaveN([]float32{1, 2, 3}, nil)
	InterleaveN(nil, [][]float32{{1}, {2}})
	DeinterleaveN(nil, nil)
	DeinterleaveN(nil, []float32{1, 2, 3})
	DeinterleaveN([][]float32{{0}, {0}}, nil)
	// A single empty source: nc=1, n=0 -> no-op.
	InterleaveN([]float32{9}, [][]float32{{}})
}

// TestInterleaveN_AllocFree asserts the streaming contract: no per-call allocs.
func TestInterleaveN_AllocFree(t *testing.T) {
	for _, nc := range interleaveNStreamCounts {
		const n = 512
		srcs := make([][]float32, nc)
		for c := range srcs {
			srcs[c] = make([]float32, n)
		}
		dst := make([]float32, n*nc)
		dsts := make([][]float32, nc)
		for c := range dsts {
			dsts[c] = make([]float32, n)
		}
		src := make([]float32, n*nc)

		if a := testing.AllocsPerRun(50, func() { InterleaveN(dst, srcs) }); a != 0 {
			t.Errorf("InterleaveN N=%d allocated %v times per run, want 0", nc, a)
		}
		if a := testing.AllocsPerRun(50, func() { DeinterleaveN(dsts, src) }); a != 0 {
			t.Errorf("DeinterleaveN N=%d allocated %v times per run, want 0", nc, a)
		}
	}
}

func benchInterleaveN(b *testing.B, nc int) {
	b.Helper()
	const n = 1024
	srcs := make([][]float32, nc)
	for c := range srcs {
		srcs[c] = make([]float32, n)
		for i := range srcs[c] {
			srcs[c][i] = chanVal(c, i)
		}
	}
	dst := make([]float32, n*nc)
	b.SetBytes(int64(n * nc * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		InterleaveN(dst, srcs)
	}
}

func benchInterleaveNScalar(b *testing.B, nc int) {
	b.Helper()
	const n = 1024
	srcs := make([][]float32, nc)
	for c := range srcs {
		srcs[c] = make([]float32, n)
	}
	dst := make([]float32, n*nc)
	b.SetBytes(int64(n * nc * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		interleaveNRef(dst, srcs, n)
	}
}

func BenchmarkInterleaveN_N3(b *testing.B)       { benchInterleaveN(b, 3) }
func BenchmarkInterleaveN_N4(b *testing.B)       { benchInterleaveN(b, 4) }
func BenchmarkInterleaveN_N6(b *testing.B)       { benchInterleaveN(b, 6) }
func BenchmarkInterleaveN_N8(b *testing.B)       { benchInterleaveN(b, 8) }
func BenchmarkInterleaveNScalar_N3(b *testing.B) { benchInterleaveNScalar(b, 3) }
func BenchmarkInterleaveNScalar_N6(b *testing.B) { benchInterleaveNScalar(b, 6) }

func benchDeinterleaveN(b *testing.B, nc int) {
	b.Helper()
	const n = 1024
	src := make([]float32, n*nc)
	for i := range src {
		src[i] = float32(i)
	}
	dsts := make([][]float32, nc)
	for c := range dsts {
		dsts[c] = make([]float32, n)
	}
	b.SetBytes(int64(n * nc * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeinterleaveN(dsts, src)
	}
}

func BenchmarkDeinterleaveN_N3(b *testing.B) { benchDeinterleaveN(b, 3) }
func BenchmarkDeinterleaveN_N4(b *testing.B) { benchDeinterleaveN(b, 4) }
func BenchmarkDeinterleaveN_N6(b *testing.B) { benchDeinterleaveN(b, 6) }
func BenchmarkDeinterleaveN_N8(b *testing.B) { benchDeinterleaveN(b, 8) }
