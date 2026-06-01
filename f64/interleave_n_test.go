package f64

import (
	"fmt"
	"testing"
)

// interleaveNRef is the independent scalar oracle for InterleaveN:
// dst[i*nc+c] = srcs[c][i]. It is the most literal transcription of the
// contract so it cannot share a bug with the production generic path.
func interleaveNRef(dst []float64, srcs [][]float64, n int) {
	nc := len(srcs)
	for i := range n {
		for c := range nc {
			dst[i*nc+c] = srcs[c][i]
		}
	}
}

// deinterleaveNRef is the independent scalar oracle for DeinterleaveN:
// dsts[c][i] = src[i*nc+c].
func deinterleaveNRef(dsts [][]float64, src []float64, n int) {
	nc := len(dsts)
	for i := range n {
		for c := range nc {
			dsts[c][i] = src[i*nc+c]
		}
	}
}

// chanVal returns a value unique to (channel, frame) so a misplaced lane is
// always caught.
func chanVal(c, i int) float64 { return float64(c*100000 + i) }

var interleaveNFrameCounts = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 255, 1000}

var interleaveNStreamCounts = []int{1, 2, 3, 4, 5, 6, 8}

func TestInterleaveN_Parity(t *testing.T) {
	const pad = 5
	const sentinel = float64(-7777)
	for _, nc := range interleaveNStreamCounts {
		for _, n := range interleaveNFrameCounts {
			t.Run(fmt.Sprintf("N=%d/n=%d", nc, n), func(t *testing.T) {
				srcs := make([][]float64, nc)
				for c := range srcs {
					srcs[c] = make([]float64, n)
					for i := range srcs[c] {
						srcs[c][i] = chanVal(c, i)
					}
				}
				dst := make([]float64, (n+pad)*nc)
				for i := range dst {
					dst[i] = sentinel
				}
				InterleaveN(dst, srcs)

				want := make([]float64, (n+pad)*nc)
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
	const sentinel = float64(-7777)
	for _, nc := range interleaveNStreamCounts {
		for _, n := range interleaveNFrameCounts {
			t.Run(fmt.Sprintf("N=%d/n=%d", nc, n), func(t *testing.T) {
				src := make([]float64, n*nc)
				for i := range src {
					src[i] = float64(i + 1)
				}
				dsts := make([][]float64, nc)
				want := make([][]float64, nc)
				for c := range dsts {
					dsts[c] = make([]float64, n+pad)
					want[c] = make([]float64, n+pad)
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

func TestInterleaveN_MatchesInterleave2(t *testing.T) {
	for _, n := range interleaveNFrameCounts {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range n {
			a[i] = float64(i) * 1.5
			b[i] = float64(i)*-2.25 + 0.5
		}
		got := make([]float64, n*2)
		want := make([]float64, n*2)
		InterleaveN(got, [][]float64{a, b})
		Interleave2(want, a, b)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d: InterleaveN[%d]=%v, Interleave2=%v", n, i, got[i], want[i])
			}
		}
	}
}

func TestDeinterleaveN_MatchesDeinterleave2(t *testing.T) {
	for _, n := range interleaveNFrameCounts {
		src := make([]float64, n*2)
		for i := range src {
			src[i] = float64(i)*0.75 - 3
		}
		gotA := make([]float64, n)
		gotB := make([]float64, n)
		wantA := make([]float64, n)
		wantB := make([]float64, n)
		DeinterleaveN([][]float64{gotA, gotB}, src)
		Deinterleave2(wantA, wantB, src)
		for i := range n {
			if gotA[i] != wantA[i] || gotB[i] != wantB[i] {
				t.Fatalf("n=%d: i=%d got (%v,%v) want (%v,%v)", n, i, gotA[i], gotB[i], wantA[i], wantB[i])
			}
		}
	}
}

func TestInterleaveN_RaggedClamp(t *testing.T) {
	srcs := [][]float64{
		{1, 2, 3, 4, 5},
		{10, 20, 30},
		{100, 200, 300, 400},
	}
	const sentinel = float64(-1)
	dst := make([]float64, 20)
	for i := range dst {
		dst[i] = sentinel
	}
	InterleaveN(dst, srcs)
	want := []float64{1, 10, 100, 2, 20, 200, 3, 30, 300}
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

func TestInterleaveN_ShortDst(t *testing.T) {
	srcs := [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	dst := make([]float64, 7)
	InterleaveN(dst, srcs)
	want := []float64{1, 5, 9, 2, 6, 10, 0}
	for i, w := range want {
		if dst[i] != w {
			t.Fatalf("dst[%d]=%v want %v", i, dst[i], w)
		}
	}
}

func TestDeinterleaveN_RaggedClamp(t *testing.T) {
	src := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	const sentinel = float64(-1)
	dsts := [][]float64{
		make([]float64, 5),
		make([]float64, 2),
		make([]float64, 4),
	}
	for c := range dsts {
		for i := range dsts[c] {
			dsts[c][i] = sentinel
		}
	}
	DeinterleaveN(dsts, src)
	wantPrefix := [][]float64{{1, 4}, {2, 5}, {3, 6}}
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

func TestInterleaveN_RoundTrip(t *testing.T) {
	for _, nc := range interleaveNStreamCounts {
		for _, n := range []int{1, 3, 4, 8, 17, 100} {
			srcs := make([][]float64, nc)
			for c := range srcs {
				srcs[c] = make([]float64, n)
				for i := range srcs[c] {
					srcs[c][i] = chanVal(c, i)
				}
			}
			inter := make([]float64, n*nc)
			InterleaveN(inter, srcs)
			out := make([][]float64, nc)
			for c := range out {
				out[c] = make([]float64, n)
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

func TestInterleaveN_Empty(_ *testing.T) {
	InterleaveN(nil, nil)
	InterleaveN([]float64{1, 2, 3}, nil)
	InterleaveN(nil, [][]float64{{1}, {2}})
	DeinterleaveN(nil, nil)
	DeinterleaveN(nil, []float64{1, 2, 3})
	DeinterleaveN([][]float64{{0}, {0}}, nil)
	InterleaveN([]float64{9}, [][]float64{{}})
}

func TestInterleaveN_AllocFree(t *testing.T) {
	for _, nc := range interleaveNStreamCounts {
		const n = 512
		srcs := make([][]float64, nc)
		for c := range srcs {
			srcs[c] = make([]float64, n)
		}
		dst := make([]float64, n*nc)
		dsts := make([][]float64, nc)
		for c := range dsts {
			dsts[c] = make([]float64, n)
		}
		src := make([]float64, n*nc)

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
	srcs := make([][]float64, nc)
	for c := range srcs {
		srcs[c] = make([]float64, n)
		for i := range srcs[c] {
			srcs[c][i] = chanVal(c, i)
		}
	}
	dst := make([]float64, n*nc)
	b.SetBytes(int64(n * nc * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		InterleaveN(dst, srcs)
	}
}

func benchInterleaveNScalar(b *testing.B, nc int) {
	b.Helper()
	const n = 1024
	srcs := make([][]float64, nc)
	for c := range srcs {
		srcs[c] = make([]float64, n)
	}
	dst := make([]float64, n*nc)
	b.SetBytes(int64(n * nc * 8))
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
	src := make([]float64, n*nc)
	for i := range src {
		src[i] = float64(i)
	}
	dsts := make([][]float64, nc)
	for c := range dsts {
		dsts[c] = make([]float64, n)
	}
	b.SetBytes(int64(n * nc * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeinterleaveN(dsts, src)
	}
}

func BenchmarkDeinterleaveN_N3(b *testing.B) { benchDeinterleaveN(b, 3) }
func BenchmarkDeinterleaveN_N4(b *testing.B) { benchDeinterleaveN(b, 4) }
func BenchmarkDeinterleaveN_N6(b *testing.B) { benchDeinterleaveN(b, 6) }
func BenchmarkDeinterleaveN_N8(b *testing.B) { benchDeinterleaveN(b, 8) }
