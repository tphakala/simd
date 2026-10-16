//go:build amd64 || arm64

package f32

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// The tests in this file prove which path BiquadCascadeN takes for a pair of
// channels without any seam in the production code, the way
// biquad_dispatch_test.go does for BiquadCascade. They exploit one difference
// between the routes.
//
// The pair kernel packs both channels' states into biquadLanes before it writes
// either dst (biquadCascadePair: packPair, then the kernel call, then
// unpackPair). Per-channel calls run one channel after the other, so channel 1
// reads its state after channel 0 has written dst[0].
//
// The probe therefore gives channel 1 a state slice whose first element is the
// last element of dst[0]: x1 of its first section. The layout deliberately
// violates BiquadCascadeN's rule that state must not overlap dst, because it is
// what makes the route observable. The mirror overlap (channel 0's state on
// dst[1]) makes the probe symmetric. The last element of each dst is left out
// of the comparison: the final state write-back lands there on every route. The
// rest of dst equals one of two references:
//
//   - a Go run over clean (zero, non-aliased) states, when the pair kernel packed
//     both states up front: the pair route;
//   - a Go run, channel by channel, over the same aliased layout, when the
//     channels ran one after the other: the sequential route (the kernel or Go,
//     which give the same bits).

type biquadPairRoute int

const (
	biquadRoutePair       biquadPairRoute = iota // both states packed before any dst write
	biquadRouteSequential                        // channel 1 read its state after channel 0 wrote dst
	biquadRouteOther                             // neither reference
)

func (r biquadPairRoute) String() string {
	switch r {
	case biquadRoutePair:
		return "pair"
	case biquadRouteSequential:
		return "sequential"
	default:
		return "other"
	}
}

// biquadPairAliasRun runs fn on two channels of m samples each, with state[1]
// starting at the last element of dst[0] and state[0] at the last element of
// dst[1], and returns copies of dst[0][:m-1] and dst[1][:m-1].
func biquadPairAliasRun(fn biquadNFunc, m, sections int, coeffs []float32, src [][]float32) (d0, d1 []float32) {
	big0 := make([]float32, m-1+biquadStateLen*sections)
	big1 := make([]float32, m-1+biquadStateLen*sections)
	dst := [][]float32{big0[:m:m], big1[:m:m]}
	state := [][]float32{big1[m-1:], big0[m-1:]}
	fn(dst, src, coeffs, state)
	return slices.Clone(dst[0][:m-1]), slices.Clone(dst[1][:m-1])
}

// biquadPairClassifyRoute reports which route fn takes for two channels of m
// samples through the given number of sections.
func biquadPairClassifyRoute(t *testing.T, fn biquadNFunc, m, sections int) biquadPairRoute {
	t.Helper()
	r := rand.New(rand.NewPCG(uint64(m), uint64(sections)))
	coeffs := biquadRBJ(sections, r)
	src := [][]float32{biquadNoise(m, r), biquadNoise(m, r)}

	var clean [2][]float32
	for c := range clean {
		clean[c] = make([]float32, m)
		biquadCascadeGo(clean[c], src[c], coeffs, make([]float32, biquadStateLen*sections))
		clean[c] = clean[c][:m-1]
	}
	goFn := biquadNBlocksFunc(0, 0, 0)
	seq0, seq1 := biquadPairAliasRun(goFn, m, sections, coeffs, src)

	// The premise: with per-channel Go calls the overlap changes channel 1's
	// result. If it did not, the probe could not tell the routes apart, and it
	// must fail here instead of passing blindly.
	if slices.EqualFunc(clean[1], seq1, bitsEqF32) {
		t.Fatalf("probe premise broken at sections=%d m=%d: the aliased sequential result "+
			"equals the clean one, so the route cannot be told from the output", sections, m)
	}

	got0, got1 := biquadPairAliasRun(fn, m, sections, coeffs, src)
	switch {
	case slices.EqualFunc(got0, clean[0], bitsEqF32) && slices.EqualFunc(got1, clean[1], bitsEqF32):
		return biquadRoutePair
	case slices.EqualFunc(got0, seq0, bitsEqF32) && slices.EqualFunc(got1, seq1, bitsEqF32):
		return biquadRouteSequential
	}
	return biquadRouteOther
}

// biquadPairRouteRow is one probe input with the route it must take on a host
// with a kernel. A host without one (lanes 0) is sequential for every row.
type biquadPairRouteRow struct {
	sections, m int
	want        biquadPairRoute
}

// biquadPairRouteRows derives the rows from biquadPairMinWork, so a retune of the
// floor moves the boundary rows with it. The fixed rows only mean what their
// comments say while the floor is even and large enough for the 2-section row
// to meet the kernel precondition m >= biquadSkew.
func biquadPairRouteRows(t *testing.T) []biquadPairRouteRow {
	t.Helper()
	pm := biquadPairMinWork
	if pm%2 != 0 || pm < 2*biquadSkew {
		t.Fatalf("biquadPairMinWork = %d: the route table assumes an even value of at least %d; "+
			"rederive the rows", pm, 2*biquadSkew)
	}
	return []biquadPairRouteRow{
		// Floor on m*sections.
		{1, pm - 1, biquadRouteSequential},
		{1, pm, biquadRoutePair},
		{2, pm/2 - 1, biquadRouteSequential},
		{2, pm / 2, biquadRoutePair},
		// Three sections never take the pair kernel, however long.
		{3, 200, biquadRouteSequential},
		{8, 200, biquadRouteSequential},
	}
}

func checkBiquadPairRoutes(t *testing.T, name string, fn biquadNFunc, lanes int) {
	t.Helper()
	for _, row := range biquadPairRouteRows(t) {
		want := row.want
		if lanes == 0 {
			want = biquadRouteSequential
		}
		if got := biquadPairClassifyRoute(t, fn, row.m, row.sections); got != want {
			t.Errorf("%s lanes=%d sections=%d m=%d: route %v, want %v", name, lanes, row.sections, row.m, got, want)
		}
	}
}

// TestBiquadCascadeNRoutesToPair drives the public BiquadCascadeN and checks it
// reaches the pair kernel, or the per-channel path, as biquadKernelLanes says it
// should. The expectation follows biquadKernelLanes, which
// TestBiquadKernelLanesMatchesCPU_* pins. It reads package state and must not
// call t.Parallel().
func TestBiquadCascadeNRoutesToPair(t *testing.T) {
	lanes := biquadKernelLanes()
	t.Logf("biquadKernelLanes() = %d", lanes)
	checkBiquadPairRoutes(t, "BiquadCascadeN", BiquadCascadeN, lanes)
}

// TestBiquadCascadeNBlocksRoutes drives biquadCascadeNBlocks directly at every
// lane count the host can execute, with the production floors, and at lanes 0,
// where every row must be sequential.
func TestBiquadCascadeNBlocksRoutes(t *testing.T) {
	checkBiquadPairRoutes(t, "biquadCascadeNBlocks", biquadNBlocksFunc(0, biquadMinWork, biquadPairMinWork), 0)
	lanesList := biquadRunnableLanes()
	if len(lanesList) == 0 {
		t.Skip("no biquad kernel can run on this CPU")
	}
	for _, lanes := range lanesList {
		checkBiquadPairRoutes(t, "biquadCascadeNBlocks", biquadNBlocksFunc(lanes, biquadMinWork, biquadPairMinWork), lanes)
	}
}

// TestBiquadCascadeNPairFloor pins the floor straight on the predicate the pair
// driver uses, including the kernel precondition.
func TestBiquadCascadeNPairFloor(t *testing.T) {
	pm := biquadPairMinWork
	cases := []struct {
		m, sections int
		want        bool
	}{
		{0, 1, false},
		{1, 2, false}, // below biquadSkew*(sections-1)
		{pm - 1, 1, false},
		{pm, 1, true},
		{pm/2 - 1, 2, false},
		{pm / 2, 2, true},
	}
	for _, c := range cases {
		if got := biquadKernelOK(c.m, c.sections, pm); got != c.want {
			t.Errorf("biquadKernelOK(m=%d, sections=%d, pairMinWork=%d) = %v, want %v", c.m, c.sections, pm, got, c.want)
		}
	}
}
