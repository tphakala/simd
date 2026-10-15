//go:build amd64 || arm64

package f32

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// The tests in this file prove which path BiquadCascade takes without any seam in
// the production code. They exploit one difference between the two paths.
//
// The Go path runs one section at a time (biquadCascadeGo), so section k reads
// its state only after sections 0..k-1 have written dst. A kernel pass packs the
// state of every section in the pass into biquadLanes before it writes anything
// (biquadCascadeBlocks: pack, then the kernel call, then unpack).
//
// The probe therefore gives the last section's state the same memory as
// dst[0:4]. That overlap deliberately violates BiquadCascade's rule that state
// must not overlap dst (see the BiquadCascade doc comment in biquad.go), because
// it is what makes the route observable. dst[0:4] is left out of the comparison:
// the final state write-back lands there on both paths. The rest of dst equals
// one of two references, or neither:
//
//   - a Go run over clean (non-aliased) state, when a kernel packed the state up
//     front: the kernel route;
//   - a Go run over the same aliased layout, when the Go path ran;
//   - neither, when a second pass of a multi-pass cascade packed the state after
//     the first pass had already written dst: the later route.

type biquadRoute int

const (
	biquadRouteKernel biquadRoute = iota // aliased state was packed before any dst write
	biquadRouteGo                        // section-by-section Go order
	biquadRouteLater                     // packed by a pass after an earlier pass wrote dst
)

func (r biquadRoute) String() string {
	switch r {
	case biquadRouteKernel:
		return "kernel"
	case biquadRouteGo:
		return "go"
	default:
		return "later"
	}
}

// biquadAliasRun runs fn with the last section's state placed at dst[0:4] and
// returns a copy of dst[biquadStateLen:].
func biquadAliasRun(fn biquadFunc, n, sections int, coeffs, src []float32) []float32 {
	off := biquadStateLen * (sections - 1)
	buf := make([]float32, off+n)
	state := buf[:biquadStateLen*sections]
	dst := buf[off:]
	fn(dst, src, coeffs, state)
	return slices.Clone(dst[biquadStateLen:])
}

// biquadClassifyRoute reports which route fn takes for n samples through the
// given number of sections.
func biquadClassifyRoute(t *testing.T, fn biquadFunc, n, sections int) biquadRoute {
	t.Helper()
	r := rand.New(rand.NewPCG(uint64(n), uint64(sections)))
	coeffs := biquadRBJ(sections, r)
	src := biquadNoise(n, r)

	clean := make([]float32, n)
	biquadCascadeGo(clean, src, coeffs, make([]float32, biquadStateLen*sections))
	cleanTail := clean[biquadStateLen:]
	aliasedGo := biquadAliasRun(biquadCascadeGo, n, sections, coeffs, src)

	// The premise: on the Go path the overlap changes the result. If a refactor
	// made the Go path read its state up front too, the probe could no longer
	// tell the routes apart, and it must fail here instead of passing blindly.
	if slices.EqualFunc(cleanTail, aliasedGo, bitsEqF32) {
		t.Fatalf("probe premise broken at sections=%d n=%d: the aliased Go result "+
			"equals the clean one, so the route cannot be told from the output", sections, n)
	}

	got := biquadAliasRun(fn, n, sections, coeffs, src)
	switch {
	case slices.EqualFunc(got, cleanTail, bitsEqF32):
		return biquadRouteKernel
	case slices.EqualFunc(got, aliasedGo, bitsEqF32):
		return biquadRouteGo
	}
	return biquadRouteLater
}

// biquadRouteRow is one probe input with the route it must take at each lane
// count. A host with no kernel (lanes 0) takes the Go path for every row.
type biquadRouteRow struct {
	sections, n    int
	lanes8, lanes4 biquadRoute
}

func (r biquadRouteRow) want(lanes int) biquadRoute {
	switch lanes {
	case biquadMaxLanes:
		return r.lanes8
	case biquadNarrowLanes:
		return r.lanes4
	}
	return biquadRouteGo
}

// biquadRouteRows derives the rows from biquadMinWork, so a retune of the floor
// moves the boundary rows with it. The fixed rows only mean what their comments
// say while the floor sits in the range the guards below check.
func biquadRouteRows(t *testing.T) []biquadRouteRow {
	t.Helper()
	m := biquadMinWork
	// The 8-section rows at n = 13 and n = 14 test the skew precondition only
	// while 13*8 clears the floor; the lanes-4 row at n = 14 runs 4 sections per
	// pass, so it is below the floor only while 14*4 < m.
	if m%4 != 0 || m > 13*biquadMaxLanes || m <= 14*biquadNarrowLanes {
		t.Fatalf("biquadMinWork = %d: the route table assumes a multiple of 4 in (%d, %d]; "+
			"rederive the rows", m, 14*biquadNarrowLanes, 13*biquadMaxLanes)
	}
	return []biquadRouteRow{
		// Floor on n*sections, for the sizes that take the 4-section kernel.
		{2, m/2 - 1, biquadRouteGo, biquadRouteGo},
		{2, m / 2, biquadRouteKernel, biquadRouteKernel},
		{4, m/4 - 1, biquadRouteGo, biquadRouteGo},
		{4, m / 4, biquadRouteKernel, biquadRouteKernel},
		// Skew precondition n >= biquadSkew*(sections-1), above the floor.
		{8, 13, biquadRouteGo, biquadRouteGo},
		{8, 14, biquadRouteKernel, biquadRouteGo},
		// Two passes at lanes 4: the second packs after the first wrote dst.
		{8, 64, biquadRouteKernel, biquadRouteLater},
	}
}

func checkBiquadRoutes(t *testing.T, name string, fn biquadFunc, lanes int) {
	t.Helper()
	for _, row := range biquadRouteRows(t) {
		want := row.want(lanes)
		if got := biquadClassifyRoute(t, fn, row.n, row.sections); got != want {
			t.Errorf("%s lanes=%d sections=%d n=%d: route %v, want %v",
				name, lanes, row.sections, row.n, got, want)
		}
	}
}

// biquadBlocksFuncFloor adapts biquadCascadeBlocks to biquadFunc with the given
// lane count and the production performance floor, unlike biquadBlocksFunc which
// turns the floor off.
func biquadBlocksFuncFloor(lanes int) biquadFunc {
	return func(dst, src, coeffs, state []float32) {
		biquadCascadeBlocks(dst, src, coeffs, state, lanes, biquadMinWork)
	}
}

// TestBiquadCascadeRoutesToKernel drives the public BiquadCascade and checks it
// reaches a kernel, or the Go path, as biquadKernelLanes says it should. The
// expectation follows biquadKernelLanes, so on its own this does not notice a
// wrong lane count; TestBiquadKernelLanesMatchesCPU_* pins that value.
// It reads package state and must not call t.Parallel().
func TestBiquadCascadeRoutesToKernel(t *testing.T) {
	public := func(dst, src, coeffs, state []float32) { BiquadCascade(dst, src, coeffs, state) }
	lanes := biquadKernelLanes()
	t.Logf("biquadKernelLanes() = %d", lanes)
	checkBiquadRoutes(t, "BiquadCascade", public, lanes)
}

// TestBiquadCascadeBlocksRoutesByLanes drives biquadCascadeBlocks directly at
// every lane count the host can execute, with the production floor, so the 4-lane
// pass structure is exercised on hosts whose public path uses 8 lanes.
func TestBiquadCascadeBlocksRoutesByLanes(t *testing.T) {
	lanesList := biquadRunnableLanes()
	if len(lanesList) == 0 {
		t.Skip("no biquad kernel can run on this CPU")
	}
	for _, lanes := range lanesList {
		checkBiquadRoutes(t, "biquadCascadeBlocks", biquadBlocksFuncFloor(lanes), lanes)
	}
}

// TestBiquadKernelOK pins the kernel precondition and the floor straight on
// biquadKernelOK.
func TestBiquadKernelOK(t *testing.T) {
	m := biquadMinWork
	cases := []struct {
		n, sections, minWork int
		want                 bool
	}{
		{0, 1, 0, false}, // nothing to process
		{1, 1, 0, true},
		{13, 8, 0, false}, // below biquadSkew*(sections-1) = 14
		{14, 8, 0, true},
		{m/2 - 1, 2, m, false}, // just under the floor
		{m / 2, 2, m, true},    // exactly at the floor
		{m/4 - 1, 4, m, false},
		{m / 4, 4, m, true},
	}
	for _, c := range cases {
		if got := biquadKernelOK(c.n, c.sections, c.minWork); got != c.want {
			t.Errorf("biquadKernelOK(n=%d, sections=%d, minWork=%d) = %v, want %v",
				c.n, c.sections, c.minWork, got, c.want)
		}
	}
}
