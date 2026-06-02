package i32

import (
	"math"
	"math/rand"
	"testing"
)

// Tests for the Rice zigzag cost search (RiceSums / RiceBestParam).
//
// zigzag maps a signed residual to its unsigned Rice symbol,
// zigzag(r) = (r<<1) ^ (r>>31): 0,-1,1,-2,2 -> 0,1,2,3,4. RiceSums returns
// sums[k] = Σ_i (zigzag(res[i]) >> k), the unary-bit total for Rice parameter k;
// the full code cost of parameter k over n residuals is sums[k] + n*(k+1).
// RiceBestParam scans k and returns the cost-minimizing parameter.

// zigzag is the scalar reference fold used by the oracle below.
func zigzag(r int32) uint64 {
	return uint64(uint32((r << 1) ^ (r >> 31)))
}

// riceSumsOracle is an independent, obviously-correct implementation of the
// per-parameter unary-bit sums, written separately from riceSumsGo so the two
// agreeing is a real cross-check rather than a tautology.
func riceSumsOracle(res []int32, params int) []uint64 {
	sums := make([]uint64, params)
	for _, r := range res {
		u := zigzag(r)
		for k := range params {
			sums[k] += u >> uint(k)
		}
	}
	return sums
}

func TestRiceSumsSmall(t *testing.T) {
	res := []int32{0, -1, 1, -2, 2} // zigzag -> 0,1,2,3,4
	sums := make([]uint64, 4)
	RiceSums(sums, res)
	want := []uint64{10, 4, 1, 0}
	for k := range want {
		if sums[k] != want[k] {
			t.Errorf("RiceSums[%d] = %d, want %d", k, sums[k], want[k])
		}
	}
}

func TestRiceBestParamSmall(t *testing.T) {
	res := []int32{0, -1, 1, -2, 2}
	// cost(0)=10+5=15, cost(1)=4+10=14, cost(2)=1+15=16, cost(3)=0+20=20.
	gotK, gotBits := RiceBestParam(res, 14)
	if gotK != 1 || gotBits != 14 {
		t.Errorf("RiceBestParam = (%d, %d), want (1, 14)", gotK, gotBits)
	}
}

func TestRiceSumsMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 1024, 1025} {
		res := make([]int32, n)
		for i := range res {
			res[i] = int32(rng.Uint32())
		}
		sums := make([]uint64, riceParamCount)
		RiceSums(sums, res)
		want := riceSumsOracle(res, riceParamCount)
		for k := range want {
			if sums[k] != want[k] {
				t.Fatalf("n=%d RiceSums[%d] = %d, want %d", n, k, sums[k], want[k])
			}
		}
	}
}

// TestRiceSumsExtremes exercises the int32 sign bit and the zigzag overflow at
// math.MinInt32 (zigzag = 2^32-1), the worst case for the unsigned widening.
func TestRiceSumsExtremes(t *testing.T) {
	res := []int32{math.MinInt32, math.MaxInt32, -1, 0, math.MinInt32, math.MaxInt32, 1, -2, 3}
	sums := make([]uint64, riceParamCount)
	RiceSums(sums, res)
	want := riceSumsOracle(res, riceParamCount)
	for k := range want {
		if sums[k] != want[k] {
			t.Errorf("RiceSums extremes [%d] = %d, want %d", k, sums[k], want[k])
		}
	}
}

func TestRiceBestParamMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 8, 9, 17, 100, 1000} {
		res := make([]int32, n)
		// Bias toward small magnitudes so the optimum parameter is interior.
		for i := range res {
			res[i] = int32(rng.Intn(2000) - 1000)
		}
		for _, maxParam := range []uint{0, 5, 14, 30} {
			gotK, gotBits := RiceBestParam(res, maxParam)
			// brute force over the same range using the oracle
			sums := riceSumsOracle(res, int(maxParam)+1)
			wantK, wantBits := uint(0), sums[0]+uint64(n)
			for k := 1; k <= int(maxParam); k++ {
				c := sums[k] + uint64(n)*uint64(k+1)
				if c < wantBits {
					wantBits = c
					wantK = uint(k)
				}
			}
			if gotK != wantK || gotBits != wantBits {
				t.Errorf("n=%d maxParam=%d RiceBestParam=(%d,%d), want (%d,%d)", n, maxParam, gotK, gotBits, wantK, wantBits)
			}
		}
	}
}

func TestRiceBestParamEmpty(t *testing.T) {
	k, bits := RiceBestParam(nil, 14)
	if k != 0 || bits != 0 {
		t.Errorf("RiceBestParam(nil) = (%d, %d), want (0, 0)", k, bits)
	}
}

func TestRiceSumsClearsDst(t *testing.T) {
	// RiceSums must fully overwrite sums, not accumulate into existing content.
	res := []int32{1, 2, 3}
	sums := make([]uint64, riceParamCount)
	for i := range sums {
		sums[i] = 999
	}
	RiceSums(sums, res)
	want := riceSumsOracle(res, riceParamCount)
	for k := range want {
		if sums[k] != want[k] {
			t.Errorf("RiceSums did not clear dst [%d] = %d, want %d", k, sums[k], want[k])
		}
	}
}

func TestRiceAllocFree(t *testing.T) {
	res := make([]int32, 1024)
	for i := range res {
		res[i] = int32(i*7 - 3)
	}
	sums := make([]uint64, riceParamCount)
	if got := testing.AllocsPerRun(100, func() { RiceSums(sums, res) }); got != 0 {
		t.Errorf("RiceSums allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { RiceBestParam(res, 14) }); got != 0 {
		t.Errorf("RiceBestParam allocated %v times per run, want 0", got)
	}
}
