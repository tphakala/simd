package i32

import (
	"math"
	"math/big"
	"testing"
)

// Tests for the quantized-LPC encode FIR (LPCResidualEncode) and its decode
// inverse (LPCRestore), the Phase 3 kernels of the FLAC integer roadmap.
//
// LPCResidualEncode writes the residual a FLAC LPC subframe Rice-codes:
//
//	res[i] = samples[i]                                       for i < order
//	res[i] = samples[i] - int32((Σ_j coeffs[j]*samples[i-1-j]) >> shift) for i >= order
//
// LPCRestore inverts it with the serial recurrence:
//
//	out[i] = residual[i]                                      for i < order
//	out[i] = residual[i] + int32((Σ_j coeffs[j]*out[i-1-j]) >> shift)   for i >= order
//
// The prediction sum is accumulated in int64 (matching libFLAC); only the
// final >>shift result is truncated to int32. The strongest correctness check
// is the LPCRestore(LPCResidualEncode(x)) == x round-trip, because the forward
// FIR and the backward recurrence are independent kernels. lpcPredictOracle is
// a math/big reference that depends on neither the int64 accumulation width nor
// the package's shift handling, so its agreement pins the exact semantics.

// lpcPredictOracle returns int32((Σ_j coeffs[j]*window[j]) >> shift) computed in
// arbitrary precision. window[j] is the sample multiplied by coeffs[j]; for the
// encode/decode loops window[j] = src[i-1-j]. big.Int Rsh is floor division by
// 2^shift, matching Go's arithmetic >> on a signed integer, and masking the low
// 32 bits then reinterpreting reproduces the int32() truncation exactly, without
// assuming the running sum fits in int64.
func lpcPredictOracle(coeffs, window []int32, shift uint) int32 {
	acc := new(big.Int)
	term := new(big.Int)
	for j := range coeffs {
		term.Mul(big.NewInt(int64(coeffs[j])), big.NewInt(int64(window[j])))
		acc.Add(acc, term)
	}
	acc.Rsh(acc, shift) // arithmetic shift right = floor(acc / 2^shift)
	low := new(big.Int).And(acc, big.NewInt(0xFFFFFFFF))
	return int32(uint32(low.Uint64()))
}

// lpcEncodeOracle is the independent encode reference built on lpcPredictOracle.
func lpcEncodeOracle(res, samples, coeffs []int32, shift uint) {
	order := len(coeffs)
	n := min(len(res), len(samples))
	w := min(order, n)
	copy(res[:w], samples[:w])
	window := make([]int32, order)
	for i := order; i < n; i++ {
		for j := range order {
			window[j] = samples[i-1-j]
		}
		res[i] = samples[i] - lpcPredictOracle(coeffs, window, shift)
	}
}

// lpcRestoreOracle is the independent decode reference built on lpcPredictOracle.
func lpcRestoreOracle(out, residual, coeffs []int32, shift uint) {
	order := len(coeffs)
	n := min(len(out), len(residual))
	w := min(order, n)
	copy(out[:w], residual[:w])
	window := make([]int32, order)
	for i := order; i < n; i++ {
		for j := range order {
			window[j] = out[i-1-j]
		}
		out[i] = residual[i] + lpcPredictOracle(coeffs, window, shift)
	}
}

// lpcSizes straddle both SIMD block sizes (4 lanes on NEON, 8 on AVX) and
// include inputs at or below the predictor order (all warm-up, Go path).
var lpcSizes = []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 1000, 1024, 1025}

// lpcCoeffSets are representative quantized-LPC coefficient vectors. Orders span
// 1..32 (FLAC's range), values stay within ~15-bit qlp precision, and the signs
// vary so the int64 accumulation and the >>shift sign handling are exercised.
func lpcCoeffSets() [][]int32 {
	// An order-12 and an order-32 set with alternating-sign, decaying magnitudes.
	orders := []int{12, 32}
	sets := make([][]int32, 0, 4+len(orders))
	sets = append(sets,
		[]int32{4096},
		[]int32{7000, -3000},
		[]int32{8000, -5000, 2000, -512},
		[]int32{6000, -4000, 3000, -2000, 1500, -1000, 700, -300},
	)
	for _, order := range orders {
		c := make([]int32, order)
		for j := range c {
			v := int32(16000 >> (j / 4))
			if j%2 == 1 {
				v = -v
			}
			c[j] = v
		}
		sets = append(sets, c)
	}
	return sets
}

var lpcShifts = []uint{0, 1, 9, 14, 15}

// lpcAllocCoeffs is an order-8 coefficient vector for the kernel allocation-free
// assertions (the per-architecture *_test.go files reference it).
var lpcAllocCoeffs = []int32{6000, -4000, 3000, -2000, 1500, -1000, 700, -300}

// reverseCoeffs returns coeffs reversed, the layout the SIMD restore kernels
// consume (rcoeffs[k] = coeffs[order-1-k]).
func reverseCoeffs(coeffs []int32) []int32 {
	rc := make([]int32, len(coeffs))
	for k := range coeffs {
		rc[k] = coeffs[len(coeffs)-1-k]
	}
	return rc
}

// fillLPCSamples fills src with values that exercise the sign bit and the int32
// extremes. Round-trip holds for any int32 input because encode and decode use
// identical arithmetic, so extreme inputs are fair game and stress wraparound.
func fillLPCSamples(src []int32) {
	for i := range src {
		src[i] = int32(uint32(i)*2654435761) ^ (int32(i) << 20)
	}
	if len(src) > 2 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
		src[2] = -1
	}
}

func TestLPCRoundTrip(t *testing.T) {
	for _, n := range lpcSizes {
		samples := make([]int32, n)
		fillLPCSamples(samples)
		for _, coeffs := range lpcCoeffSets() {
			for _, shift := range lpcShifts {
				res := make([]int32, n)
				LPCResidualEncode(res, samples, coeffs, shift)
				got := make([]int32, n)
				LPCRestore(got, res, coeffs, shift)
				for i := range samples {
					if got[i] != samples[i] {
						t.Fatalf("n=%d order=%d shift=%d LPCRestore(LPCResidualEncode)[%d] = %d, want %d",
							n, len(coeffs), shift, i, got[i], samples[i])
					}
				}
			}
		}
	}
}

func TestLPCEncodeMatchesOracle(t *testing.T) {
	for _, n := range lpcSizes {
		samples := make([]int32, n)
		fillLPCSamples(samples)
		for _, coeffs := range lpcCoeffSets() {
			for _, shift := range lpcShifts {
				got := make([]int32, n)
				want := make([]int32, n)
				LPCResidualEncode(got, samples, coeffs, shift)
				lpcEncodeOracle(want, samples, coeffs, shift)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("n=%d order=%d shift=%d LPCResidualEncode[%d] = %d, want %d (big.Int oracle)",
							n, len(coeffs), shift, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestLPCRestoreMatchesOracle(t *testing.T) {
	for _, n := range lpcSizes {
		res := make([]int32, n)
		fillLPCSamples(res) // treat as an arbitrary residual stream
		for _, coeffs := range lpcCoeffSets() {
			for _, shift := range lpcShifts {
				got := make([]int32, n)
				want := make([]int32, n)
				LPCRestore(got, res, coeffs, shift)
				lpcRestoreOracle(want, res, coeffs, shift)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("n=%d order=%d shift=%d LPCRestore[%d] = %d, want %d (big.Int oracle)",
							n, len(coeffs), shift, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestLPCEncodeSimple checks a hand-computed order-2 case with shift.
func TestLPCEncodeSimple(t *testing.T) {
	// coeffs={2,-1}, shift=1. pred[i] = (2*s[i-1] - s[i-2]) >> 1.
	samples := []int32{10, 20, 34, 50}
	// i=2: pred=(2*20-10)>>1=(30)>>1=15; res=34-15=19
	// i=3: pred=(2*34-20)>>1=(48)>>1=24; res=50-24=26
	want := []int32{10, 20, 19, 26}
	res := make([]int32, len(samples))
	LPCResidualEncode(res, samples, []int32{2, -1}, 1)
	for i, w := range want {
		if res[i] != w {
			t.Errorf("LPCResidualEncode[%d] = %d, want %d", i, res[i], w)
		}
	}
}

// TestLPCRestoreSimple inverts TestLPCEncodeSimple's residual.
func TestLPCRestoreSimple(t *testing.T) {
	res := []int32{10, 20, 19, 26}
	want := []int32{10, 20, 34, 50}
	out := make([]int32, len(res))
	LPCRestore(out, res, []int32{2, -1}, 1)
	for i, w := range want {
		if out[i] != w {
			t.Errorf("LPCRestore[%d] = %d, want %d", i, out[i], w)
		}
	}
}

// TestLPCWraps verifies int32 truncation of the prediction matches the oracle at
// the type extremes.
func TestLPCWraps(t *testing.T) {
	samples := []int32{math.MaxInt32, math.MinInt32, math.MaxInt32, math.MinInt32, 1, -1, 0, 7, 11, 13}
	coeffs := []int32{12345, -6789}
	got := make([]int32, len(samples))
	want := make([]int32, len(samples))
	LPCResidualEncode(got, samples, coeffs, 3)
	lpcEncodeOracle(want, samples, coeffs, 3)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("LPCResidualEncode wrap [%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestLPC_Clamp(t *testing.T) {
	samples := []int32{1, 2, 4, 7, 11, 16, 22, 29, 37, 46}
	coeffs := []int32{2, -1}
	res := make([]int32, 100)
	LPCResidualEncode(res, samples, coeffs, 0)
	want := make([]int32, len(samples))
	lpcEncodeOracle(want, samples, coeffs, 0)
	for i, w := range want {
		if res[i] != w {
			t.Errorf("LPCResidualEncode clamp res[%d] = %d, want %d", i, res[i], w)
		}
	}
	for i := len(want); i < len(res); i++ {
		if res[i] != 0 {
			t.Errorf("LPCResidualEncode wrote past clamp at res[%d] = %d, want untouched 0", i, res[i])
		}
	}
}

// TestLPC_ShortInput checks inputs at or below the order: with order or fewer
// samples there is no residual, only warm-up copied verbatim.
func TestLPC_ShortInput(t *testing.T) {
	coeffs := []int32{1, 2, 3, 4} // order 4
	samples := []int32{42, 7, 9}  // only 3 samples
	res := make([]int32, len(samples))
	LPCResidualEncode(res, samples, coeffs, 2)
	if res[0] != 42 || res[1] != 7 || res[2] != 9 {
		t.Errorf("LPCResidualEncode short input = %v, want [42 7 9] warm-up", res)
	}
	out := make([]int32, len(res))
	LPCRestore(out, res, coeffs, 2)
	if out[0] != 42 || out[1] != 7 || out[2] != 9 {
		t.Errorf("LPCRestore short input = %v, want [42 7 9] warm-up", out)
	}
}

// TestLPC_ZeroOrder treats an empty coefficient vector as identity: no predictor,
// every output is warm-up.
func TestLPC_ZeroOrder(t *testing.T) {
	samples := []int32{3, 1, 4, 1, 5}
	res := make([]int32, len(samples))
	LPCResidualEncode(res, samples, nil, 4)
	for i := range samples {
		if res[i] != samples[i] {
			t.Errorf("LPCResidualEncode zero order res[%d] = %d, want %d", i, res[i], samples[i])
		}
	}
}

// TestLPC_ShiftClamp checks that an out-of-range shift (>= 64) is clamped to 63
// at the public boundary, so the Go and SIMD paths stay consistent (FLAC never
// uses such shifts; this guards a caller passing a bad value). It also confirms
// the round-trip still holds and that shift 64..67 all behave like 63.
func TestLPC_ShiftClamp(t *testing.T) {
	const n = 200
	samples := make([]int32, n)
	fillLPCSamples(samples)
	coeffs := []int32{6000, -4000, 3000, -2000, 1500, -1000, 700, -300}

	ref := make([]int32, n)
	LPCResidualEncode(ref, samples, coeffs, 63)
	for _, shift := range []uint{64, 65, 100, 1 << 20} {
		got := make([]int32, n)
		LPCResidualEncode(got, samples, coeffs, shift)
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("shift=%d not clamped to 63: res[%d] = %d, want %d", shift, i, got[i], ref[i])
			}
		}
		out := make([]int32, n)
		LPCRestore(out, got, coeffs, shift)
		for i := range samples {
			if out[i] != samples[i] {
				t.Fatalf("shift=%d round-trip[%d] = %d, want %d", shift, i, out[i], samples[i])
			}
		}
	}
}

func TestLPC_Empty(t *testing.T) {
	coeffs := []int32{1, -1}
	LPCResidualEncode(nil, nil, coeffs, 0)
	LPCResidualEncode([]int32{}, []int32{}, coeffs, 0)
	LPCRestore(nil, nil, coeffs, 0)
	res := []int32{99}
	LPCResidualEncode(res, nil, coeffs, 0)
	if res[0] != 99 {
		t.Errorf("LPCResidualEncode wrote on empty input: %v", res)
	}
}

func TestLPC_AllocFree(t *testing.T) {
	const n = 1024
	samples := make([]int32, n)
	fillLPCSamples(samples)
	res := make([]int32, n)
	out := make([]int32, n)
	for _, coeffs := range lpcCoeffSets() {
		name := "order" + itoa(len(coeffs))
		if got := testing.AllocsPerRun(50, func() { LPCResidualEncode(res, samples, coeffs, 12) }); got != 0 {
			t.Errorf("LPCResidualEncode %s allocated %v times per run, want 0", name, got)
		}
		if got := testing.AllocsPerRun(50, func() { LPCRestore(out, res, coeffs, 12) }); got != 0 {
			t.Errorf("LPCRestore %s allocated %v times per run, want 0", name, got)
		}
	}
}

// itoa is a tiny allocation-free-friendly int formatter for subtest labels.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [12]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
