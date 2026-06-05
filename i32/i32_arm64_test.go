//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestInterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)

		gotNEON := make([]int32, n*2)
		gotGo := make([]int32, n*2)
		interleave2NEON(gotNEON, a, b)
		interleave2Go(gotGo, a, b)

		for i := range gotGo {
			if gotNEON[i] != gotGo[i] {
				t.Fatalf("n=%d: interleave2NEON[%d] = %d, want %d (Go)", n, i, gotNEON[i], gotGo[i])
			}
		}
	}
}

func TestDeinterleave2NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		src := make([]int32, n*2)
		for i := range src {
			src[i] = int32(i) ^ math.MinInt32
		}

		aNEON := make([]int32, n)
		bNEON := make([]int32, n)
		aGo := make([]int32, n)
		bGo := make([]int32, n)
		deinterleave2NEON(aNEON, bNEON, src)
		deinterleave2Go(aGo, bGo, src)

		for i := range aGo {
			if aNEON[i] != aGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON a[%d] = %d, want %d (Go)", n, i, aNEON[i], aGo[i])
			}
			if bNEON[i] != bGo[i] {
				t.Fatalf("n=%d: deinterleave2NEON b[%d] = %d, want %d (Go)", n, i, bNEON[i], bGo[i])
			}
		}
	}
}

// TestInterleave2NEON_NoOverwrite guards the scalar tail: the kernel must not
// write past n*2 output elements when n is not a multiple of the 4-lane block.
func TestInterleave2NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 17
	a := make([]int32, n)
	b := make([]int32, n)
	fillPattern(a, b)
	dst := make([]int32, n*2+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	interleave2NEON(dst[:n*2], a, b)
	for i := n * 2; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("interleave2NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// fillDiffSrc fills src with values that exercise the sign bit and force the
// residual to wrap int32 at the extremes.
func fillDiffSrc(src []int32) {
	for i := range src {
		src[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(src) > 1 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
	}
}

func TestAddSubNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)
		for _, tc := range []struct {
			name string
			simd func(dst, a, b []int32)
			ref  func(dst, a, b []int32)
		}{
			{"add", addNEON, addGo},
			{"sub", subNEON, subGo},
		} {
			gotNEON := make([]int32, n)
			gotGo := make([]int32, n)
			tc.simd(gotNEON, a, b)
			tc.ref(gotGo, a, b)
			for i := range gotGo {
				if gotNEON[i] != gotGo[i] {
					t.Fatalf("n=%d: %sNEON[%d] = %d, want %d (Go)", n, tc.name, i, gotNEON[i], gotGo[i])
				}
			}
		}
	}
}

func TestMidSideNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		left := make([]int32, n)
		right := make([]int32, n)
		fillPattern(left, right)

		midNEON := make([]int32, n)
		sideNEON := make([]int32, n)
		midGo := make([]int32, n)
		sideGo := make([]int32, n)
		midSideEncodeNEON(midNEON, sideNEON, left, right)
		midSideEncodeGo(midGo, sideGo, left, right)
		for i := range midGo {
			if midNEON[i] != midGo[i] || sideNEON[i] != sideGo[i] {
				t.Fatalf("n=%d: midSideEncodeNEON[%d] = (%d,%d), want (%d,%d)", n, i, midNEON[i], sideNEON[i], midGo[i], sideGo[i])
			}
		}

		lNEON := make([]int32, n)
		rNEON := make([]int32, n)
		lGo := make([]int32, n)
		rGo := make([]int32, n)
		midSideDecodeNEON(lNEON, rNEON, midNEON, sideNEON)
		midSideDecodeGo(lGo, rGo, midGo, sideGo)
		for i := range lGo {
			if lNEON[i] != lGo[i] || rNEON[i] != rGo[i] {
				t.Fatalf("n=%d: midSideDecodeNEON[%d] = (%d,%d), want (%d,%d)", n, i, lNEON[i], rNEON[i], lGo[i], rGo[i])
			}
		}
	}
}

func TestDiffNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	kernels := []struct {
		name string
		simd func(dst, src []int32)
		ref  func(dst, src []int32)
	}{
		{"diff1", diff1NEON, diff1Go},
		{"diff2", diff2NEON, diff2Go},
		{"diff3", diff3NEON, diff3Go},
		{"diff4", diff4NEON, diff4Go},
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // the kernel's warm-up reads assume len >= order; the
			// dispatch only calls it for len >= minNEONElements, routing
			// shorter inputs to the Go path.
		}
		src := make([]int32, n)
		fillDiffSrc(src)
		for _, k := range kernels {
			gotNEON := make([]int32, n)
			gotGo := make([]int32, n)
			k.simd(gotNEON, src)
			k.ref(gotGo, src)
			for i := range gotGo {
				if gotNEON[i] != gotGo[i] {
					t.Fatalf("n=%d: %sNEON[%d] = %d, want %d (Go)", n, k.name, i, gotNEON[i], gotGo[i])
				}
			}
		}
	}
}

func TestCumsumNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // the dispatch routes shorter inputs to the Go path; the
			// kernel assumes at least one full block (its scalar tail reads the
			// previous cumulative value from a[-1] of the tail).
		}
		gotNEON := make([]int32, n)
		gotGo := make([]int32, n)
		fillDiffSrc(gotNEON)
		copy(gotGo, gotNEON)
		cumsumNEON(gotNEON)
		cumsumGo(gotGo)
		for i := range gotGo {
			if gotNEON[i] != gotGo[i] {
				t.Fatalf("n=%d: cumsumNEON[%d] = %d, want %d (Go)", n, i, gotNEON[i], gotGo[i])
			}
		}
	}
}

// TestCumsumNEON_NoOverwrite guards the scalar tail: the in-place kernel must
// touch exactly n elements when n is not a multiple of the 4-lane block.
func TestCumsumNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 17
	buf := make([]int32, n+4)
	for i := range buf {
		buf[i] = math.MaxInt32 // sentinel
	}
	in := make([]int32, n)
	fillDiffSrc(in)
	copy(buf[:n], in)
	cumsumNEON(buf[:n])
	for i := n; i < len(buf); i++ {
		if buf[i] != math.MaxInt32 {
			t.Errorf("cumsumNEON wrote past end at buf[%d] = %d", i, buf[i])
		}
	}
}

// TestNEONKernels_AllocFree asserts each NEON kernel runs allocation-free, the
// repo's zero-allocation contract enforced directly at the kernel boundary
// (the public-API alloc tests cover the dispatch, these cover the kernels).
func TestNEONKernels_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	a := make([]int32, n)
	b := make([]int32, n)
	dst := make([]int32, n)
	dst2 := make([]int32, n)
	var fasSums [5]uint64
	riceHi := make([]uint64, riceMaxParam5+1-riceParamCount)
	checks := []struct {
		name string
		fn   func()
	}{
		{"addNEON", func() { addNEON(dst, a, b) }},
		{"subNEON", func() { subNEON(dst, a, b) }},
		{"midSideEncodeNEON", func() { midSideEncodeNEON(dst, dst2, a, b) }},
		{"midSideDecodeNEON", func() { midSideDecodeNEON(dst, dst2, a, b) }},
		{"diff1NEON", func() { diff1NEON(dst, a) }},
		{"diff2NEON", func() { diff2NEON(dst, a) }},
		{"diff3NEON", func() { diff3NEON(dst, a) }},
		{"diff4NEON", func() { diff4NEON(dst, a) }},
		{"cumsumNEON", func() { cumsumNEON(dst) }},
		{"zigzagSumNEON", func() { _ = zigzagSumNEON(a) }},
		{"minMaxNEON", func() { _, _ = minMaxNEON(a) }},
		{"fixedAbsSumsNEON", func() { fixedAbsSumsNEON(a, &fasSums) }},
		{"riceSumsHighNEON", func() { riceSumsHighNEON(riceHi, a) }},
		{"lpcResidualEncodeNEON", func() { lpcResidualEncodeNEON(dst, a, lpcAllocCoeffs, 12) }},
		{"lpcRestoreNEON", func() { lpcRestoreNEON(dst, a, lpcAllocCoeffs, 12) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

func TestLPCResidualEncodeNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		samples := make([]int32, n)
		fillLPCSamples(samples)
		for _, coeffs := range lpcCoeffSets() {
			order := len(coeffs)
			if n-order < minNEONElements {
				continue // the dispatch routes these to the Go path
			}
			for _, shift := range lpcShifts {
				gotNEON := make([]int32, n)
				gotGo := make([]int32, n)
				lpcResidualEncodeNEON(gotNEON, samples, coeffs, shift)
				lpcResidualEncodeGo(gotGo, samples, coeffs, shift)
				for i := range gotGo {
					if gotNEON[i] != gotGo[i] {
						t.Fatalf("n=%d order=%d shift=%d lpcResidualEncodeNEON[%d] = %d, want %d (Go)",
							n, order, shift, i, gotNEON[i], gotGo[i])
					}
				}
			}
		}
	}
}

func TestLPCRestoreNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		res := make([]int32, n)
		fillLPCSamples(res)
		for _, coeffs := range lpcCoeffSets() {
			order := len(coeffs)
			if order < minNEONRestoreOrder || order > maxLPCRestoreOrder || n-order < 1 {
				continue
			}
			rc := reverseCoeffs(coeffs)
			for _, shift := range lpcShifts {
				gotNEON := make([]int32, n)
				gotGo := make([]int32, n)
				lpcRestoreNEON(gotNEON, res, rc, shift)
				lpcRestoreGo(gotGo, res, coeffs, shift)
				for i := range gotGo {
					if gotNEON[i] != gotGo[i] {
						t.Fatalf("n=%d order=%d shift=%d lpcRestoreNEON[%d] = %d, want %d (Go)",
							n, order, shift, i, gotNEON[i], gotGo[i])
					}
				}
			}
		}
	}
}

// TestLPCResidualEncodeNEON_NoOverwrite guards the scalar tail: an order-4
// predictor over 23 outputs is 4 full 4-output blocks plus a 3-output tail.
func TestLPCResidualEncodeNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 23
	coeffs := []int32{8000, -5000, 2000, -512}
	samples := make([]int32, n)
	fillLPCSamples(samples)
	res := make([]int32, n+4)
	for i := range res {
		res[i] = math.MaxInt32 // sentinel
	}
	lpcResidualEncodeNEON(res[:n], samples, coeffs, 9)
	for i := n; i < len(res); i++ {
		if res[i] != math.MaxInt32 {
			t.Errorf("lpcResidualEncodeNEON wrote past end at res[%d] = %d", i, res[i])
		}
	}
}

// TestLPCRestoreNEON_NoOverwrite guards the output tail: an order-12 predictor
// over 35 outputs exercises both the vector groups and the scalar tap remainder.
func TestLPCRestoreNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 35
	coeffs := make([]int32, 12)
	for j := range coeffs {
		coeffs[j] = int32(1000 - 137*j)
	}
	res := make([]int32, n)
	fillLPCSamples(res)
	rc := reverseCoeffs(coeffs)
	out := make([]int32, n+4)
	for i := range out {
		out[i] = math.MaxInt32 // sentinel
	}
	lpcRestoreNEON(out[:n], res, rc, 10)
	for i := n; i < len(out); i++ {
		if out[i] != math.MaxInt32 {
			t.Errorf("lpcRestoreNEON wrote past end at out[%d] = %d", i, out[i])
		}
	}
}

// TestDiff1NEON_NoOverwrite guards the scalar tail.
func TestDiff1NEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 17
	src := make([]int32, n)
	fillDiffSrc(src)
	dst := make([]int32, n+4)
	for i := range dst {
		dst[i] = math.MaxInt32
	}
	diff1NEON(dst[:n], src)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("diff1NEON wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

func TestRiceSumsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		gotNEON := make([]uint64, riceParamCount)
		gotGo := make([]uint64, riceParamCount)
		riceSumsNEON(gotNEON, res)
		riceSumsGo(gotGo, res)
		for k := range gotGo {
			if gotNEON[k] != gotGo[k] {
				t.Fatalf("n=%d: riceSumsNEON[%d] = %d, want %d (Go)", n, k, gotNEON[k], gotGo[k])
			}
		}
	}
}

// TestRiceSumsNEON_NoOverwrite guards the fixed 15-wide write: the kernel must
// fill exactly riceParamCount sums and touch nothing past them.
func TestRiceSumsNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 100
	res := make([]int32, n)
	fillRiceRes(res)
	sums := make([]uint64, riceParamCount+4)
	for i := range sums {
		sums[i] = math.MaxUint64 // sentinel
	}
	riceSumsNEON(sums[:riceParamCount], res)
	for i := riceParamCount; i < len(sums); i++ {
		if sums[i] != math.MaxUint64 {
			t.Errorf("riceSumsNEON wrote past end at sums[%d] = %d", i, sums[i])
		}
	}
}

func TestZigzagSumNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		got := zigzagSumNEON(res)
		want := zigzagSumGo(res)
		if got != want {
			t.Fatalf("n=%d: zigzagSumNEON = %d, want %d (Go)", n, got, want)
		}
	}
}

func TestMinMaxNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	// A tame in-range body keeps the planted extremes the unique min/max, so a
	// kernel that drops a vector lane or skips the scalar tail is caught: one
	// variant plants the extremes in a mid-block lane and in the tail, the other
	// swaps them, covering both a dropped lane and a dropped tail on both reduces.
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		for _, swap := range []bool{false, true} {
			res := make([]int32, n)
			for i := range res {
				res[i] = int32(i%13) - 6
			}
			mid, tail := int32(math.MinInt32), int32(math.MaxInt32)
			if swap {
				mid, tail = tail, mid
			}
			res[n/2] = mid
			res[n-1] = tail
			gotMin, gotMax := minMaxNEON(res)
			wantMin, wantMax := minMaxGo(res)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxNEON = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}

func TestFixedAbsSumsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		src := make([]int32, n)
		fillDiffSrc(src) // sign-exercising values incl. MinInt32/MaxInt32 extremes
		var gotNEON, gotGo [5]uint64
		fixedAbsSumsNEON(src, &gotNEON)
		fixedAbsSumsGo(src, &gotGo)
		if gotNEON != gotGo {
			t.Fatalf("n=%d: fixedAbsSumsNEON = %v, want %v (Go)", n, gotNEON, gotGo)
		}
	}
}

// TestRiceSumsHighNEON_ParityWithGo checks the upper-half kernel (columns
// 15..30) against the pure-Go reference's matching columns.
func TestRiceSumsHighNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const lo = riceParamCount         // 15
	const hi = riceMaxParam5 + 1 - lo // 16 columns: k=15..30
	for _, n := range paritySizes {
		if n < minNEONElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		gotHigh := make([]uint64, hi)
		riceSumsHighNEON(gotHigh, res)
		full := make([]uint64, riceMaxParam5+1)
		riceSumsGo(full, res)
		for j := range gotHigh {
			if gotHigh[j] != full[lo+j] {
				t.Fatalf("n=%d: riceSumsHighNEON[%d] (k=%d) = %d, want %d (Go)", n, j, lo+j, gotHigh[j], full[lo+j])
			}
		}
	}
}

// TestRiceSumsHighNEON_NoOverwrite guards the fixed 16-wide write.
func TestRiceSumsHighNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const hi = riceMaxParam5 + 1 - riceParamCount // 16
	const n = 100
	res := make([]int32, n)
	fillRiceRes(res)
	sums := make([]uint64, hi+4)
	for i := range sums {
		sums[i] = math.MaxUint64 // sentinel
	}
	riceSumsHighNEON(sums[:hi], res)
	for i := hi; i < len(sums); i++ {
		if sums[i] != math.MaxUint64 {
			t.Errorf("riceSumsHighNEON wrote past end at sums[%d] = %d", i, sums[i])
		}
	}
}
