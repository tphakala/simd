//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestInterleave2AVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	for _, n := range paritySizes {
		a := make([]int32, n)
		b := make([]int32, n)
		fillPattern(a, b)

		gotAVX := make([]int32, n*2)
		gotGo := make([]int32, n*2)
		interleave2AVX(gotAVX, a, b)
		interleave2Go(gotGo, a, b)

		for i := range gotGo {
			if gotAVX[i] != gotGo[i] {
				t.Fatalf("n=%d: interleave2AVX[%d] = %d, want %d (Go)", n, i, gotAVX[i], gotGo[i])
			}
		}
	}
}

func TestDeinterleave2AVX_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	for _, n := range paritySizes {
		src := make([]int32, n*2)
		for i := range src {
			src[i] = int32(i) ^ math.MinInt32
		}

		aAVX := make([]int32, n)
		bAVX := make([]int32, n)
		aGo := make([]int32, n)
		bGo := make([]int32, n)
		deinterleave2AVX(aAVX, bAVX, src)
		deinterleave2Go(aGo, bGo, src)

		for i := range aGo {
			if aAVX[i] != aGo[i] {
				t.Fatalf("n=%d: deinterleave2AVX a[%d] = %d, want %d (Go)", n, i, aAVX[i], aGo[i])
			}
			if bAVX[i] != bGo[i] {
				t.Fatalf("n=%d: deinterleave2AVX b[%d] = %d, want %d (Go)", n, i, bAVX[i], bGo[i])
			}
		}
	}
}

// TestInterleave2AVX_NoOverwrite guards the scalar tail: the kernel must not
// write past n*2 output elements even when n is not a multiple of the block.
func TestInterleave2AVX_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX {
		t.Skip("AVX not available")
	}
	const n = 17
	a := make([]int32, n)
	b := make([]int32, n)
	fillPattern(a, b)
	dst := make([]int32, n*2+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	interleave2AVX(dst[:n*2], a, b)
	for i := n * 2; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("interleave2AVX wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// fillDiffSrc fills src with values that exercise the sign bit and force the
// residual to wrap int32 at the extremes, so a kernel that handled overflow
// differently than Go would be caught.
func fillDiffSrc(src []int32) {
	for i := range src {
		src[i] = int32(i*7-3) ^ math.MinInt32
	}
	if len(src) > 1 {
		src[0] = math.MaxInt32
		src[1] = math.MinInt32
	}
}

func TestAddSubAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
			{"add", addAVX2, addGo},
			{"sub", subAVX2, subGo},
		} {
			gotAVX := make([]int32, n)
			gotGo := make([]int32, n)
			tc.simd(gotAVX, a, b)
			tc.ref(gotGo, a, b)
			for i := range gotGo {
				if gotAVX[i] != gotGo[i] {
					t.Fatalf("n=%d: %sAVX2[%d] = %d, want %d (Go)", n, tc.name, i, gotAVX[i], gotGo[i])
				}
			}
		}
	}
}

func TestMidSideAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		left := make([]int32, n)
		right := make([]int32, n)
		fillPattern(left, right)

		midAVX := make([]int32, n)
		sideAVX := make([]int32, n)
		midGo := make([]int32, n)
		sideGo := make([]int32, n)
		midSideEncodeAVX2(midAVX, sideAVX, left, right)
		midSideEncodeGo(midGo, sideGo, left, right)
		for i := range midGo {
			if midAVX[i] != midGo[i] || sideAVX[i] != sideGo[i] {
				t.Fatalf("n=%d: midSideEncodeAVX2[%d] = (%d,%d), want (%d,%d)", n, i, midAVX[i], sideAVX[i], midGo[i], sideGo[i])
			}
		}

		lAVX := make([]int32, n)
		rAVX := make([]int32, n)
		lGo := make([]int32, n)
		rGo := make([]int32, n)
		midSideDecodeAVX2(lAVX, rAVX, midAVX, sideAVX)
		midSideDecodeGo(lGo, rGo, midGo, sideGo)
		for i := range lGo {
			if lAVX[i] != lGo[i] || rAVX[i] != rGo[i] {
				t.Fatalf("n=%d: midSideDecodeAVX2[%d] = (%d,%d), want (%d,%d)", n, i, lAVX[i], rAVX[i], lGo[i], rGo[i])
			}
		}
	}
}

func TestDiffAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	kernels := []struct {
		name string
		simd func(dst, src []int32)
		ref  func(dst, src []int32)
	}{
		{"diff1", diff1AVX2, diff1Go},
		{"diff2", diff2AVX2, diff2Go},
		{"diff3", diff3AVX2, diff3Go},
		{"diff4", diff4AVX2, diff4Go},
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // the kernel's warm-up reads assume len >= order; the
			// dispatch only calls it for len >= minAVXElements, routing
			// shorter inputs (and len < order) to the Go path.
		}
		src := make([]int32, n)
		fillDiffSrc(src)
		for _, k := range kernels {
			gotAVX := make([]int32, n)
			gotGo := make([]int32, n)
			k.simd(gotAVX, src)
			k.ref(gotGo, src)
			for i := range gotGo {
				if gotAVX[i] != gotGo[i] {
					t.Fatalf("n=%d: %sAVX2[%d] = %d, want %d (Go)", n, k.name, i, gotAVX[i], gotGo[i])
				}
			}
		}
	}
}

func TestCumsumAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // the dispatch routes shorter inputs to the Go path; the
			// kernel assumes at least one full block (its scalar tail reads the
			// previous cumulative value from a[-1] of the tail).
		}
		gotAVX := make([]int32, n)
		gotGo := make([]int32, n)
		fillDiffSrc(gotAVX)
		copy(gotGo, gotAVX)
		cumsumAVX2(gotAVX)
		cumsumGo(gotGo)
		for i := range gotGo {
			if gotAVX[i] != gotGo[i] {
				t.Fatalf("n=%d: cumsumAVX2[%d] = %d, want %d (Go)", n, i, gotAVX[i], gotGo[i])
			}
		}
	}
}

// TestCumsumAVX2_NoOverwrite guards the scalar tail: the in-place kernel must
// touch exactly n elements when n is not a multiple of the block.
func TestCumsumAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 17
	buf := make([]int32, n+4)
	for i := range buf {
		buf[i] = math.MaxInt32 // sentinel
	}
	in := make([]int32, n)
	fillDiffSrc(in)
	copy(buf[:n], in)
	cumsumAVX2(buf[:n])
	for i := n; i < len(buf); i++ {
		if buf[i] != math.MaxInt32 {
			t.Errorf("cumsumAVX2 wrote past end at buf[%d] = %d", i, buf[i])
		}
	}
}

// TestDiff1AVX2_NoOverwrite guards the scalar tail: the kernel must write exactly
// n elements and not run past the end when n is not a multiple of the block.
func TestDiff1AVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 17
	src := make([]int32, n)
	fillDiffSrc(src)
	dst := make([]int32, n+4)
	for i := range dst {
		dst[i] = math.MaxInt32 // sentinel
	}
	diff1AVX2(dst[:n], src)
	for i := n; i < len(dst); i++ {
		if dst[i] != math.MaxInt32 {
			t.Errorf("diff1AVX2 wrote past end at dst[%d] = %d", i, dst[i])
		}
	}
}

// TestAVX2Kernels_AllocFree asserts each AVX2 kernel runs allocation-free, the
// repo's zero-allocation contract enforced directly at the kernel boundary
// (the public-API alloc tests cover the dispatch, these cover the kernels).
func TestAVX2Kernels_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
		{"addAVX2", func() { addAVX2(dst, a, b) }},
		{"subAVX2", func() { subAVX2(dst, a, b) }},
		{"midSideEncodeAVX2", func() { midSideEncodeAVX2(dst, dst2, a, b) }},
		{"midSideDecodeAVX2", func() { midSideDecodeAVX2(dst, dst2, a, b) }},
		{"diff1AVX2", func() { diff1AVX2(dst, a) }},
		{"diff2AVX2", func() { diff2AVX2(dst, a) }},
		{"diff3AVX2", func() { diff3AVX2(dst, a) }},
		{"diff4AVX2", func() { diff4AVX2(dst, a) }},
		{"cumsumAVX2", func() { cumsumAVX2(dst) }},
		{"zigzagSumAVX2", func() { _ = zigzagSumAVX2(a) }},
		{"minMaxAVX2", func() { _, _ = minMaxAVX2(a) }},
		{"fixedAbsSumsAVX2", func() { fixedAbsSumsAVX2(a, &fasSums) }},
		{"riceSumsHighAVX2", func() { riceSumsHighAVX2(riceHi, a) }},
		{"lpcResidualEncodeAVX2", func() { lpcResidualEncodeAVX2(dst, a, lpcAllocCoeffs, 12) }},
		{"lpcRestoreAVX2", func() { lpcRestoreAVX2(dst, a, lpcAllocCoeffs, 12) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(100, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

func TestLPCResidualEncodeAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		samples := make([]int32, n)
		fillLPCSamples(samples)
		for _, coeffs := range lpcCoeffSets() {
			order := len(coeffs)
			if n-order < minAVXElements {
				continue // the dispatch routes these to the Go path; the kernel
				// assumes order >= 1 and at least one full 8-output block.
			}
			for _, shift := range lpcShifts {
				gotAVX := make([]int32, n)
				gotGo := make([]int32, n)
				lpcResidualEncodeAVX2(gotAVX, samples, coeffs, shift)
				lpcResidualEncodeGo(gotGo, samples, coeffs, shift)
				for i := range gotGo {
					if gotAVX[i] != gotGo[i] {
						t.Fatalf("n=%d order=%d shift=%d lpcResidualEncodeAVX2[%d] = %d, want %d (Go)",
							n, order, shift, i, gotAVX[i], gotGo[i])
					}
				}
			}
		}
	}
}

// TestLPCResidualEncodeAVX2_NoOverwrite guards the scalar tail: an order-4
// predictor over 23 samples is 2 full 8-output blocks plus a 3-output tail, so a
// kernel that ran past n would be caught here.
func TestLPCResidualEncodeAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 23
	coeffs := []int32{8000, -5000, 2000, -512}
	samples := make([]int32, n)
	fillLPCSamples(samples)
	res := make([]int32, n+4)
	for i := range res {
		res[i] = math.MaxInt32 // sentinel
	}
	lpcResidualEncodeAVX2(res[:n], samples, coeffs, 9)
	for i := n; i < len(res); i++ {
		if res[i] != math.MaxInt32 {
			t.Errorf("lpcResidualEncodeAVX2 wrote past end at res[%d] = %d", i, res[i])
		}
	}
}

func TestLPCRestoreAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		res := make([]int32, n)
		fillLPCSamples(res) // arbitrary residual stream
		for _, coeffs := range lpcCoeffSets() {
			order := len(coeffs)
			if order < minLPCRestoreOrder || order > maxLPCRestoreOrder || n-order < 1 {
				continue // dispatch routes these to the Go recurrence
			}
			rc := reverseCoeffs(coeffs)
			for _, shift := range lpcShifts {
				gotAVX := make([]int32, n)
				gotGo := make([]int32, n)
				lpcRestoreAVX2(gotAVX, res, rc, shift)
				lpcRestoreGo(gotGo, res, coeffs, shift)
				for i := range gotGo {
					if gotAVX[i] != gotGo[i] {
						t.Fatalf("n=%d order=%d shift=%d lpcRestoreAVX2[%d] = %d, want %d (Go)",
							n, order, shift, i, gotAVX[i], gotGo[i])
					}
				}
			}
		}
	}
}

// TestLPCRestoreAVX2_NoOverwrite guards the output tail: an order-12 predictor
// over 35 outputs exercises both the vector groups and the scalar tap remainder.
func TestLPCRestoreAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
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
	lpcRestoreAVX2(out[:n], res, rc, 10)
	for i := n; i < len(out); i++ {
		if out[i] != math.MaxInt32 {
			t.Errorf("lpcRestoreAVX2 wrote past end at out[%d] = %d", i, out[i])
		}
	}
}

// TestMidSideEncodeAVX2_NoOverwrite guards both output tails.
func TestMidSideEncodeAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 17
	left := make([]int32, n)
	right := make([]int32, n)
	fillPattern(left, right)
	mid := make([]int32, n+4)
	side := make([]int32, n+4)
	for i := range mid {
		mid[i] = math.MaxInt32
		side[i] = math.MaxInt32
	}
	midSideEncodeAVX2(mid[:n], side[:n], left, right)
	for i := n; i < len(mid); i++ {
		if mid[i] != math.MaxInt32 {
			t.Errorf("midSideEncodeAVX2 wrote past mid end at [%d] = %d", i, mid[i])
		}
		if side[i] != math.MaxInt32 {
			t.Errorf("midSideEncodeAVX2 wrote past side end at [%d] = %d", i, side[i])
		}
	}
}

func TestRiceSumsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		gotAVX := make([]uint64, riceParamCount)
		gotGo := make([]uint64, riceParamCount)
		riceSumsAVX2(gotAVX, res)
		riceSumsGo(gotGo, res)
		for k := range gotGo {
			if gotAVX[k] != gotGo[k] {
				t.Fatalf("n=%d: riceSumsAVX2[%d] = %d, want %d (Go)", n, k, gotAVX[k], gotGo[k])
			}
		}
	}
}

// TestRiceSumsAVX2_NoOverwrite guards the fixed 15-wide write: the kernel must
// fill exactly riceParamCount sums and touch nothing past them.
func TestRiceSumsAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 100
	res := make([]int32, n)
	fillRiceRes(res)
	sums := make([]uint64, riceParamCount+4)
	for i := range sums {
		sums[i] = math.MaxUint64 // sentinel
	}
	riceSumsAVX2(sums[:riceParamCount], res)
	for i := riceParamCount; i < len(sums); i++ {
		if sums[i] != math.MaxUint64 {
			t.Errorf("riceSumsAVX2 wrote past end at sums[%d] = %d", i, sums[i])
		}
	}
}

func TestZigzagSumAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		got := zigzagSumAVX2(res)
		want := zigzagSumGo(res)
		if got != want {
			t.Fatalf("n=%d: zigzagSumAVX2 = %d, want %d (Go)", n, got, want)
		}
	}
}

func TestMinMaxAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	// A tame in-range body keeps the planted extremes the unique min/max, so a
	// kernel that drops a vector lane or skips the scalar tail is caught: one
	// variant plants the extremes in a mid-block lane and in the tail, the other
	// swaps them, covering both a dropped lane and a dropped tail on both reduces.
	for _, n := range paritySizes {
		if n < minAVXElements {
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
			gotMin, gotMax := minMaxAVX2(res)
			wantMin, wantMax := minMaxGo(res)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxAVX2 = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}

func TestFixedAbsSumsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		src := make([]int32, n)
		fillDiffSrc(src) // sign-exercising values incl. MinInt32/MaxInt32 extremes
		var gotAVX, gotGo [5]uint64
		fixedAbsSumsAVX2(src, &gotAVX)
		fixedAbsSumsGo(src, &gotGo)
		if gotAVX != gotGo {
			t.Fatalf("n=%d: fixedAbsSumsAVX2 = %v, want %v (Go)", n, gotAVX, gotGo)
		}
	}
}

// TestRiceSumsHighAVX2_ParityWithGo checks the upper-half kernel (columns
// 15..30) against the pure-Go reference's matching columns.
func TestRiceSumsHighAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const lo = riceParamCount         // 15
	const hi = riceMaxParam5 + 1 - lo // 16 columns: k=15..30
	for _, n := range paritySizes {
		if n < minAVXElements {
			continue // dispatch routes these to Go; the kernel is not called
		}
		res := make([]int32, n)
		fillRiceRes(res)
		gotHigh := make([]uint64, hi)
		riceSumsHighAVX2(gotHigh, res)
		full := make([]uint64, riceMaxParam5+1)
		riceSumsGo(full, res)
		for j := range gotHigh {
			if gotHigh[j] != full[lo+j] {
				t.Fatalf("n=%d: riceSumsHighAVX2[%d] (k=%d) = %d, want %d (Go)", n, j, lo+j, gotHigh[j], full[lo+j])
			}
		}
	}
}

// TestRiceSumsHighAVX2_NoOverwrite guards the fixed 16-wide write.
func TestRiceSumsHighAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const hi = riceMaxParam5 + 1 - riceParamCount // 16
	const n = 100
	res := make([]int32, n)
	fillRiceRes(res)
	sums := make([]uint64, hi+4)
	for i := range sums {
		sums[i] = math.MaxUint64 // sentinel
	}
	riceSumsHighAVX2(sums[:hi], res)
	for i := hi; i < len(sums); i++ {
		if sums[i] != math.MaxUint64 {
			t.Errorf("riceSumsHighAVX2 wrote past end at sums[%d] = %d", i, sums[i])
		}
	}
}
