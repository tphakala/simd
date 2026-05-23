//go:build amd64

package f64

import (
	"reflect"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// snapshotImpls captures every dispatch function pointer plus minSIMDElements
// so a test that calls one of the init* functions can restore them in t.Cleanup.
type implSnapshot struct {
	dotProduct        dotProductFunc
	add               binaryOpFunc
	sub               binaryOpFunc
	mul               binaryOpFunc
	div               binaryOpFunc
	scale             scaleFunc
	addScalar         scaleFunc
	sum               reduceFunc
	min               reduceFunc
	max               reduceFunc
	abs               unaryOpFunc
	neg               unaryOpFunc
	sqrt              unaryOpFunc
	reciprocal        unaryOpFunc
	round             unaryOpFunc
	fma               fmaFunc
	clamp             clampFunc
	variance          varianceFunc
	euclideanDistance euclideanDistanceFunc
	interleave2       interleave2Func
	deinterleave2     deinterleave2Func
	addScaled         addScaledFunc
	minSIMDElements   int
}

func snapshotDispatch() implSnapshot {
	return implSnapshot{
		dotProduct:        dotProductImpl,
		add:               addImpl,
		sub:               subImpl,
		mul:               mulImpl,
		div:               divImpl,
		scale:             scaleImpl,
		addScalar:         addScalarImpl,
		sum:               sumImpl,
		min:               minImpl,
		max:               maxImpl,
		abs:               absImpl,
		neg:               negImpl,
		sqrt:              sqrtImpl,
		reciprocal:        reciprocalImpl,
		round:             roundImpl,
		fma:               fmaImpl,
		clamp:             clampImpl,
		variance:          varianceImpl,
		euclideanDistance: euclideanDistanceImpl,
		interleave2:       interleave2Impl,
		deinterleave2:     deinterleave2Impl,
		addScaled:         addScaledImpl,
		minSIMDElements:   minSIMDElements,
	}
}

func restoreDispatch(s implSnapshot) {
	dotProductImpl = s.dotProduct
	addImpl = s.add
	subImpl = s.sub
	mulImpl = s.mul
	divImpl = s.div
	scaleImpl = s.scale
	addScalarImpl = s.addScalar
	sumImpl = s.sum
	minImpl = s.min
	maxImpl = s.max
	absImpl = s.abs
	negImpl = s.neg
	sqrtImpl = s.sqrt
	reciprocalImpl = s.reciprocal
	roundImpl = s.round
	fmaImpl = s.fma
	clampImpl = s.clamp
	varianceImpl = s.variance
	euclideanDistanceImpl = s.euclideanDistance
	interleave2Impl = s.interleave2
	deinterleave2Impl = s.deinterleave2
	addScaledImpl = s.addScaled
	minSIMDElements = s.minSIMDElements
}

func samePointer(a, b interface{}) bool {
	return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer()
}

// Tests for init functions to ensure they properly configure function pointers
// These tests are AMD64-specific because they test x86 SIMD initialization paths.

func TestInitGo(t *testing.T) {
	saved := snapshotDispatch()
	t.Cleanup(func() { restoreDispatch(saved) })

	initGo()

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20) // 1*4 + 2*3 + 3*2 + 4*1 = 20
	if got != want {
		t.Errorf("After initGo, dotProduct = %v, want %v", got, want)
	}
	if roundImpl == nil {
		t.Error("initGo: roundImpl is nil")
	}
}

func TestInitSSE2(t *testing.T) {
	saved := snapshotDispatch()
	t.Cleanup(func() { restoreDispatch(saved) })

	initSSE2()

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20)
	if got != want {
		t.Errorf("After initSSE2, dotProduct = %v, want %v", got, want)
	}
	if roundImpl == nil {
		t.Error("initSSE2: roundImpl is nil")
	}
}

func TestInitAVX512(t *testing.T) {
	if !cpu.X86.AVX512F || !cpu.X86.AVX512VL {
		t.Skip("AVX-512 not supported on this CPU")
	}

	saved := snapshotDispatch()
	t.Cleanup(func() { restoreDispatch(saved) })

	initAVX512()

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	got := dotProductImpl(a, b)
	want := float64(20)
	if got != want {
		t.Errorf("After initAVX512, dotProduct = %v, want %v", got, want)
	}

	if minSIMDElements != minAVX512Elements {
		t.Errorf("initAVX512 didn't set minSIMDElements correctly")
	}
	if !samePointer(roundImpl, roundAVX) {
		t.Error("initAVX512: roundImpl should route to roundAVX")
	}
}

func TestInitAVXNoFMA(t *testing.T) {
	// initAVXNoFMA assigns AVX kernels (e.g. addAVX) to the global impl pointers.
	// Skip on CPUs without AVX so calling addImpl after init can't SIGILL.
	if !cpu.X86.AVX {
		t.Skip("AVX not supported on this CPU")
	}

	saved := snapshotDispatch()
	t.Cleanup(func() { restoreDispatch(saved) })

	initAVXNoFMA()

	// FMA-free AVX kernels stay on AVX paths.
	for _, c := range []struct {
		name string
		got  interface{}
		want interface{}
	}{
		{"addImpl", addImpl, addAVX},
		{"subImpl", subImpl, subAVX},
		{"mulImpl", mulImpl, mulAVX},
		{"divImpl", divImpl, divAVX},
		{"scaleImpl", scaleImpl, scaleAVX},
		{"addScalarImpl", addScalarImpl, addScalarAVX},
		{"sumImpl", sumImpl, sumAVX},
		{"minImpl", minImpl, minAVX},
		{"maxImpl", maxImpl, maxAVX},
		{"absImpl", absImpl, absAVX},
		{"negImpl", negImpl, negAVX},
		{"sqrtImpl", sqrtImpl, sqrtAVX},
		{"reciprocalImpl", reciprocalImpl, reciprocalAVX},
		{"roundImpl", roundImpl, roundAVX},
		{"clampImpl", clampImpl, clampAVX},
		{"interleave2Impl", interleave2Impl, interleave2AVX},
		{"deinterleave2Impl", deinterleave2Impl, deinterleave2AVX},
	} {
		if !samePointer(c.got, c.want) {
			t.Errorf("initAVXNoFMA: %s not routed to AVX kernel", c.name)
		}
	}

	// FMA-dependent kernels fall back to SSE2/scalar variants.
	for _, c := range []struct {
		name string
		got  interface{}
		want interface{}
	}{
		{"dotProductImpl", dotProductImpl, dotProductSSE2},
		{"fmaImpl", fmaImpl, fmaSSE2},
		{"varianceImpl", varianceImpl, varianceSSE2},
		{"euclideanDistanceImpl", euclideanDistanceImpl, euclideanDistanceSSE2},
		{"addScaledImpl", addScaledImpl, addScaledSSE2},
	} {
		if !samePointer(c.got, c.want) {
			t.Errorf("initAVXNoFMA: %s should fall back to SSE2 (FMA-free)", c.name)
		}
	}

	if minSIMDElements != minAVXElements {
		t.Errorf("initAVXNoFMA: minSIMDElements = %d, want %d", minSIMDElements, minAVXElements)
	}

	// Sanity check the kernels still produce correct values.
	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}
	if got := dotProductImpl(a, b); got != 20 {
		t.Errorf("dotProductImpl(a,b) = %v, want 20", got)
	}
	dst := make([]float64, len(a))
	addImpl(dst, a, b)
	for i, want := range []float64{5, 5, 5, 5} {
		if dst[i] != want {
			t.Errorf("addImpl[%d] = %v, want %v", i, dst[i], want)
		}
	}
}
