package f32

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// bucketFunc is the shape shared by the public Float32ToUint8Bucket, the Go
// fallback adapter and the per-architecture direct-kernel adapters, so one suite
// drives every path.
type bucketFunc func(dst []uint8, src []float32, scale, offset float32, buckets int)

const bucketSentinel = 0xA5

// float32ToUint8BucketRef is an independent oracle for Float32ToUint8Bucket. It
// computes the two float32 roundings through float64 (a float32 product is exact
// in float64 and float64 holds the float32 sum with innocuous double rounding),
// so it reaches the result by a different route than the production Go reference
// and cannot be compiler-fused. Any non-finite input maps to bucket 0; a NaN
// affine maps to 0; the clamp is in the float domain; the conversion truncates.
func float32ToUint8BucketRef(dst []uint8, src []float32, scale, offset float32, buckets int) {
	hi := float32(buckets - 1)
	n := min(len(dst), len(src))
	for i := range n {
		x := float64(src[i])
		if math.IsNaN(x) || math.IsInf(x, 0) {
			dst[i] = 0
			continue
		}
		p := float32(x * float64(scale))           // product, rounded to float32
		v := float32(float64(p) + float64(offset)) // + offset, rounded to float32
		if v != v {
			dst[i] = 0
			continue
		}
		if v < 0 {
			v = 0
		}
		if v > hi {
			v = hi
		}
		dst[i] = uint8(int(v)) // truncate toward zero
	}
}

// bucketGoFunc adapts the Go fallback to bucketFunc.
func bucketGoFunc(dst []uint8, src []float32, scale, offset float32, buckets int) {
	float32ToUint8BucketGo(dst, src, scale, offset, float32(buckets-1))
}

type bucketSetting struct {
	scale, offset float32
	buckets       int
}

func (s bucketSetting) String() string {
	return fmt.Sprintf("scale=%g offset=%g buckets=%d", s.scale, s.offset, s.buckets)
}

var bucketSettings = []bucketSetting{
	{5, 65, 96},
	{1, 0, 256},
	{1, 0, 1},
	{-3, 100, 256},
	{0, 7.5, 10},
	{float32(math.NaN()), 3, 50},
	{float32(math.Inf(1)), 3, 50},
	{float32(math.Inf(-1)), 3, 50},
	{2, float32(math.NaN()), 50},
	{1e30, 0, 256},
}

func bucketEdgeValues() []float32 {
	inf := float32(math.Inf(1))
	return []float32{
		float32(math.NaN()), inf, -inf, 0, float32(math.Copysign(0, -1)),
		-0.5, -1e-30, 1e-30, 5e9, -5e9, 1e38, -1e38,
		math.MaxFloat32, -math.MaxFloat32,
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		1e-30, // with scale 1e-10 the product underflows to a denormal; see bucketCheckEdges
		0.9999999, 1, 1.0000001, 254.99998, 255, 255.00002, 256, 300,
	}
}

// bucketCheck runs fn and the oracle on identical sentinel-filled destinations
// and fails if any byte differs, which also checks that dst[n:] is untouched.
func bucketCheck(t *testing.T, fn bucketFunc, dstLen int, src []float32, s bucketSetting) {
	t.Helper()
	got := make([]uint8, dstLen)
	want := make([]uint8, dstLen)
	for i := range got {
		got[i], want[i] = bucketSentinel, bucketSentinel
	}
	fn(got, src, s.scale, s.offset, s.buckets)
	float32ToUint8BucketRef(want, src, s.scale, s.offset, s.buckets)
	for i := range got {
		if got[i] != want[i] {
			in := "past processed range"
			if i < len(src) {
				in = fmt.Sprintf("src=%g (bits %#08x)", src[i], math.Float32bits(src[i]))
			}
			t.Fatalf("len(dst)=%d len(src)=%d [%d] = %d, want %d (%s, %v)",
				dstLen, len(src), i, got[i], want[i], in, s)
		}
	}
}

func bucketCheckEdges(t *testing.T, fn bucketFunc) {
	t.Helper()
	edges := bucketEdgeValues()
	for _, s := range bucketSettings {
		for _, n := range []int{16, 33} {
			// Rotate so every edge value visits every lane position, going through
			// both the vector body and the overlapping final block.
			for rot := range edges {
				src := make([]float32, n)
				for i := range src {
					src[i] = edges[(i+rot)%len(edges)]
				}
				bucketCheck(t, fn, n, src, s)
			}
		}
	}
	// A product that underflows to a denormal.
	src := make([]float32, 20)
	for i := range src {
		src[i] = 1e-30
	}
	bucketCheck(t, fn, len(src), src, bucketSetting{1e-10, 0.5, 10})
}

func bucketBoundarySrc(scale, offset float32) []float32 {
	var out []float32
	add := func(v float32) {
		out = append(out, v, math.Nextafter32(v, float32(math.Inf(-1))), math.Nextafter32(v, float32(math.Inf(1))))
	}
	for k := -2; k <= 258; k++ {
		add(float32(k))
		// Values whose affine lands at or next to the integer k.
		add((float32(k) - offset) / scale)
	}
	return out
}

func bucketCheckBoundaries(t *testing.T, fn bucketFunc) {
	t.Helper()
	for _, s := range []bucketSetting{{1, 0, 256}, {1, 0, 96}, {5, 65, 96}, {5, 65, 256}} {
		src := bucketBoundarySrc(s.scale, s.offset)
		for _, shift := range []int{0, 1, 5} { // vary the overlap-tail alignment
			bucketCheck(t, fn, len(src)-shift, src[shift:], s)
		}
	}
}

func bucketCheckLengths(t *testing.T, fn bucketFunc) {
	t.Helper()
	rng := rand.New(rand.NewPCG(310, 256))
	lengths := make([]int, 0, 90)
	for n := 0; n <= 70; n++ {
		lengths = append(lengths, n)
	}
	lengths = append(lengths, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2049, 4103)
	for _, n := range lengths {
		src := make([]float32, n)
		for i := range src {
			if i%3 == 0 {
				src[i] = math.Float32frombits(rng.Uint32()) // any bit pattern
			} else {
				src[i] = float32(rng.Float64()*60 - 14) // values from -14 to 46
			}
		}
		for _, s := range bucketSettings[:5] {
			bucketCheck(t, fn, n, src, s)
		}
	}
}

func bucketCheckOrder(t *testing.T, fn bucketFunc) {
	t.Helper()
	for _, n := range []int{16, 17, 31, 32, 33, 300} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i % 256)
		}
		dst := make([]uint8, n)
		fn(dst, src, 1, 0, 256)
		for i := range dst {
			if dst[i] != uint8(i%256) {
				t.Fatalf("n=%d [%d] = %d, want %d", n, i, dst[i], i%256)
			}
		}
	}
}

func bucketCheckUnaligned(t *testing.T, fn bucketFunc) {
	t.Helper()
	rng := rand.New(rand.NewPCG(7, 310))
	for off := 1; off <= 3; off++ {
		for doff := 1; doff <= 15; doff++ {
			for n := 16; n <= 40; n++ {
				backing := make([]float32, off+n)
				for i := range backing {
					backing[i] = float32(rng.Float64()*60 - 14)
				}
				src := backing[off:]
				got := make([]uint8, doff+n)
				want := make([]uint8, doff+n)
				for i := range got {
					got[i], want[i] = bucketSentinel, bucketSentinel
				}
				fn(got[doff:], src, 5, 65, 96)
				float32ToUint8BucketRef(want[doff:], src, 5, 65, 96)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("off=%d doff=%d n=%d [%d] = %d, want %d", off, doff, n, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func bucketCheckLengthMismatch(t *testing.T, fn bucketFunc) {
	t.Helper()
	s := bucketSetting{5, 65, 96}
	rng := rand.New(rand.NewPCG(11, 22))
	mk := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.Float64()*60 - 14)
		}
		return v
	}
	// dst shorter than src: only len(dst) elements are written.
	bucketCheck(t, fn, 20, mk(40), s)
	// src shorter than dst: dst[len(src):] keeps its sentinel.
	bucketCheck(t, fn, 40, mk(20), s)
	bucketCheck(t, fn, 40, mk(17), s)
	// dst resliced from a larger backing array: bytes in its capacity but past
	// len are not written.
	backing := make([]uint8, 64)
	for i := range backing {
		backing[i] = bucketSentinel
	}
	src := mk(30)
	fn(backing[:25], src, s.scale, s.offset, s.buckets)
	for i := 25; i < len(backing); i++ {
		if backing[i] != bucketSentinel {
			t.Fatalf("wrote into capacity past len(dst): backing[%d] = %d", i, backing[i])
		}
	}
	want := make([]uint8, 25)
	float32ToUint8BucketRef(want, src, s.scale, s.offset, s.buckets)
	for i := range want {
		if backing[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d", i, backing[i], want[i])
		}
	}
}

// bucketCheckExplicit pins the documented edge values with literal expectations
// rather than oracle parity, on a buffer long enough for the vector body.
func bucketCheckExplicit(t *testing.T, fn bucketFunc) {
	t.Helper()
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	cases := []struct {
		name string
		x    float32
		s    bucketSetting
		want uint8
	}{
		{"+Inf scale 5", inf, bucketSetting{5, 65, 96}, 0},
		{"+Inf scale -3", inf, bucketSetting{-3, 100, 256}, 0},
		{"NaN scale 5", nan, bucketSetting{5, 65, 96}, 0},
		{"-Inf scale 5", -inf, bucketSetting{5, 65, 96}, 0},
		{"-Inf scale -3", -inf, bucketSetting{-3, 100, 256}, 0},
		{"-Inf scale 0", -inf, bucketSetting{0, 100, 256}, 0},
		{"finite overflow to +Inf clamps high", 1e38, bucketSetting{5, 65, 96}, 95},
		{"finite overflow via negative scale clamps high", -1e38, bucketSetting{-5, 65, 96}, 95},
		{"huge negative clamps low", -1e38, bucketSetting{5, 65, 96}, 0},
		{"(-1,0) affine truncates to 0", -0.5, bucketSetting{1, 0, 256}, 0},
		{"value 255 survives", 255, bucketSetting{1, 0, 256}, 255},
		{"value 128 survives", 128, bucketSetting{1, 0, 256}, 128},
		{"clamps to buckets-1", 1000, bucketSetting{1, 0, 100}, 99},
		{"single bucket", 1000, bucketSetting{1, 0, 1}, 0},
	}
	for _, c := range cases {
		for _, n := range []int{20, 40} {
			src := make([]float32, n)
			for i := range src {
				src[i] = c.x
			}
			dst := make([]uint8, n)
			fn(dst, src, c.s.scale, c.s.offset, c.s.buckets)
			for i, g := range dst {
				if g != c.want {
					t.Fatalf("%s: n=%d [%d] = %d, want %d (%v)", c.name, n, i, g, c.want, c.s)
				}
			}
		}
	}
}

// bucketCheckNoFMA is an FMA detector. With src=scale=4097 and
// offset=-(2^24+2^13) the true product 16785409 rounds to float32 16785408, so
// two roundings give exactly 0 and a fused multiply-add gives 1.
func bucketCheckNoFMA(t *testing.T, fn bucketFunc) {
	t.Helper()
	for _, n := range []int{20, 40} { // vector body plus overlapping block
		src := make([]float32, n)
		for i := range src {
			src[i] = 4097
		}
		dst := make([]uint8, n)
		fn(dst, src, 4097, -16785408, 256)
		for i, g := range dst {
			if g != 0 {
				t.Fatalf("n=%d [%d] = %d, want 0 (non-zero means the multiply and add fused)", n, i, g)
			}
		}
	}
}

// runBucketSuite drives every behavior check against fn.
func runBucketSuite(t *testing.T, fn bucketFunc) {
	t.Helper()
	t.Run("Edges", func(t *testing.T) { bucketCheckEdges(t, fn) })
	t.Run("Explicit", func(t *testing.T) { bucketCheckExplicit(t, fn) })
	t.Run("Boundaries", func(t *testing.T) { bucketCheckBoundaries(t, fn) })
	t.Run("Order", func(t *testing.T) { bucketCheckOrder(t, fn) })
	t.Run("NoFMA", func(t *testing.T) { bucketCheckNoFMA(t, fn) })
	t.Run("Lengths", func(t *testing.T) { bucketCheckLengths(t, fn) })
	t.Run("Unaligned", func(t *testing.T) { bucketCheckUnaligned(t, fn) })
	t.Run("LengthMismatch", func(t *testing.T) { bucketCheckLengthMismatch(t, fn) })
}

func TestFloat32ToUint8Bucket(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		runBucketSuite(t, Float32ToUint8Bucket)
	})
}

// TestFloat32ToUint8BucketGo runs the whole suite against the Go fallback
// directly, so it is covered when dispatch lands on a SIMD kernel and so a
// compiler-fused multiply-add in the fallback (arm64) is caught.
func TestFloat32ToUint8BucketGo(t *testing.T) {
	runBucketSuite(t, bucketGoFunc)
}

// TestFloat32ToUint8Bucket_Float32NotFloat64 pins the documented example: the
// arithmetic is float32, so near a bucket edge it differs from a float64
// evaluation of src*scale+offset.
func TestFloat32ToUint8Bucket_Float32NotFloat64(t *testing.T) {
	src := float32(-1e-7) // a variable: a constant would fold at arbitrary precision
	if got := int(float64(src)*5 + 65); got != 64 {
		t.Fatalf("float64 evaluation = %d, want 64 (the premise of the documented example)", got)
	}
	buf := make([]float32, 20)
	for i := range buf {
		buf[i] = src
	}
	forTiers(t, func(t *testing.T) {
		t.Helper()
		dst := make([]uint8, len(buf))
		Float32ToUint8Bucket(dst, buf, 5, 65, 96)
		for i, g := range dst {
			if g != 65 {
				t.Fatalf("[%d] = %d, want 65", i, g)
			}
		}
	})
}

func TestFloat32ToUint8Bucket_PanicsOnBadBuckets(t *testing.T) {
	mustPanic := func(t *testing.T, dst []uint8, src []float32, buckets int) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("buckets=%d len(dst)=%d: no panic", buckets, len(dst))
			}
			msg, ok := r.(string)
			if !ok || !strings.HasPrefix(msg, "simd: ") {
				t.Fatalf("buckets=%d: panic value %v, want a \"simd: \" message", buckets, r)
			}
		}()
		Float32ToUint8Bucket(dst, src, 1, 0, buckets)
	}
	for _, b := range []int{0, -1, 257, math.MaxInt, math.MinInt} {
		mustPanic(t, make([]uint8, 20), make([]float32, 20), b)
		mustPanic(t, nil, nil, b) // checked even when there is nothing to process
		mustPanic(t, make([]uint8, 4), make([]float32, 4), b)
	}
	for _, b := range []int{1, 256} {
		Float32ToUint8Bucket(make([]uint8, 20), make([]float32, 20), 1, 0, b)
		Float32ToUint8Bucket(nil, nil, 1, 0, b)
	}
}

func TestFloat32ToUint8Bucket_AllocFree(t *testing.T) {
	src := make([]float32, 1025)
	for i := range src {
		src[i] = float32(i%50) - 5
	}
	dst := make([]uint8, len(src))
	allocs := testing.AllocsPerRun(100, func() {
		Float32ToUint8Bucket(dst, src, 5, 65, 96)
	})
	if allocs != 0 {
		t.Fatalf("AllocsPerRun = %v, want 0", allocs)
	}
}

func FuzzFloat32ToUint8Bucket(f *testing.F) {
	f.Add(math.Float32bits(5), math.Float32bits(65), uint16(95), uint16(300), uint32(1))
	f.Add(math.Float32bits(1), math.Float32bits(0), uint16(255), uint16(33), uint32(2))
	f.Add(math.Float32bits(float32(math.NaN())), math.Float32bits(0), uint16(0), uint16(17), uint32(3))
	f.Fuzz(func(t *testing.T, scaleBits, offsetBits uint32, bucketsRaw, nRaw uint16, seed uint32) {
		buckets := 1 + int(bucketsRaw)%256
		n := int(nRaw) % 300
		s := bucketSetting{math.Float32frombits(scaleBits), math.Float32frombits(offsetBits), buckets}
		src := make([]float32, n)
		for i := range src {
			bits := seed*uint32(i+1) ^ offsetBits
			if i%4 == 0 {
				src[i] = float32(int32(bits>>8)%4000) / 100 // magnitudes below 40
			} else {
				src[i] = math.Float32frombits(bits)
			}
		}
		// NaN scale and offset are in contract: both are compared to the oracle.
		bucketCheck(t, Float32ToUint8Bucket, n, src, s)
	})
}

// bucketBenchScalar is the consumer-style loop the kernel replaces: a float64
// affine, a NaN/+Inf guard, a clamp and an integer conversion.
func bucketBenchScalar(dst []uint8, src []float32, nb int) {
	for i, lp := range src {
		b := 0
		if lp == lp && !math.IsInf(float64(lp), 1) {
			b = min(max(int(float64(lp)*5+65), 0), nb-1)
		}
		dst[i] = uint8(b)
	}
}

func BenchmarkFloat32ToUint8Bucket(b *testing.B) {
	for _, size := range []int{257, 513, 1025, 2049, 4096, 16384} {
		src := make([]float32, size)
		for i := range src {
			src[i] = float32(i%520)*0.1 - 12 // about [-12, 40]
		}
		src[size/3] = float32(math.NaN())
		src[size/2] = float32(math.Inf(1))
		dst := make([]uint8, size)
		benchScalePair(b, size, 5,
			func() { Float32ToUint8Bucket(dst, src, 5, 65, 96) },
			func() { float32ToUint8BucketGo(dst, src, 5, 65, 95) })
		b.Run(fmt.Sprintf("Scalar_%d", size), func(b *testing.B) {
			for b.Loop() {
				bucketBenchScalar(dst, src, 96)
			}
			b.SetBytes(int64(size * 5))
		})
	}
}
