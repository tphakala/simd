package f64

import (
	"math"
	"math/rand"
	"testing"
)

// quantPow34Kernel is one implementation of the banded quantizer under test.
type quantPow34Kernel struct {
	name string
	fn   func(dst []int32, src []float64, widths []int, scales []float64, rounding, limit, z float64)
}

// quantPow34Spec is the literal per-line formula with no zero skip.
func quantPow34Spec(x, s, rounding, limit float64) int32 {
	t := float64(math.Abs(x) * s)
	v := math.Sqrt(float64(t * math.Sqrt(t)))
	n := v + rounding
	var m int32
	if n <= limit {
		m = int32(n)
	} else {
		m = int32(limit)
	}
	if x < 0 {
		m = -m
	}
	return m
}

// quantPow34API runs the public function through the kernel signature; it
// ignores z because the wrapper derives its own.
var quantPow34API = quantPow34Kernel{"api", func(dst []int32, src []float64, widths []int, scales []float64, rounding, limit, _ float64) {
	QuantizePow34Bands(dst, src, widths, scales, rounding, limit)
}}

func quantPow34Total(widths []int) int {
	total := 0
	for _, w := range widths {
		total += w
	}
	return total
}

// runQuantPow34Kernel drives a kernel exactly as QuantizePow34Bands does: the
// slices are cut to the band total and z comes from quantPow34Threshold.
func runQuantPow34Kernel(k quantPow34Kernel, dst []int32, src []float64, widths []int, scales []float64, rounding, limit float64) {
	total := quantPow34Total(widths)
	z := quantPow34Threshold(scales, rounding)
	k.fn(dst[:total], src[:total], widths, scales, rounding, limit, z)
}

func quantPow34Want(src []float64, widths []int, scales []float64, rounding, limit float64) []int32 {
	want := make([]int32, 0, len(src))
	i := 0
	for b, w := range widths {
		for range w {
			want = append(want, quantPow34Spec(src[i], scales[b], rounding, limit))
			i++
		}
	}
	return want
}

func quantPow34Edges(limit float64) []float64 {
	negNaN := math.Float64frombits(0xfff8000000000001)
	posNaN := math.Float64frombits(0x7ff8000000000001)
	edges := make([]float64, 0, 48)
	edges = append(edges,
		posNaN, negNaN, math.NaN(),
		math.Inf(1), math.Inf(-1),
		0, math.Copysign(0, -1),
		5e-324, -5e-324, 1e-300, -1e-300,
		1e300, -1e300, 1e154, 1e205,
		0.3, -0.3, 0.49, 0.5, -0.5, 1, -1, 2, 7.5, -7.5, 1e5, -1e5, 1e7,
	)
	// Values straddling the clamp boundary of the power law.
	for _, v := range quantPow34UlpNeighbours(math.Pow(limit-0.4054, 4.0/3)) {
		edges = append(edges, v, -v)
	}
	return edges
}

func TestQuantizePow34Bands_Edges(t *testing.T) {
	scales := []float64{1, 0.5, 2, 0x1p-11, 0, math.Inf(1), math.Copysign(0, -1), -1, math.NaN(), 1e300}
	type rl struct{ r, lim float64 }
	rls := []rl{
		{0.4054, 8206}, {0, 8206}, {0.5, 8206}, {-0.5, 8206},
		{math.Nextafter(-1, 0), 8206}, {0.999, 8206}, {1, 8206}, {1.5, 8206},
		{math.Inf(1), 8206}, {0.4054, 0}, {0.4054, 0.5}, {0.4054, math.MaxInt32},
		{0.4054, 8206.7}, {-0.5, 0},
	}
	kernels := append([]quantPow34Kernel{quantPow34API}, quantPow34Kernels()...)
	for _, c := range rls {
		src := quantPow34Edges(c.lim)
		widths := []int{len(src)}
		for _, s := range scales {
			want := quantPow34Want(src, widths, []float64{s}, c.r, c.lim)
			for _, k := range kernels {
				dst := make([]int32, len(src))
				runQuantPow34Kernel(k, dst, src, widths, []float64{s}, c.r, c.lim)
				for i := range dst {
					if dst[i] != want[i] {
						t.Fatalf("%s: x=%v scale=%v rounding=%v limit=%v: got %d, want %d", k.name, src[i], s, c.r, c.lim, dst[i], want[i])
					}
				}
			}
		}
	}
}

func TestQuantizePow34Bands_Pinned(t *testing.T) {
	negNaN := math.Float64frombits(0xfff8000000000001)
	src := []float64{math.NaN(), negNaN, math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1), 0.3, 1, -1, 1e5, -1e5, 1e300}
	want := []int32{8206, 8206, 8206, -8206, 0, 0, 0, 1, -1, 5623, -5623, 8206}
	dst := make([]int32, len(src))
	QuantizePow34Bands(dst, src, []int{len(src)}, []float64{1}, 0.4054, 8206)
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("x=%v: got %d, want %d", src[i], dst[i], want[i])
		}
	}
	// Per-band scales: the same input scaled by the band's scale.
	dst = make([]int32, 4)
	QuantizePow34Bands(dst, []float64{1, 1, 16, 16}, []int{2, 2}, []float64{1, 0}, 0.4054, 8206)
	if want := []int32{1, 1, 0, 0}; dst[0] != want[0] || dst[1] != want[1] || dst[2] != want[2] || dst[3] != want[3] {
		t.Errorf("banded scales: got %v, want %v", dst, want)
	}
	// A negative scale makes sqrt see a negative t, which yields the limit.
	dst = make([]int32, 2)
	QuantizePow34Bands(dst, []float64{0.3, -0.3}, []int{2}, []float64{-1}, 0.4054, 8206)
	if dst[0] != 8206 || dst[1] != -8206 {
		t.Errorf("negative scale: got %v, want [8206 -8206]", dst)
	}
	// +-0 input with a finite negative scale gives int32(rounding), not the limit,
	// and is never negated.
	dst = make([]int32, 2)
	QuantizePow34Bands(dst, []float64{0, math.Copysign(0, -1)}, []int{2}, []float64{-1}, 1.5, 8206)
	if dst[0] != 1 || dst[1] != 1 {
		t.Errorf("zero input, negative scale: got %v, want [1 1]", dst)
	}
	// limit 0 with a zero-skippable input still gives 0, and limit 0.5 clamps to 0.
	dst = make([]int32, 2)
	QuantizePow34Bands(dst, []float64{0.3, 100}, []int{2}, []float64{1}, 0.4054, 0.5)
	if dst[0] != 0 || dst[1] != 0 {
		t.Errorf("limit 0.5: got %v, want [0 0]", dst)
	}
}

func quantPow34Layouts() [][]int {
	long := [][]int{
		{4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76, 158},
		{4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54, 192},
		{4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102, 26},
	}
	short := [][]int{
		{4, 4, 4, 4, 6, 8, 10, 12, 14, 18, 22, 30, 56},
		{4, 4, 4, 4, 6, 6, 10, 12, 14, 16, 20, 26, 66},
		{4, 4, 4, 4, 6, 8, 12, 16, 20, 26, 34, 42, 12},
	}
	layouts := append([][]int{}, long...)
	for _, s := range short {
		var w []int
		for range 3 {
			w = append(w, s...)
		}
		layouts = append(layouts, w)
	}
	return append(layouts, []int{1, 3, 0, 5, 7, 2, 0, 9}, []int{0}, []int{3}, []int{1}, []int{2, 2, 2}, []int{})
}

// quantPow34RandomBands draws per-band scales (every third band negated when
// negScale is set) and lines in one of three regimes by trial: wide magnitudes,
// values around the zero boundary of the band, and mostly skippable lines with
// sparse non-zero ones. Edge values are planted at random positions.
func quantPow34RandomBands(rng *rand.Rand, widths []int, trial int, negScale bool) (src, scales []float64) {
	scales = make([]float64, len(widths))
	for b := range scales {
		scales[b] = math.Ldexp(1, rng.Intn(51)-30)
		if negScale && b%3 == 1 {
			scales[b] = -scales[b]
		}
	}
	src = make([]float64, 0, quantPow34Total(widths))
	for b, w := range widths {
		for range w {
			var x float64
			switch trial % 3 {
			case 0:
				x = math.Pow(10, rng.Float64()*11-4)
			case 1:
				x = (0.3 + 0.4*rng.Float64()) / math.Abs(scales[b])
			default:
				x = 0.1 / math.Abs(scales[b])
				if rng.Intn(5) == 0 {
					x *= 40
				}
			}
			if rng.Intn(2) == 0 {
				x = -x
			}
			src = append(src, x)
		}
	}
	edges := quantPow34Edges(8206)
	for range len(src) / 8 {
		src[rng.Intn(len(src))] = edges[rng.Intn(len(edges))]
	}
	return src, scales
}

func TestQuantizePow34Bands_KernelParity(t *testing.T) {
	const sentinel = int32(-12345)
	rng := rand.New(rand.NewSource(34))
	roundings := []float64{0.4054, 0, 0.5}
	for _, k := range quantPow34Kernels() {
		t.Run(k.name, func(t *testing.T) {
			for li, widths := range quantPow34Layouts() {
				total := quantPow34Total(widths)
				for trial := range 120 {
					src, scales := quantPow34RandomBands(rng, widths, trial, trial%10 == 9)
					lim := 8206.0
					if trial%7 == 0 {
						lim = math.MaxInt32
					}
					for _, r := range roundings {
						dst := make([]int32, total+5)
						for j := range dst {
							dst[j] = sentinel
						}
						runQuantPow34Kernel(k, dst, src, widths, scales, r, lim)
						want := quantPow34Want(src, widths, scales, r, lim)
						for j := range total {
							if dst[j] != want[j] {
								t.Fatalf("layout %d trial %d line %d: x=%v got %d, want %d (rounding %v limit %v)", li, trial, j, src[j], dst[j], want[j], r, lim)
							}
						}
						for j := total; j < len(dst); j++ {
							if dst[j] != sentinel {
								t.Fatalf("layout %d trial %d: wrote past the band total at %d", li, trial, j)
							}
						}
					}
				}
			}
		})
	}
}

// quantPow34FirstNonZero returns the smallest t >= 0 whose chain output at
// scale 1 is non-zero, found by bisection over the float64 bit patterns (the
// chain is monotone non-decreasing in t).
func quantPow34FirstNonZero(rounding, limit float64) float64 {
	lo, hi := uint64(0), math.Float64bits(4)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if quantPow34Spec(math.Float64frombits(mid), 1, rounding, limit) != 0 {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return math.Float64frombits(lo)
}

// quantPow34UlpNeighbours returns base and its 8 float64 neighbours on each side.
func quantPow34UlpNeighbours(base float64) []float64 {
	ts := make([]float64, 0, 17)
	ts = append(ts, base)
	up, down := base, base
	for range 8 {
		up = math.Nextafter(up, math.Inf(1))
		down = math.Nextafter(down, 0)
		ts = append(ts, up, down)
	}
	return ts
}

// checkQuantPow34Boundary feeds every kernel lines whose scaled magnitude is ts,
// laid out as uniform groups (all lanes alike, taking the pair skip) and as
// mixed groups, at the given scale.
func checkQuantPow34Boundary(t *testing.T, ts []float64, scale, rounding float64) {
	t.Helper()
	uniform := make([]float64, 0, 8*len(ts))
	mixed := make([]float64, 0, 4*len(ts))
	for _, tv := range ts {
		x := tv / scale
		for range 8 {
			uniform = append(uniform, x)
		}
		mixed = append(mixed, x, 0, -x, 1e3/scale)
	}
	for _, k := range quantPow34Kernels() {
		for _, src := range [][]float64{uniform, mixed} {
			widths := []int{len(src) / 2, len(src) - len(src)/2}
			scales := []float64{scale, scale}
			dst := make([]int32, len(src))
			runQuantPow34Kernel(k, dst, src, widths, scales, rounding, 8206)
			want := quantPow34Want(src, widths, scales, rounding, 8206)
			for i := range dst {
				if dst[i] != want[i] {
					t.Fatalf("%s rounding %v scale %v: x=%v got %d, want %d", k.name, rounding, scale, src[i], dst[i], want[i])
				}
			}
		}
	}
}

func TestQuantizePow34Bands_ZeroSkipBoundary(t *testing.T) {
	roundings := []float64{0.4054, 0.5, 0, -0.5, math.Nextafter(-1, 0), 0.999, 1 - 0x1p-30, 1 - 0x1p-45}
	for _, r := range roundings {
		t0 := quantPow34FirstNonZero(r, 8206)
		z := pow34ZeroBound(r)
		disabled := math.IsInf(z, -1)
		switch {
		case r == 0.4054 && z < 0.49:
			t.Errorf("rounding %v: z = %v, want >= 0.49", r, z)
		case disabled && r != 1-0x1p-45:
			// The bound is derivable for every other rounding here: -Inf
			// would silently disable the skip.
			t.Errorf("rounding %v: skip unexpectedly disabled", r)
		case !disabled && r == 1-0x1p-45:
			t.Errorf("rounding %v: z = %v, want -Inf", r, z)
		case !disabled && z/t0 < 1-0x1p-16:
			t.Errorf("rounding %v: z/t0 = %v, want >= 1-2^-16", r, z/t0)
		}
		ts := quantPow34UlpNeighbours(t0)
		if !disabled && z > 0 {
			ts = append(ts, quantPow34UlpNeighbours(z)...)
		}
		for _, scale := range []float64{1, 0x1p-11} {
			checkQuantPow34Boundary(t, ts, scale, r)
		}
	}
}

func TestQuantizePow34Bands_Panics(t *testing.T) {
	src := make([]float64, 8)
	dst := make([]int32, 8)
	nan := math.NaN()
	rows := []struct {
		name     string
		dst      []int32
		widths   []int
		scales   []float64
		rounding float64
		limit    float64
	}{
		{"fewer scales than widths", dst, []int{4, 4}, []float64{1}, 0.4, 8206},
		{"more scales than widths", dst, []int{4}, []float64{1, 1}, 0.4, 8206},
		{"negative width", dst, []int{-1, 4}, []float64{1, 1}, 0.4, 8206},
		{"widths exceed src", dst, []int{4, 5}, []float64{1, 1}, 0.4, 8206},
		{"widths exceed dst", dst[:6], []int{4, 4}, []float64{1, 1}, 0.4, 8206},
		{"limit negative", dst, []int{8}, []float64{1}, 0.4, -1},
		{"limit NaN", dst, []int{8}, []float64{1}, 0.4, nan},
		{"limit 2^31", dst, []int{8}, []float64{1}, 0.4, 0x1p31},
		{"rounding -1", dst, []int{8}, []float64{1}, -1, 8206},
		{"rounding NaN", dst, []int{8}, []float64{1}, nan, 8206},
		{"empty widths with bad limit", dst, nil, nil, 0.4, -1},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("expected panic")
				}
			}()
			QuantizePow34Bands(r.dst, src, r.widths, r.scales, r.rounding, r.limit)
		})
	}
}

func TestQuantizePow34Bands_NoLines(t *testing.T) {
	dst := []int32{9, 9}
	QuantizePow34Bands(dst, []float64{1, 1}, nil, nil, 0.4054, 8206)
	QuantizePow34Bands(dst, []float64{1, 1}, []int{0, 0}, []float64{1, 1}, 0.4054, 8206)
	if dst[0] != 9 || dst[1] != 9 {
		t.Errorf("zero lines wrote to dst: %v", dst)
	}
}

func TestQuantPow34Threshold(t *testing.T) {
	want := pow34ZeroBound(0.4054)
	if got := quantPow34Threshold([]float64{1, 0, math.Inf(1)}, 0.4054); got != want {
		t.Errorf("non-negative scales: got %v, want %v", got, want)
	}
	for _, bad := range []float64{-1, math.NaN()} {
		if got := quantPow34Threshold([]float64{1, bad}, 0.4054); !math.IsInf(got, -1) {
			t.Errorf("scale %v: got %v, want -Inf", bad, got)
		}
	}
}

func TestQuantizePow34Bands_LeavesTailUntouched(t *testing.T) {
	src := []float64{1, 1, 1, 1, 1}
	dst := []int32{9, 9, 9, 9, 9}
	QuantizePow34Bands(dst, src, []int{2, 1}, []float64{1, 1}, 0.4054, 8206)
	if want := []int32{1, 1, 1, 9, 9}; dst[0] != want[0] || dst[1] != want[1] || dst[2] != want[2] || dst[3] != want[3] || dst[4] != want[4] {
		t.Errorf("got %v, want %v", dst, want)
	}
}

func TestQuantizePow34Bands_AllocFree(t *testing.T) {
	var xr [576]float64
	var ix [576]int32
	var widths [39]int
	var scales [39]float64
	for i := range xr {
		xr[i] = float64(i%17) - 8
	}
	for i := range widths {
		widths[i] = 576 / 39
		scales[i] = 1
	}
	widths[38] += 576 - 39*(576/39)
	n := testing.AllocsPerRun(100, func() {
		QuantizePow34Bands(ix[:], xr[:], widths[:], scales[:], 0.4054, 8206)
	})
	if n != 0 {
		t.Errorf("allocations per call = %v, want 0", n)
	}
}

func BenchmarkQuantizePow34Bands(b *testing.B) {
	layouts := []struct {
		name   string
		widths []int
	}{
		{"long", quantPow34Layouts()[0]},
		{"short", quantPow34Layouts()[3]},
	}
	for _, ly := range layouts {
		total := quantPow34Total(ly.widths)
		base := make([]float64, total)
		seed := uint32(1)
		for i := range base {
			seed = seed*1664525 + 1013904223
			base[i] = (float64(seed>>8)/float64(1<<24)*2 - 1) * 2000
		}
		datasets := []struct {
			name  string
			scale float64
			src   []float64
		}{
			{"dense", 1, base},
			{"sparse", 0x1p-11, base},
			{"tail", 0x1p-11, func() []float64 {
				s := append([]float64(nil), base...)
				for i := 200; i < total; i++ {
					s[i] *= 0.2
				}
				return s
			}()},
		}
		for _, ds := range datasets {
			scales := make([]float64, len(ly.widths))
			for i := range scales {
				scales[i] = ds.scale
			}
			dst := make([]int32, total)
			name := ly.name + "/" + ds.name
			b.Run(name+"/api", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					QuantizePow34Bands(dst, ds.src, ly.widths, scales, 0.4054, 8206)
				}
			})
			z := quantPow34Threshold(scales, 0.4054)
			for _, k := range quantPow34Kernels() {
				b.Run(name+"/"+k.name, func(b *testing.B) {
					b.ReportAllocs()
					for range b.N {
						k.fn(dst, ds.src, ly.widths, scales, 0.4054, 8206, z)
					}
				})
			}
			b.Run(name+"/spec-loop", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					quantPow34SpecLoop(dst, ds.src, ly.widths, scales)
				}
			})
		}
	}
}

// quantPow34SpecLoop is the per-line scalar loop with the fixed 0.49 zero skip
// that a caller would write without the kernel.
func quantPow34SpecLoop(dst []int32, src []float64, widths []int, scales []float64) {
	i := 0
	for b, w := range widths {
		s := scales[b]
		for end := i + w; i < end; i++ {
			t := math.Abs(src[i]) * s
			if t < 0.49 {
				dst[i] = 0
				continue
			}
			v := math.Sqrt(t * math.Sqrt(t))
			var m int32
			if nint := v + 0.4054; nint <= 8206 {
				m = int32(nint)
			} else {
				m = 8206
			}
			if src[i] < 0 {
				m = -m
			}
			dst[i] = m
		}
	}
}
