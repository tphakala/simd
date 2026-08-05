// Package aliastest provides generic, bit-exact aliasing sweep helpers shared by
// the per-package aliasing_test.go suites (issue #221).
//
// Each helper computes a reference result into a destination that does not
// overlap any input, then re-runs the same operation with the destination
// physically overlaid on one of the inputs, and asserts the two results are
// bit-identical. Both runs execute on whatever kernel the caller has bound: the
// per-package suites force each amd64 function-pointer dispatch tier in turn and
// flip the arm64 hasNEON (or amd64 hasAVX2) gate, so every tier reachable through
// those mechanisms is measured. An op dispatched by an inline CPU-feature branch
// that the harness cannot rebind (a few f32/f64 kernels) is still swept across
// sizes, which runs both its Go and its native-SIMD kernel, and its overlay
// safety is verified by source audit; where such an op has a distinct SSE kernel
// the sweep cannot reach on an AVX host, the per-package suite exercises that
// kernel directly. A bit-exact match is the correct bar because both runs use the
// same kernel at the same length; a mismatch means the in-place overlay clobbered
// an input lane the kernel had not finished reading, which is exactly the
// corruption an "exact overlay supported" contract forbids.
//
// The helpers deliberately assert nothing about the corruption pattern of a
// non-overlapping op: that pattern varies with kernel width and length and is
// undefined behaviour, so only the supported overlays are exercised here.
package aliastest

import "testing"

// Sizes sweeps lengths that cross every dispatch boundary: scalar-only tails,
// whole SIMD blocks, block-plus-remainder, and the AVX-512 widths, plus a few
// large sizes. It intentionally includes 0 (the empty no-op) and lengths one
// above each vector width.
var Sizes = []int{
	0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
	63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1024, 1031,
}

// Case pairs an operation name with the closures that run its aliasing check and
// its allocation check.
type Case struct {
	// Name is the sub-test name (the operation under test).
	Name string
	// Check runs the operation's want-vs-overlay comparison at length n.
	Check func(t *testing.T, n int)
	// Alloc asserts the in-place overlay call allocates nothing.
	Alloc func(t *testing.T)
}

// Sweep runs every case's overlay comparison over every length in Sizes under the
// currently bound kernel tier.
func Sweep(t *testing.T, cases []Case) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			for _, n := range Sizes {
				c.Check(t, n)
			}
		})
	}
}

// SweepAlloc runs every case's allocation check under the currently bound kernel
// tier, enforcing the package's zero-allocation contract on the in-place overlay
// path (a kernel that allocates only when dst aliases an input would pass the
// value comparison but fail here).
func SweepAlloc(t *testing.T, cases []Case) {
	t.Helper()
	for _, c := range cases {
		if c.Alloc != nil {
			t.Run(c.Name, c.Alloc)
		}
	}
}

// zeroAllocRuns is the sample count for the AllocsPerRun measurements. A small
// count is enough: an allocating call allocates on every invocation.
const zeroAllocRuns = 4

// allocSize is the fixed length used for allocation checks; it clears every
// vector width so the measured call runs a real SIMD body, not only a tail.
const allocSize = 64

// ZeroAlloc asserts run allocates nothing, labelling a failure with name.
func ZeroAlloc(t *testing.T, name string, run func()) {
	t.Helper()
	if a := testing.AllocsPerRun(zeroAllocRuns, run); a != 0 {
		t.Errorf("%s: %.0f allocs/op on the overlay path, want 0", name, a)
	}
}

func buildOff[T any](n int, gen func(i int) T, off int) []T {
	s := make([]T, n)
	for i := range s {
		s[i] = gen(i + off)
	}
	return s
}

func clone[T any](s []T) []T {
	c := make([]T, len(s))
	copy(c, s)
	return c
}

// Report fails t with the first bit-level mismatch between want and got at length
// n, tagging the failure with mode. eq is the element comparator. It is exported
// so per-package suites can reuse it for op shapes the generic helpers do not
// model (the split-format complex products and the AXPY accumulator).
func Report[T any](t *testing.T, n int, mode string, eq func(x, y T) bool, want, got []T) {
	t.Helper()
	if len(want) != len(got) {
		t.Errorf("n=%d overlay %s: length mismatch: want %d, got %d", n, mode, len(want), len(got))
		return
	}
	for i := range want {
		if !eq(want[i], got[i]) {
			t.Errorf("n=%d overlay %s: index %d differs: want %v got %v",
				n, mode, i, want[i], got[i])
			return
		}
	}
}

// Unary checks the single overlay dst==a for a one-input operation op(dst, a).
func Unary[T any](t *testing.T, n int, eq func(x, y T) bool, gen func(i int) T, op func(dst, a []T)) {
	t.Helper()
	a := buildOff(n, gen, 0)

	want := make([]T, n)
	op(want, a)

	got := clone(a)
	op(got, got)
	Report(t, n, "dst=a", eq, want, got)
}

// Binary checks the overlays dst==a, dst==b, and dst==a==b for a two-input
// operation op(dst, a, b). The two operands are drawn from the same generator at
// distinct offsets so a and b differ elementwise.
func Binary[T any](t *testing.T, n int, eq func(x, y T) bool, gen func(i int) T, op func(dst, a, b []T)) {
	t.Helper()
	a := buildOff(n, gen, 0)
	b := buildOff(n, gen, offB)

	want := make([]T, n)
	op(want, a, b)

	got := clone(a)
	op(got, got, b)
	Report(t, n, "dst=a", eq, want, got)

	got = clone(b)
	op(got, a, got)
	Report(t, n, "dst=b", eq, want, got)

	// dst == a == b: destination and both operands are the same slice.
	wantSame := make([]T, n)
	op(wantSame, a, a)
	got = clone(a)
	op(got, got, got)
	Report(t, n, "dst=a=b", eq, wantSame, got)
}

// Ternary checks the overlays dst==a, dst==b, and dst==c for a three-input
// operation op(dst, a, b, c) such as a fused multiply-add. The three operands are
// drawn from the same generator at distinct offsets so they differ elementwise.
func Ternary[T any](t *testing.T, n int, eq func(x, y T) bool, gen func(i int) T, op func(dst, a, b, c []T)) {
	t.Helper()
	a := buildOff(n, gen, 0)
	b := buildOff(n, gen, offB)
	c := buildOff(n, gen, offC)

	want := make([]T, n)
	op(want, a, b, c)

	got := clone(a)
	op(got, got, b, c)
	Report(t, n, "dst=a", eq, want, got)

	got = clone(b)
	op(got, a, got, c)
	Report(t, n, "dst=b", eq, want, got)

	got = clone(c)
	op(got, a, b, got)
	Report(t, n, "dst=c", eq, want, got)
}

// Per-index offsets into the generator sequence for the second and third
// operands of Binary and Ternary, so a, b and c differ elementwise (they are
// shifted views of the same sequence, not disjoint ranges).
const (
	offB = 101
	offC = 202
)

// ForGate runs the sweep on both settings of a boolean dispatch gate (an arm64
// hasNEON or amd64 hasAVX2 var): once with the gate on (named onName) when the
// host supports it, and once forced off (the pure-Go path), restoring the gate on
// cleanup. It is the bool-gate counterpart of ForTiers for packages that dispatch
// on a mutable feature flag rather than a function-pointer table.
func ForGate(t *testing.T, gate *bool, onName string, run func(t *testing.T)) {
	t.Helper()
	saved := *gate
	ForTiers(t, []Tier{
		{Name: onName, Bind: func() { *gate = saved }, Supported: saved},
		{Name: "Go", Bind: func() { *gate = false }, Supported: true},
	}, run)
}

// Tier names one bindable kernel configuration (an amd64 SIMD tier, or the arm64
// Go/NEON gate) together with the CPU support it needs.
type Tier struct {
	// Name is the sub-test name for this tier.
	Name string
	// Bind installs the tier's kernels (an init* function on amd64, a hasNEON
	// flip on arm64).
	Bind func()
	// Supported reports whether the host can run this tier.
	Supported bool
}

// ForTiers binds each supported tier in turn and runs run under it, then restores
// the host default on cleanup. tiers must be listed in descending priority so the
// first supported entry is the configuration the package's own init would have
// chosen; that entry is re-bound on cleanup so the sweep leaves global dispatch
// state untouched for the rest of the suite.
func ForTiers(t *testing.T, tiers []Tier, run func(t *testing.T)) {
	t.Helper()
	for _, tr := range tiers {
		if tr.Supported {
			t.Cleanup(tr.Bind)
			break
		}
	}
	for _, tr := range tiers {
		if !tr.Supported {
			continue
		}
		tr.Bind()
		t.Run(tr.Name, run)
	}
}

// UnaryCase builds a Case that runs Unary for op(dst, a). Its Alloc check runs the
// dst==a overlay through AllocsPerRun.
func UnaryCase[T any](name string, eq func(x, y T) bool, gen func(i int) T, op func(dst, a []T)) Case {
	return Case{
		Name:  name,
		Check: func(t *testing.T, n int) { t.Helper(); Unary(t, n, eq, gen, op) },
		Alloc: func(t *testing.T) {
			t.Helper()
			a := buildOff(allocSize, gen, 0)
			ZeroAlloc(t, name+" dst=a", func() { op(a, a) })
		},
	}
}

// BinaryCase builds a Case that runs Binary for op(dst, a, b). Its Alloc check runs
// every overlay Binary claims (dst==a, dst==b, dst==a==b), each on fresh inputs, so
// an allocation reachable only in one overlay mode is still caught.
func BinaryCase[T any](name string, eq func(x, y T) bool, gen func(i int) T, op func(dst, a, b []T)) Case {
	return Case{
		Name:  name,
		Check: func(t *testing.T, n int) { t.Helper(); Binary(t, n, eq, gen, op) },
		Alloc: func(t *testing.T) {
			t.Helper()
			fresh := func() (a, b []T) {
				return buildOff(allocSize, gen, 0), buildOff(allocSize, gen, offB)
			}
			a, b := fresh()
			ZeroAlloc(t, name+" dst=a", func() { op(a, a, b) })
			a, b = fresh()
			ZeroAlloc(t, name+" dst=b", func() { op(b, a, b) })
			a, _ = fresh()
			ZeroAlloc(t, name+" dst=a=b", func() { op(a, a, a) })
		},
	}
}

// TernaryCase builds a Case that runs Ternary for op(dst, a, b, c). Its Alloc check
// runs every overlay Ternary claims (dst==a, dst==b, dst==c), each on fresh inputs,
// so an allocation reachable only in one overlay mode is still caught.
func TernaryCase[T any](name string, eq func(x, y T) bool, gen func(i int) T, op func(dst, a, b, c []T)) Case {
	return Case{
		Name:  name,
		Check: func(t *testing.T, n int) { t.Helper(); Ternary(t, n, eq, gen, op) },
		Alloc: func(t *testing.T) {
			t.Helper()
			fresh := func() (a, b, c []T) {
				return buildOff(allocSize, gen, 0), buildOff(allocSize, gen, offB), buildOff(allocSize, gen, offC)
			}
			a, b, c := fresh()
			ZeroAlloc(t, name+" dst=a", func() { op(a, a, b, c) })
			a, b, c = fresh()
			ZeroAlloc(t, name+" dst=b", func() { op(b, a, b, c) })
			a, b, c = fresh()
			ZeroAlloc(t, name+" dst=c", func() { op(c, a, b, c) })
		},
	}
}
