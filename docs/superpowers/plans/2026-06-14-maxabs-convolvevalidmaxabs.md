# MaxAbs + fused ConvolveValid+MaxAbs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MaxAbs`, `ConvolveValidMaxAbs`, and `ConvolveValidMaxAbsMulti` to the `f64` and `f32` packages (tphakala/simd#138).

**Architecture:** `MaxAbs` is a vectorized infinity-norm reduction reusing the `Sum`/`Min`/`Max` dispatch machinery (Go + AVX2 + SSE2 + NEON kernels; AVX-512 routes to AVX2). `ConvolveValidMaxAbs[Multi]` are Go-level fusions over the already-dispatched SIMD `dotProduct`: no scratch buffer, no second pass, no new assembly.

**Tech Stack:** Pure Go + hand-written Plan 9 assembly (amd64 AVX/SSE2, arm64 NEON as hand-encoded `WORD`s). Tests with `testing`, fuzzing, asmcheck.

**Verification commands** (run from repo root):
- amd64: `go test ./f64/ ./f32/` and `go vet ./f64/ ./f32/`
- lint: `golangci-lint run ./f64/... ./f32/...`
- arm64 (NEON): cross-compile test binary, copy to `thakala@rpi5.local`, run there.
  `GOOS=linux GOARCH=arm64 go test -c -o /tmp/f64.arm64.test ./f64/` then
  `scp /tmp/f64.arm64.test rpi5.local:/tmp/ && ssh rpi5.local /tmp/f64.arm64.test`
- asmcheck: `go test ./ -run TestAsm` (validates NEON WORD encodings).

---

## Phase 1: f64 MaxAbs

### Task 1: f64 maxAbsGo reference + public API + dispatch (Go fallback only)

**Files:**
- Modify: `f64/f64_go.go` (add `maxAbsGo` after `maxGo`)
- Modify: `f64/f64.go` (add `MaxAbs` public func after `Max`)
- Modify: `f64/f64_amd64.go` (add `maxAbsImpl` var, `maxAbs64` dispatcher, wire all 5 init funcs to a temporary Go impl until kernels land)
- Modify: `f64/f64_arm64.go` (add `maxAbs64` dispatcher -> Go for now)
- Modify: `f64/f64_other.go` (add `maxAbs64` -> `maxAbsGo`)
- Test: `f64/f64_test.go` (or a new `f64/maxabs_test.go`)

- [ ] **Step 1: Write the failing test** in `f64/maxabs_test.go`:

```go
package f64

import (
	"math"
	"testing"
)

// scalarMaxAbs is the independent reference: max(|a[i]|), 0 for empty.
func scalarMaxAbs(a []float64) float64 {
	m := 0.0
	for _, v := range a {
		av := math.Abs(v)
		if av > m {
			m = av
		}
	}
	return m
}

func TestMaxAbs(t *testing.T) {
	cases := [][]float64{
		nil,
		{},
		{0},
		{-0.0},
		{3, -7, 2},
		{-1, -2, -3, -4, -5, -6, -7, -8, -9},
		{1.5, -1.5, 0.25, -0.25, 100, -99.9, 0, 42},
		{math.Inf(1), -3},
		{math.Inf(-1), 3},
	}
	for i, a := range cases {
		got := MaxAbs(a)
		want := scalarMaxAbs(a)
		if got != want {
			t.Errorf("case %d: MaxAbs(%v) = %v, want %v", i, a, got, want)
		}
	}
}

func TestMaxAbsParity(t *testing.T) {
	// Lengths spanning the SIMD body, the scalar tail, and the small fallback.
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 256} {
		a := make([]float64, n)
		for i := range a {
			a[i] = math.Sin(float64(i)*0.7) * float64((i%5)-2) * 13.3
		}
		if got, want := MaxAbs(a), scalarMaxAbs(a); got != want {
			t.Errorf("n=%d: MaxAbs=%v want %v", n, got, want)
		}
	}
}

func TestMaxAbsNoAlloc(t *testing.T) {
	a := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i%7) - 3
	}
	if n := testing.AllocsPerRun(100, func() { _ = MaxAbs(a) }); n != 0 {
		t.Errorf("MaxAbs allocated %v times, want 0", n)
	}
}
```

- [ ] **Step 2: Run, verify it fails to compile** (`MaxAbs` undefined): `go test ./f64/ -run TestMaxAbs`

- [ ] **Step 3: Add `maxAbsGo`** to `f64/f64_go.go` (right after `maxGo`):

```go
// maxAbsGo returns max_i |a[i]| (the infinity norm), 0 for an empty slice.
// It is the bit-exact source of truth for the MaxAbs kernels.
func maxAbsGo(a []float64) float64 {
	m := 0.0
	for _, v := range a {
		av := math.Abs(v)
		if av > m {
			m = av
		}
	}
	return m
}
```

- [ ] **Step 4: Add `MaxAbs`** to `f64/f64.go` (right after `Max`):

```go
// MaxAbs returns the maximum absolute value in the slice (the infinity norm),
// max_i |a[i]|. Returns 0 for an empty slice.
//
// Uses AVX2/SSE2 on AMD64 (AVX-512 CPUs reuse the AVX2 kernel), NEON on ARM64,
// with a pure Go fallback. a is read-only; the call allocates nothing.
//
// NaN handling: |NaN| is NaN and compares false, so the Go path skips NaN. On the
// SIMD paths NaN handling is architecture-dependent, matching [Min] and [Max].
// Callers needing strict NaN semantics should filter NaN first.
func MaxAbs(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return maxAbs64(a)
}
```

- [ ] **Step 5: Wire dispatch.** In `f64/f64_amd64.go`: add `maxAbsImpl reduceFunc` to the `var (...)` block (next to `maxImpl`); add `maxAbsImpl = maxAbsGo` to **all five** init funcs for now (initAVX512, initAVX, initAVXNoFMA, initSSE2, initGo); add the dispatcher:

```go
func maxAbs64(a []float64) float64 {
	if len(a) < minSIMDElements {
		return maxAbsGo(a)
	}
	return maxAbsImpl(a)
}
```

In `f64/f64_arm64.go` add a temporary dispatcher (replaced in Task 3):

```go
func maxAbs64(a []float64) float64 {
	return maxAbsGo(a)
}
```

In `f64/f64_other.go` add: `func maxAbs64(a []float64) float64 { return maxAbsGo(a) }`

- [ ] **Step 6: Run tests, verify pass:** `go test ./f64/ -run TestMaxAbs` -> PASS.

- [ ] **Step 7: Commit:**

```bash
git add f64/
git commit -m "f64: add MaxAbs infinity-norm reduction (Go path)"
```

### Task 2: f64 maxAbsAVX + maxAbsSSE2 kernels

**Files:**
- Modify: `f64/f64_amd64.s` (add two kernels)
- Modify: `f64/f64_amd64.go` (replace temporary Go wiring; add `//go:noescape` decls)
- Test: extend `f64/maxabs_test.go` (parity already covers it once kernels are active)

- [ ] **Step 1: Add `maxAbsAVX`** to `f64/f64_amd64.s` (mirror `maxAVX` at line ~536 with the abs mask folded in):

```
// func maxAbsAVX(a []float64) float64
TEXT ·maxAbsAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD absf64mask<>(SB), Y2   // Y2 = abs mask (X2 = low 128 for the tail)
    VMOVUPD (SI), Y0
    VANDPD Y0, Y2, Y0             // Y0 = |a[0:4]|
    ADDQ $32, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   maxabs_avx_reduce

maxabs_avx_loop4:
    VMOVUPD (SI), Y1
    VANDPD Y1, Y2, Y1
    VMAXPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs_avx_loop4

maxabs_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VMAXPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMAXSD X0, X1, X0

    ANDQ $3, CX
    JZ   maxabs_avx_done

maxabs_avx_scalar:
    VMOVSD (SI), X1
    VANDPD X1, X2, X1
    VMAXSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  maxabs_avx_scalar

maxabs_avx_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET
```

- [ ] **Step 2: Add `maxAbsSSE2`** to `f64/f64_amd64.s` (mirror `maxSSE2` at line ~2746 with `ANDPD`):

```
// func maxAbsSSE2(a []float64) float64
TEXT ·maxAbsSSE2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPD absf64mask<>(SB), X2
    MOVUPD (SI), X0
    ANDPD X2, X0                 // X0 = |a[0:2]|
    ADDQ $16, SI
    SUBQ $2, CX

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   maxabs_sse2_reduce

maxabs_sse2_loop2:
    MOVUPD (SI), X1
    ANDPD X2, X1
    MAXPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  maxabs_sse2_loop2

maxabs_sse2_reduce:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    MAXSD X1, X0

    ANDQ $1, CX
    JZ   maxabs_sse2_done

    MOVSD (SI), X1
    ANDPD X2, X1
    MAXSD X1, X0

maxabs_sse2_done:
    MOVSD X0, ret+24(FP)
    RET
```

- [ ] **Step 3: Declare + wire.** In `f64/f64_amd64.go`: add near the other reduction decls:

```go
//go:noescape
func maxAbsAVX(a []float64) float64

//go:noescape
func maxAbsSSE2(a []float64) float64
```

Then set the impls: `initAVX512`, `initAVX`, `initAVXNoFMA` -> `maxAbsImpl = maxAbsAVX`; `initSSE2` -> `maxAbsImpl = maxAbsSSE2`; `initGo` stays `maxAbsGo`.

- [ ] **Step 4: Run, verify pass:** `go test ./f64/ -run TestMaxAbs` -> PASS. Also force SSE2 via the existing vector-path test mechanism if present (`vector_path_test.go`); otherwise the parity test on this AVX2 host covers AVX.

- [ ] **Step 5: Vet + lint:** `go vet ./f64/` and `golangci-lint run ./f64/...`.

- [ ] **Step 6: Commit:**

```bash
git add f64/
git commit -m "f64: add AVX2 + SSE2 MaxAbs kernels"
```

### Task 3: f64 maxAbsNEON kernel

**Files:**
- Modify: `f64/f64_arm64.s` (add kernel)
- Modify: `f64/f64_arm64.go` (replace temporary dispatcher; add `//go:noescape` decl)

- [ ] **Step 1: Add `maxAbsNEON`** to `f64/f64_arm64.s` (mirror `maxNEON` at line ~330 with `FABS V*.2D`). Encodings: `FABS V0.2D = 0x4EE0F800`, `FABS V1.2D = 0x4EE0F821` (`0x4EE0F800 | (Vn=1<<5) | Vd=1`):

```
// func maxAbsNEON(a []float64) float64
TEXT ·maxAbsNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.D2]
    WORD $0x4EE0F800           // FABS V0.2D, V0.2D
    SUB $2, R1

    LSR $1, R1, R2
    CBZ R2, maxabs_scalar

maxabs_loop2:
    VLD1.P 16(R0), [V1.D2]
    WORD $0x4EE0F821           // FABS V1.2D, V1.2D
    WORD $0x4E61F400           // FMAX V0.2D, V0.2D, V1.2D
    SUB $1, R2
    CBNZ R2, maxabs_loop2

maxabs_scalar:
    AND $1, R1
    CBZ R1, maxabs_reduce

    VDUP V0.D[1], V1.D2
    FMAXD F0, F1, F0
    FMOVD (R0), F1
    FABSD F1, F1
    FMAXD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

maxabs_reduce:
    VDUP V0.D[1], V1.D2
    FMAXD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET
```

(`FABSD` is a Go-assembler scalar mnemonic; if the assembler rejects it, hand-encode scalar `FABS D1,D1 = 0x1E60C021`. Verify during build.)

- [ ] **Step 2: Declare + wire.** In `f64/f64_arm64.go`: replace the temporary `maxAbs64` with:

```go
func maxAbs64(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return maxAbsNEON(a)
	}
	return maxAbsGo(a)
}
```

Add with the other NEON decls:

```go
//go:noescape
func maxAbsNEON(a []float64) float64
```

- [ ] **Step 3: Build arm64 + asmcheck on host:** `GOOS=linux GOARCH=arm64 go build ./f64/` and `go test ./ -run TestAsm` (asmcheck validates the new WORDs).

- [ ] **Step 4: Run on rpi5:**

```bash
GOOS=linux GOARCH=arm64 go test -c -o /tmp/f64.arm64.test ./f64/
scp /tmp/f64.arm64.test rpi5.local:/tmp/
ssh rpi5.local /tmp/f64.arm64.test -test.run 'TestMaxAbs|TestAsm'
```
Expected: PASS.

- [ ] **Step 5: Commit:**

```bash
git add f64/
git commit -m "f64: add NEON MaxAbs kernel"
```

---

## Phase 2: f32 MaxAbs

### Task 4: f32 MaxAbs (Go + AVX + SSE + NEON), mirroring Phase 1

**Files:** the `f32/` analogues of every file in Phase 1, plus `f32/maxabs_test.go`.

- [ ] **Step 1:** Write `f32/maxabs_test.go` identical to `f64/maxabs_test.go` but with `float32`, `package f32`, and `float32(math.Abs(float64(v)))` in the reference (or a `float32`-native abs helper). Use `float32` literals.

- [ ] **Step 2:** Add `maxAbsGo` to `f32/f32_go.go`, `MaxAbs` to `f32/f32.go`, `maxAbsImpl`/`maxAbs64`/init wiring to `f32/f32_amd64.go`, dispatcher to `f32/f32_arm64.go`, fallback to `f32/f32_other.go` — same shapes as Phase 1.

- [ ] **Step 3:** Add `maxAbsAVX` to `f32/f32_amd64.s` mirroring f32 `maxAVX` (line ~539, 8-wide): load `VMOVUPS absf32mask<>(SB), Y2`; `VANDPS` each loaded vector; `SHRQ $3` / `ANDQ $7` tail; scalar uses `VMOVSS`+`VANDPS X1,X2,X1`+`VMAXSS`. Add `maxAbsSSE` mirroring f32 `maxSSE` (line ~1812) with `MOVUPS absf32mask<>(SB), X2` + `ANDPS X2, X*`.

- [ ] **Step 4:** Add `maxAbsNEON` to `f32/f32_arm64.s` mirroring f32 `maxNEON` (4S, 4 lanes/vector) with `FABS V*.4S`. Encodings: `FABS V0.4S = 0x4EA0F800`, `FABS V1.4S = 0x4EA0F821`. Reduce path matches f32 `maxNEON` (uses `FMAXP`/`VDUP` lane reduction as in the existing kernel — copy its reduce sequence verbatim, prepend FABS to each load, and FABS the scalar tail element).

- [ ] **Step 5:** Wire `//go:noescape` decls and init funcs in `f32/f32_amd64.go` (AVX512/AVX/AVXNoFMA -> `maxAbsAVX`, SSE2 -> `maxAbsSSE`, Go -> `maxAbsGo`) and `f32/f32_arm64.go` (`hasNEON && len>=4` -> `maxAbsNEON` else Go; note f32 NEON threshold is 4, matching f32 `max32`).

- [ ] **Step 6:** `go test ./f32/`, `go vet ./f32/`, `golangci-lint run ./f32/...`, arm64 build + asmcheck, run on rpi5.

- [ ] **Step 7: Commit:** `git commit -am "f32: add MaxAbs (Go + AVX + SSE + NEON)"`

---

## Phase 3: f64 + f32 ConvolveValidMaxAbs / ConvolveValidMaxAbsMulti

### Task 5: f64 ConvolveValidMaxAbs + Multi (Go-level fusion)

**Files:**
- Modify: `f64/f64.go` (public funcs + fusion helpers)
- Test: `f64/convolve_maxabs_test.go`

- [ ] **Step 1: Write failing test** `f64/convolve_maxabs_test.go`:

```go
package f64

import (
	"math"
	"testing"
)

// reference: full valid convolution then scalar abs-max.
func refConvolveValidMaxAbs(signal, kernel []float64) float64 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return 0
	}
	out := make([]float64, len(signal)-len(kernel)+1)
	ConvolveValid(out, signal, kernel)
	return scalarMaxAbs(out)
}

func TestConvolveValidMaxAbs(t *testing.T) {
	for _, sl := range []int{1, 4, 7, 16, 33, 128} {
		for _, kl := range []int{1, 2, 3, 5, 8} {
			if sl < kl {
				continue
			}
			signal := make([]float64, sl)
			kernel := make([]float64, kl)
			for i := range signal {
				signal[i] = math.Sin(float64(i)*0.3) - 0.4*float64(i%3)
			}
			for i := range kernel {
				kernel[i] = 0.5 - float64(i)*0.1
			}
			got := ConvolveValidMaxAbs(signal, kernel)
			want := refConvolveValidMaxAbs(signal, kernel)
			if math.Abs(got-want) > 1e-12*(1+math.Abs(want)) {
				t.Errorf("sl=%d kl=%d: got %v want %v", sl, kl, got, want)
			}
		}
	}
}

func TestConvolveValidMaxAbsEdge(t *testing.T) {
	if got := ConvolveValidMaxAbs(nil, nil); got != 0 {
		t.Errorf("nil/nil = %v want 0", got)
	}
	if got := ConvolveValidMaxAbs([]float64{1, 2}, []float64{1, 2, 3}); got != 0 {
		t.Errorf("short signal = %v want 0", got)
	}
}

func TestConvolveValidMaxAbsMulti(t *testing.T) {
	signal := make([]float64, 64)
	for i := range signal {
		signal[i] = math.Cos(float64(i) * 0.21)
	}
	kernels := [][]float64{
		{0.2, -0.1, 0.05, 0.3},
		{-0.4, 0.4, -0.2, 0.1},
		{0.9, 0.0, -0.9, 0.5},
		{0.1, 0.1, 0.1, 0.1},
	}
	got := ConvolveValidMaxAbsMulti(signal, kernels)
	want := 0.0
	for _, k := range kernels {
		if v := refConvolveValidMaxAbs(signal, k); v > want {
			want = v
		}
	}
	if math.Abs(got-want) > 1e-12*(1+math.Abs(want)) {
		t.Errorf("Multi got %v want %v", got, want)
	}
}

func TestConvolveValidMaxAbsMultiPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on mismatched kernel lengths")
		}
	}()
	ConvolveValidMaxAbsMulti([]float64{1, 2, 3, 4}, [][]float64{{1, 2}, {1, 2, 3}})
}

func TestConvolveValidMaxAbsNoAlloc(t *testing.T) {
	signal := make([]float64, 256)
	for i := range signal {
		signal[i] = float64(i%9) - 4
	}
	kernel := []float64{0.1, -0.2, 0.3, -0.4}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbs(signal, kernel) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbs allocated %v, want 0", n)
	}
	kernels := [][]float64{{0.1, -0.2, 0.3, -0.4}, {0.2, 0.2, 0.2, 0.2}}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbsMulti(signal, kernels) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbsMulti allocated %v, want 0", n)
	}
}
```

- [ ] **Step 2: Run, verify fail (undefined):** `go test ./f64/ -run TestConvolveValidMaxAbs`

- [ ] **Step 3: Implement** in `f64/f64.go` (after `ConvolveValidMulti`):

```go
// ConvolveValidMaxAbs returns max(|valid-convolution output|) without
// materializing the output slice: the peak (infinity norm) of the FIR applied to
// signal with no zero-padding. Returns 0 when len(kernel) == 0 or
// len(signal) < len(kernel). Each output is a SIMD dot product; the abs-max is
// fused in, so there is no scratch buffer and no second pass. Allocates nothing.
func ConvolveValidMaxAbs(signal, kernel []float64) float64 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return 0
	}
	return convolveValidMaxAbs64(signal, kernel)
}

func convolveValidMaxAbs64(signal, kernel []float64) float64 {
	kLen := len(kernel)
	validLen := len(signal) - kLen + 1
	var m float64 // abs values are >= 0, so 0 is the correct identity
	for i := 0; i < validLen; i++ {
		if v := math.Abs(dotProduct(signal[i:i+kLen], kernel)); v > m {
			m = v
		}
	}
	return m
}

// ConvolveValidMaxAbsMulti returns the single maximum of |valid-convolution
// output| across every kernel applied to signal, without materializing any
// output. This is the polyphase true-peak primitive: pass the N phase kernels,
// get back the peak of the reconstructed signal in one call. Returns 0 when
// kernels is empty, the first kernel is empty, or len(signal) < kernel length.
// Allocates nothing.
//
// Panics if the kernels do not all share one length, matching [ConvolveValidMulti].
func ConvolveValidMaxAbsMulti(signal []float64, kernels [][]float64) float64 {
	numKernels := len(kernels)
	if numKernels == 0 {
		return 0
	}
	kLen := len(kernels[0])
	if kLen == 0 || len(signal) < kLen {
		return 0
	}
	for i := 1; i < numKernels; i++ {
		if len(kernels[i]) != kLen {
			panic("simd: all kernels must have the same length")
		}
	}
	var m float64
	for _, kernel := range kernels {
		if km := convolveValidMaxAbs64(signal, kernel); km > m {
			m = km
		}
	}
	return m
}
```

Confirm `math` is imported in `f64/f64.go` (it is, used elsewhere).

- [ ] **Step 4: Run, verify pass:** `go test ./f64/ -run TestConvolveValidMaxAbs` -> PASS.

- [ ] **Step 5: Vet + lint:** `go vet ./f64/`; `golangci-lint run ./f64/...`.

- [ ] **Step 6: Commit:** `git commit -am "f64: add ConvolveValidMaxAbs + ConvolveValidMaxAbsMulti"`

### Task 6: f32 ConvolveValidMaxAbs + Multi

**Files:**
- Modify: `f32/f32.go`
- Test: `f32/convolve_maxabs_test.go`

- [ ] **Step 1:** Write `f32/convolve_maxabs_test.go` mirroring Task 5 with `float32` (tolerance `1e-4` relative, since f32). Reference uses `ConvolveValid` (f32) + `scalarMaxAbs` (f32 version).

- [ ] **Step 2:** Implement `ConvolveValidMaxAbs`, `convolveValidMaxAbsF32`, `ConvolveValidMaxAbsMulti` in `f32/f32.go`, identical shape to f64 with `float32`. Abs via `float32(math.Abs(float64(v)))` or a local `absF32`.

- [ ] **Step 3:** `go test ./f32/ -run TestConvolveValidMaxAbs` -> PASS; vet; lint.

- [ ] **Step 4: Commit:** `git commit -am "f32: add ConvolveValidMaxAbs + ConvolveValidMaxAbsMulti"`

---

## Phase 4: Examples + Docs

### Task 7: Example tests

**Files:** `f64/example_test.go`, `f32/example_test.go`

- [ ] **Step 1:** Add `ExampleMaxAbs`, `ExampleConvolveValidMaxAbs`, `ExampleConvolveValidMaxAbsMulti` to each package's `example_test.go`, each with `// Output:` blocks. Run `go test ./f64/ ./f32/ -run Example` -> PASS.

- [ ] **Step 2: Commit:** `git commit -am "f64,f32: add MaxAbs/ConvolveValidMaxAbs examples"`

### Task 8: doc.go + README

**Files:** `doc.go`, `README.md`

- [ ] **Step 1:** Add the three new functions (f64 + f32) to the function index in `doc.go`, in the reductions / convolution sections, matching the existing format.

- [ ] **Step 2:** Add README entries: a `MaxAbs` row/section under the reduction ops and a `ConvolveValidMaxAbs`/`ConvolveValidMaxAbsMulti` section under convolution, describing the peak-detection use case and the empty/panic behavior. Mirror the format of neighboring entries (e.g. `Max`, `ConvolveValidMulti`).

- [ ] **Step 3:** `go test ./...` (whole tree) and `golangci-lint run ./...` -> PASS.

- [ ] **Step 4: Commit:** `git commit -am "docs: document MaxAbs and ConvolveValidMaxAbs[Multi]"`

---

## Phase 5: Final verification + PR

### Task 9: Cross-arch verification and push

- [ ] **Step 1:** Full amd64 suite: `go test ./...` -> PASS.
- [ ] **Step 2:** Cross-compile + run full f64/f32 suites on rpi5 (arm64 NEON), confirm PASS.
- [ ] **Step 3:** `golangci-lint run ./...` clean.
- [ ] **Step 4:** Run `/gate` (or equivalent review) over the diff; address findings.
- [ ] **Step 5:** Push branch, open PR referencing #138, run watch-pr loop to green + merge.

---

## Self-Review notes

- Spec coverage: `MaxAbs` (Tasks 1-4), `ConvolveValidMaxAbs[Multi]` (Tasks 5-6), tests (each task), examples (Task 7), docs (Task 8), cross-arch (Task 9). All spec sections mapped.
- AVX-512 routing: handled in Task 2/4 init wiring (AVX512 init -> `maxAbsAVX`).
- Naming consistency: `maxAbs64`/`maxAbsImpl`/`maxAbsGo`/`maxAbsAVX`/`maxAbsSSE2`/`maxAbsNEON` (f64); `maxAbsF32`?? — f32 dispatcher is `maxAbs64`'s analogue. Match the f32 repo convention: f32 uses `max32` for the dispatcher, `maxGo`/`maxAVX`/`maxSSE`/`maxNEON` for kernels. So f32 names: dispatcher `maxAbs32`, kernels `maxAbsGo`/`maxAbsAVX`/`maxAbsSSE`/`maxAbsNEON`. Confirm against `f32/f32_amd64.go` `max32` at implementation time.
- Convolve helpers call the dispatched `dotProduct`, so they are SIMD on every backend with no new asm.
```
