# MaxAbs reduction + fused ConvolveValid+MaxAbs for f64/f32

GitHub issue: tphakala/simd#138

## Problem

A downstream EBU R 128 / ITU-R BS.1770-4 true-peak path runs a 4x polyphase
oversampling FIR and needs the peak (max absolute value) of the oversampled
signal. The convolution already runs through the SIMD dot-product path via
`ConvolveValid`, but two costs remain:

1. The abs-max reduction over each convolution output is a scalar loop that does
   not vectorize (~7.5% of CPU in the caller's profile).
2. A per-call scratch buffer holds the convolution outputs only so the caller can
   scan them for the peak.

## Goals

Add to both the `f64` and `f32` packages (identical API shapes, `float32`
substituted):

```go
// Vectorized infinity-norm reduction. Returns 0 for an empty slice.
func MaxAbs(a []float64) float64

// max(|valid-convolution output|) without materializing the output slice.
func ConvolveValidMaxAbs(signal, kernel []float64) float64

// Single maximum across all kernels' valid-convolution outputs.
func ConvolveValidMaxAbsMulti(signal []float64, kernels [][]float64) float64
```

Non-goals: a dedicated AVX-512 kernel (no hardware here to validate it; AVX-512
machines reuse the AVX2 kernel), and a fully fused single-pass convolution+max
assembly kernel (the dot product already dominates; Go-level fusion captures the
stated wins).

## Design

### 1. `MaxAbs` — vectorized reduction

`MaxAbs` is a pure reduction in the same family as `Sum`/`Min`/`Max`, so it reuses
that infrastructure.

- **Public wrapper** (`f64.go` / `f32.go`): `len(a)==0` returns 0; otherwise call
  the dispatched `maxAbs64` / `maxAbsF32`.
- **Dispatch** (`f64_amd64.go` / `f32_amd64.go`): a new `maxAbsImpl reduceFunc`
  function pointer, assigned in every `init*` path:
  - `initAVX512` -> `maxAbsAVX` (reuse the tested AVX2 kernel)
  - `initAVX` / `initAVXNoFMA` -> `maxAbsAVX`
  - `initSSE2` -> `maxAbsSSE2` (f64) / `maxAbsSSE` (f32)
  - `initGo` -> `maxAbsGo`
  - The dispatcher `maxAbs64` falls back to `maxAbsGo` when
    `len(a) < minSIMDElements`, mirroring `min64`/`max64` (the SIMD kernels do an
    initial full-width vector load and must not read past the slice).
- **Kernels** (each mirrors the existing `max*` reduction with abs folded into the
  per-element load):
  - `maxAbsGo` (`f64_go.go` / `f32_go.go`): bit-exact oracle.
    `m := 0; for _, v := range a { av := abs(v); if av > m { m = av } }`.
  - `maxAbsAVX` (`f64_amd64.s` / `f32_amd64.s`): the `maxAVX` body with
    `VANDPD`/`VANDPS` against the existing `absf64mask<>` / `absf32mask<>` constant
    applied to each loaded vector and to the scalar tail. 4-wide f64 / 8-wide f32.
  - `maxAbsSSE2` (f64) / `maxAbsSSE` (f32): the `maxSSE2`/`maxSSE` body with
    `ANDPD`/`ANDPS` masking.
  - `maxAbsNEON` (`f64_arm64.s` / `f32_arm64.s`): the `maxNEON` body with `FABS`
    (`0x4EE0F800` family for 2D, `0x4EA0F800` family for 4S) applied to each
    loaded vector and the scalar tail. Hand-encoded `WORD`s with decoded-mnemonic
    comments, cross-checked by the asmcheck suite.
- **`_other.go`**: `func maxAbs64(a []float64) float64 { return maxAbsGo(a) }`.
- **NaN**: `|NaN| = NaN`; comparisons against NaN are false, so the Go oracle
  skips NaN. The SIMD `MAX` instructions make NaN handling architecture-dependent,
  identical to the existing `Min`/`Max`. Documented with the same caveat; bit-exact
  tests use non-NaN inputs.
- **Empty**: returns 0.

### 2. `ConvolveValidMaxAbs` / `ConvolveValidMaxAbsMulti` — Go-level fusion

These reuse the already-dispatched SIMD `dotProduct`; no new assembly. Defined in
`f64.go` / `f32.go` (which already call the arch-dispatched `dotProduct`), so the
same source serves every backend.

```go
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
        v := math.Abs(dotProduct(signal[i:i+kLen], kernel))
        if v > m {
            m = v
        }
    }
    return m
}
```

This removes the scratch buffer and the second abs-max scan; each output is still
a SIMD dot product.

`ConvolveValidMaxAbsMulti` mirrors `ConvolveValidMulti`'s validation contract:

```go
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

Empty `kernels`, an empty first kernel, or a signal shorter than the kernel return
0. Unequal kernel lengths panic, matching `ConvolveValidMulti` and the polyphase
use case (all phase kernels share a length).

## Testing

Following the repo conventions (one Go reference, parity vs the active SIMD path,
bit-exactness vs the per-element scalar path, allocation-free assertion):

- `maxAbsGo` parity against the SIMD path on randomized data spanning lengths that
  exercise the full-width body, the scalar tail, and the sub-`minSIMDElements`
  fallback; include negatives, zeros, +/-Inf, and the empty slice.
- Bit-exactness: `MaxAbs(a)` equals the scalar `max(|a[i]|)` reference.
- `ConvolveValidMaxAbs` parity against `ConvolveValid` followed by a scalar
  abs-max over the output; same for `Multi` across multiple kernels. Cover
  `validLen == 0`, empty kernel, short signal, and the panic on mismatched kernel
  lengths.
- `testing.AllocsPerRun(... ) == 0` for `MaxAbs`, `ConvolveValidMaxAbs`, and
  `ConvolveValidMaxAbsMulti` (the caller passes the kernels slice; no internal
  allocation).
- Fuzz tests for `MaxAbs` and `ConvolveValidMaxAbs` (parity oracle).
- Runnable `Example` functions.
- asmcheck validates every new NEON `WORD` against `golang.org/x/arch/arm64asm`.
- amd64 paths (SSE2, AVX2, Go) run natively here; NEON runs on the rpi5 aarch64
  host.

## Files touched

Per package (`f64`, `f32`):
- `*.go`: public API + `ConvolveValidMaxAbs[Multi]` fusion helpers.
- `*_amd64.go`: `maxAbsImpl` var, `init*` wiring, `maxAbs64` dispatcher, `//go:noescape` decls.
- `*_amd64.s`: `maxAbsAVX`, `maxAbsSSE2`/`maxAbsSSE`.
- `*_go.go`: `maxAbsGo`.
- `*_arm64.go`: `maxAbs64` dispatcher, `//go:noescape` decl.
- `*_arm64.s`: `maxAbsNEON`.
- `*_other.go`: `maxAbs64` Go fallback.
- new `*_test.go` coverage.

Repo-level: `doc.go` function index, `README.md` sections.
