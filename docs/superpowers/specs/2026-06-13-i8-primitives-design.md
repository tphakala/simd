# i8: high-impact int8 SIMD primitives (v1)

Date: 2026-06-13
Status: approved (Approach A, seven-op v1)
Branch: `i8-primitives`

## Goal

Add an `i8` package providing SIMD-accelerated operations on `[]int8`, mirroring
the existing per-type packages (`i16`, `i32`, `f32`) in structure, conventions,
and test discipline. int8 has a hard constraint that shapes the API: at 8-bit
width arithmetic overflows almost immediately (`-128..127`), so the operations
that make sense are not a 1:1 copy of `i32`. v1 ships the genuinely high-impact,
clearly-int8-shaped kernels and defers float-quantization (which needs its own
scale/zero-point convention design) and 8-bit movement to follow-ups.

## Scope: seven operations

| Function | Signature | Semantics |
|----------|-----------|-----------|
| `AddSaturate` | `func AddSaturate(dst, a, b []int8)` | `dst[i] = clamp(int(a[i])+int(b[i]), -128, 127)` |
| `SubSaturate` | `func SubSaturate(dst, a, b []int8)` | `dst[i] = clamp(int(a[i])-int(b[i]), -128, 127)` |
| `ToInt16` | `func ToInt16(dst []int16, src []int8)` | `dst[i] = int16(src[i])` (sign-extend, exact) |
| `ToInt32` | `func ToInt32(dst []int32, src []int8)` | `dst[i] = int32(src[i])` (sign-extend, exact) |
| `Sum` | `func Sum(a []int8) int32` | `sum_i int32(a[i])`, int32 two's-complement accumulation |
| `MinMax` | `func MinMax(a []int8) (min, max int8)` | signed per-slice min and max; empty returns `(0,0)` |
| `DotProduct` | `func DotProduct(a, b []int8) int32` | `sum_i int32(a[i])*int32(b[i])`, int32 two's-complement accumulation |

`DotProduct` is the marquee int8 kernel: it is the inner loop of quantized
matmul/conv. On ARM64 with `FEAT_DotProd` a single `SDOT` does 16 int8 MACs into
4 int32 lanes; on AVX2 `VPMADDWD` over sign-extended bytes does the same work
without a new instruction-set tier.

### API conventions (match i16/i32)

- Binary ops (`AddSaturate`, `SubSaturate`) clamp `n := min(len(dst), len(a),
  len(b))`, operate on `dst[:n]/a[:n]/b[:n]`, and leave trailing `dst` capacity
  untouched. Conversions clamp `n := min(len(dst), len(src))`.
- Reductions (`Sum`, `DotProduct`) operate over `min`-clamped lengths; `Sum([])
  == 0`, `DotProduct` over empty `== 0`.
- `MinMax` guards the empty case in the public function (returns `(0,0)`), so the
  dispatch and kernels always see a non-empty slice, matching `i32.MinMax`.
- **No `Unsafe` variants.** The integer packages (`i16`, `i32`) ship none; v1
  follows that precedent. (The `Unsafe` convention is f32/f64-only today.)
- Zero allocations; all results write into caller-provided slices or are scalar
  returns. Thread-safe (read-only inputs, no shared state).

### Accumulation and bit-exactness

`Sum` and `DotProduct` accumulate in `int32` with two's-complement wraparound,
exactly like their pure-Go references. int32 wrapping addition is associative and
commutative mod 2^32, so every SIMD reduction order (lane-parallel accumulate
then horizontal fold) yields the identical result to the scalar left-to-right
reference. This makes the SIMD paths bit-exact by construction, not just
approximately equal, and is the property the parity tests assert.

Intermediate products never overflow their SIMD lane: `|int8 * int8| <= 16384`,
so the int16 products from `SMULL`/`VPMOVSXBW`+multiply are exact, and a
`VPMADDWD`/`SADALP` pair-sum (`<= 32768`) is exact in int32. Only the final
running total can wrap, and it wraps identically to the reference.

## File layout (mirrors i32)

```
i8/
  i8.go              public API (AddSaturate, SubSaturate, ToInt16, ToInt32,
                     Sum, MinMax, DotProduct) + package doc + dispatch helpers
                     that live on no specific arch
  i8_go.go           pure-Go reference kernels (source of truth, compiled on all
                     arches, used as the fallback)
  i8_amd64.go        amd64 dispatch: var hasAVX2 = cpu.X86.AVX2, per-op gates,
                     //go:noescape decls
  i8_amd64.s         AVX2 kernels
  i8_arm64.go        arm64 dispatch: var hasNEON, var hasDotProd, per-op gates,
                     //go:noescape decls
  i8_arm64.s         NEON kernels (base + SDOT fast path for DotProduct)
  i8_other.go        //go:build !amd64 && !arm64 -> calls the Go reference
  i8_test.go         parity + bit-exact + zero-alloc + edge-case tests
  i8_simd_test.go    SIMD-path-specific assertions (force lengths past thresholds)
  fuzz_test.go       differential fuzz vs the Go reference
  benchmark_test.go  benchmarks per op
  example_test.go    runnable example
```

## Backends

### amd64 (gate on AVX2, Go fallback below; mirrors i32)

All AVX2 mnemonics are expressible by the Go assembler (no hand-encoded BYTE/WORD
directives, so `TestNoUncheckedAmd64Encodings` stays clean).

| Op | AVX2 kernel | bytes/iter |
|----|-------------|------------|
| AddSaturate | `VPADDSB` (ymm) | 32 |
| SubSaturate | `VPSUBSB` (ymm) | 32 |
| ToInt16 | `VPMOVSXBW` (xmm->ymm) | 16 |
| ToInt32 | `VPMOVSXBD` (xmm->ymm) | 8 |
| Sum | `VPMOVSXBW` then `VPMADDWD` with an all-ones int16 vector (pair-sum to int32), `VPADDD` accumulate; horizontal-sum tail | 16 |
| MinMax | `VPMINSB`/`VPMAXSB` running over blocks, horizontal fold | 32 |
| DotProduct | `VPMOVSXBW` both operands, `VPMADDWD`, `VPADDD` accumulate; horizontal-sum tail | 16 |

Each kernel has a scalar tail for `n mod blocksize`. AVX-512 VNNI (`VPDPBUSD`)
for DotProduct is a deliberate follow-up, not v1.

### arm64 (gate on NEON; DotProduct gets an SDOT fast path under FEAT_DotProd)

NEON instructions are hand-encoded `WORD $0x...` with the decoded mnemonic in the
trailing comment, cross-checked by `asmcheck_test.go`. Confirmed encodings
(verified with `aarch64-linux-gnu-objdump`):

| Mnemonic | Encoding (example operands) | Hex |
|----------|------------------------------|-----|
| `SQADD Vd.16B, Vn.16B, Vm.16B` | V2,V0,V1 | `0x4E210C02` |
| `SQSUB Vd.16B, Vn.16B, Vm.16B` | V2,V0,V1 | `0x4E212C02` |
| `SXTL Vd.8H, Vn.8B` | V2,V0 | `0x0F08A402` |
| `SXTL2 Vd.8H, Vn.16B` | V3,V0 | `0x4F08A403` |
| `SMIN Vd.16B, Vn.16B, Vm.16B` | V0,V0,V2 | `0x4E226C00` |
| `SMAX Vd.16B, Vn.16B, Vm.16B` | V1,V1,V2 | `0x4E226421` |
| `SMINV Bd, Vn.16B` | B3,V0 | `0x4E31A803` |
| `SMAXV Bd, Vn.16B` | B4,V1 | `0x4E30A824` |
| `SADDLP Vd.8H, Vn.16B` | V5,V0 | `0x4E202805` |
| `SADALP Vd.4S, Vn.8H` | V4,V2 | `0x4E606844` |
| `SMULL Vd.8H, Vn.8B, Vm.8B` | V2,V0,V1 | `0x0E21C002` |
| `SMULL2 Vd.8H, Vn.16B, Vm.16B` | V3,V0,V1 | `0x4E21C003` |
| `SDOT Vd.4S, Vn.16B, Vm.16B` | V0,V0,V1 | `0x4E819400` |

(`ZIP`/`UZP`-style `ADDV Sd, Vn.4S` for the horizontal fold is likewise hand-
encoded; exact value confirmed during implementation.)

| Op | base-NEON kernel | SDOT fast path |
|----|------------------|----------------|
| AddSaturate | `SQADD .16B` (16/iter) | n/a |
| SubSaturate | `SQSUB .16B` | n/a |
| ToInt16 | `SXTL`/`SXTL2 .8H` (16/iter) | n/a |
| ToInt32 | `SXTL .8H` then widen `.8H->.4S` (8/iter) | n/a |
| Sum | `SADDLP .16B->.8H` then `SADALP` accumulate `.8H->.4S`; `ADDV` fold | n/a |
| MinMax | `SMIN`/`SMAX .16B` running; `SMINV`/`SMAXV` fold | n/a |
| DotProduct | `SMULL`/`SMULL2 .8H` then `SADALP` accumulate `.4S`; `ADDV` fold (16/iter) | `SDOT .4S` accumulate (16/iter); `ADDV` fold |

DotProduct dispatch: `hasDotProd` selects the `SDOT` kernel, else `hasNEON`
selects the `SMULL`/`SADALP` base kernel, else Go. The base path runs on any
NEON core (Cortex-A53/A72), the SDOT path on cores with `FEAT_DotProd` (the
Pi 5's Cortex-A76, our arm64 test host).

## cpu package change

Add `FEAT_DotProd` detection:

- `cpu.Features.DOTPROD bool` field (comment: `FEAT_DotProd - SDOT/UDOT int8 dot
  product`).
- `cpu.HasDOTPROD() bool` accessor.
- `cpu_arm64.go` (non-darwin): `ARM64.DOTPROD = cpu.ARM64.HasASIMDDP`.
- darwin arm64 path: set from its existing detection source (Apple M1+ has it);
  match how the file already sets NEON/FP16.
- `SIMD_DISABLE`: add a `dotprod` token, and have `clearNEON` also clear DOTPROD
  (it is a NEON extension, so disabling NEON must disable it). `Info()` is left
  reporting the broad tier; DOTPROD is an op-level fast path, not a tier.
- amd64 leaves `DOTPROD` false (ARM-only feature), like NEON.

## asmcheck change

`arm64asm` (golang.org/x/arch) cannot decode `SDOT`/`UDOT` (FEAT_DotProd), so a
raw `SDOT` WORD currently trips the "undecodable WORD that is not FP16 (.8H)"
hard failure in `asmcheck_test.go`. Extend the sanctioned-undecodable predicate:
a WORD whose comment names a DotProd mnemonic (`SDOT`/`UDOT`) is deferred to the
objdump cross-check exactly like `.8H` FP16 directives. `aarch64-linux-gnu-
objdump -D -b binary -m aarch64 -EL` decodes `SDOT` with no march flag (verified),
so the existing `DisassembleWords` path needs no change; only the predicate and a
rename of the `fp16`-specific identifiers/messages to a neutral "deferred to
objdump" naming. Behavior when objdump is absent stays lenient (accept unchecked);
CI's `SIMD_REQUIRE_OBJDUMP` enforces it.

## Testing strategy

Per the repo convention, every primitive ships:

1. A pure-Go reference (the source of truth).
2. Parity tests asserting the active SIMD path is bit-identical to the reference,
   across lengths that exercise full vector blocks, partial tails, and the
   sub-threshold (scalar) path. Include boundary values: `-128`, `127`, `0`, and
   `abs(-128)` overflow cases for saturating add/sub.
3. A `testing.AllocsPerRun == 0` allocation-free assertion per op.
4. Differential fuzzing of the SIMD path against the reference (`fuzz_test.go`).
5. `asmcheck_test.go` cross-checks every new `WORD` encoding (arm64asm directly,
   or objdump for SDOT).

Test execution:

- amd64: natively on this host (with and without `SIMD_DISABLE=avx2` to exercise
  the Go fallback).
- arm64: cross-compile here, run the test binary on `thakala@rpi5.local` (Cortex-
  A76, `asimddp` present, go1.26.1). Run with `SIMD_DISABLE=dotprod` too, to
  exercise the base-NEON DotProduct path on the same host.

## Docs

- `doc.go`: add the `i8` package to the subpackage list and the operation
  summary.
- `README.md`: add an `i8` section mirroring the `i16`/`i32` tables (operations,
  per-arch tiers, scope note explaining saturating arithmetic and int32-
  accumulated reductions, and that quantize/dequantize + movement are planned
  follow-ups).

## Out of scope (explicit follow-ups)

- `Quantize`/`Dequantize` (`float32 <-> int8` affine with scale + zero-point):
  needs a dedicated design for symmetric vs asymmetric and per-tensor vs per-
  channel conventions.
- AVX-512 VNNI (`VPDPBUSD`) DotProduct fast path (needs AVX512VNNI detection).
- AVX-512 BW wide paths for the saturating arithmetic and reductions.
- 8-bit movement (`Interleave2`/`Deinterleave2`), lower impact than the above.
- SSE2-tier saturating arithmetic (`PADDSB`/`PSUBSB`) for non-AVX2 amd64.
