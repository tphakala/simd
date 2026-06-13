# i8 AddScalarSaturate / SubScalarSaturate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `AddScalarSaturate(dst, a []int8, s int8)` and `SubScalarSaturate(dst, a []int8, s int8)` to `i8`: broadcast a scalar and apply signed saturating add/subtract. Completes the saturating-arithmetic family (the slice-vs-slice forms already exist as `AddSaturate`/`SubSaturate`).

**Architecture:** Same package structure. Reuses the saturating instructions already in the package (`VPADDSB`/`VPSUBSB`, `SQADD`/`SQSUB`) with the scalar broadcast pattern from `Clamp` (`VPBROADCASTB` on AVX2, `DUP` on NEON).

**Tech Stack:** Go 1.26, Plan9 assembly.

## Semantics

- `AddScalarSaturate(dst, a, s)`: `dst[i] = clamp(int(a[i]) + int(s), -128, 127)`, `n = min(len(dst),len(a))`. Trailing dst capacity untouched. Zero-alloc.
- `SubScalarSaturate(dst, a, s)`: `dst[i] = clamp(int(a[i]) - int(s), -128, 127)`.

## Verified encodings (cross-assembler; agy cross-checked)

- AVX2: `VPBROADCASTB s+48(FP), Y1`; `VPADDSB`/`VPSUBSB`.
- NEON: `DUP V3.16B, W5` = `0x4E010CA3`; `SQADD V2.16B, V0.16B, V3.16B` = `0x4E230C02`; `SQSUB V2.16B, V0.16B, V3.16B` = `0x4E232C02`.

## Tasks

- [ ] **Task 1 (TDD):** `i8_test.go` `TestAddScalarSaturate`/`TestSubScalarSaturate` (literal saturation cases + parity across `lengths`; extend zero-alloc + trailing-cap). RED. Then refs `addScalarSatGo`/`subScalarSatGo` in `i8_go.go` (reuse `clampI8`), API in `i8.go`, routing in `i8_other.go`. `SIMD_DISABLE=avx` test PASS. Commit.
- [ ] **Task 2:** amd64 dispatch (gate `len(dst) >= blockSat32`) + `addScalarSatAVX2`/`subScalarSatAVX2` kernels in `i8_amd64.s` (broadcast `s`, `VPADDSB`/`VPSUBSB`, scalar tail clamp). `go vet`, test PASS. Commit.
- [ ] **Task 3:** arm64 dispatch (gate `len(dst) >= minNEON16`) + NEON kernels (`DUP`, `SQADD`/`SQSUB` WORDs, CSEL clamp tail). asmcheck PASS; build/run on rpi5. Commit.
- [ ] **Task 4:** fuzz (`FuzzI8ScalarSaturate`) + benchmarks; lint; full repo test; race; docs (`doc.go`, README table+example, `i8.go` package doc). Commit; push; stacked PR (base `feat/i8-absolute-value`), referencing #132.

## Self-review

- `s` is `int8`; the broadcast uses its byte. Scalar tail sign-extends `s` once outside the loop. Saturation reuses the same widened-add/sub-then-clamp as `AddSaturate`/`SubSaturate`, so the references are trivially correct and bit-exact.
</content>
