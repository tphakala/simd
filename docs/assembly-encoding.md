# Assembly encoding notes

Reference for the instruction-selection constraints that shape every hand-written
kernel in this repo. These are Go-toolchain facts, not micro-architecture tuning:
they explain *why* the integer NEON kernels are written as raw `WORD` encodings
and why some Plan 9 mnemonics differ from their Intel names. Verified empirically
against go1.26 (one instruction at a time in a minimal `TEXT` block); re-check on
a toolchain bump, since the assembler's accepted set can change between releases.

## ARM64: Go's assembler cannot express integer vector multiply

Go's arm64 assembler rejects every integer vector multiply and multiply-accumulate
with `unrecognized instruction`. Confirmed for, among others:

```
SMLAL, VSMLAL, VSMLAL2, VSMULL, VSMULL2, VUMULL, VMUL (int vector, .S4 and .H8),
SQRDMULH, VSQRDMULH, SQDMULH, VSQDMULH, SRSHR, VSRSHR, VSSHR, VXTN, VXTN2,
VABS, VSMAX, VSMIN, VSADDLP, VSADALP, VUMAXV, VSMAXV
```

The scalar `SMULL` with vector operands fails differently, with `illegal
combination`: the general-register form exists, the vector form does not.

**So NEON integer multiply-accumulate requires raw `WORD $0x...` encodings.** That
is why `i8` and `i16` look the way they do: the widening multiply-add at the core
of those kernels has no spelling the assembler accepts. Each `WORD` carries a
decoded mnemonic in its trailing comment, and the test suite cross-checks every
one against `golang.org/x/arch/arm64asm` (see the `asmcheck` package and
`asmcheck_test.go`).

### These ARE native, do not hand-encode them

```
VADD (.S4/.H8), VSHL, VADDV, VEXT, VDUP, VMOV, VEOR, VUMAX,
VLD1/VLD1.P/VST1/VST1.P (including the multi-register forms),
VZIP1/VZIP2/VUZP1/VUZP2 (any arrangement),
VFMLA/VFMLS (.S2/.S4/.D2 only, NOT the FP16 .H4/.H8)
```

This list says these instructions do not NEED a `WORD`. It is not an instruction
to convert the existing kernels, which hand-encode them deliberately; see the note
below the tables before changing any of them.

`VADDV` and `VUMAX` are the two worth calling out: both look like they might need
encoding and both do not. Reach for a `WORD` only after confirming the mnemonic is
actually rejected.

The float permutes and the fused multiply-adds are the other easy ones to miss.
Verified against go1.26.5 by assembling each with `go tool asm` and reading the
word back with `go tool objdump`: every one of the six is accepted on both `.D2`
and `.S4`, and each emits exactly the word the repo already hand-encodes for the
same operands. The `.D2` set, against `f64/f64_arm64.s`:

| Plan 9 source                 | Emitted word | Decodes as                    |
| ----------------------------- | ------------ | ----------------------------- |
| `VUZP1 V1.D2, V0.D2, V2.D2`   | `0x4EC11802` | `UZP1 V2.2D, V0.2D, V1.2D`    |
| `VUZP2 V1.D2, V0.D2, V3.D2`   | `0x4EC15803` | `UZP2 V3.2D, V0.2D, V1.2D`    |
| `VZIP1 V11.D2, V10.D2, V0.D2` | `0x4ECB3940` | `ZIP1 V0.2D, V10.2D, V11.2D`  |
| `VZIP2 V11.D2, V10.D2, V1.D2` | `0x4ECB7941` | `ZIP2 V1.2D, V10.2D, V11.2D`  |
| `VFMLS V5.D2, V3.D2, V6.D2`   | `0x4EE5CC66` | `FMLS V6.2D, V3.2D, V5.2D`    |
| `VFMLA V4.D2, V3.D2, V7.D2`   | `0x4E64CC67` | `FMLA V7.2D, V3.2D, V4.2D`    |

Note the operand order: Plan 9 writes `Vm, Vn, Vd` where ARM writes `Vd, Vn, Vm`.

The `.S4` forms behave the same way. Spot-checked against `f32/f32_arm64.s`:
`VZIP1 V1.S4, V0.S4, V2.S4` emits `0x4E813802`, `VUZP1 V1.S4, V0.S4, V2.S4`
emits `0x4E811802`, and `VFMLA V4.S4, V2.S4, V0.S4` emits `0x4E24CC40`, each
identical to the `WORD` that file already carries for those operands.

A grep for these mnemonics at the start of a line finds no use anywhere in the
tree: every occurrence is a `WORD`. That is consistency with the neighbouring
instructions in kernels that hand-encode anyway, not a toolchain constraint. A new
kernel with no other `WORD` in it should use the mnemonics.

One caveat if you do. `TestNoFMAContract` forbids a fused multiply-add in the
kernels listed in `singleRoundingKernels`, and it sees a `WORD` through its decoded
ARM spelling (`FMLA`) but a mnemonic through Go's Plan 9 spelling (`VFMLA`). Its
regex covers both, so either form is caught; do not narrow it.

An amusing asymmetry: `go tool objdump` happily *decodes* a hand-encoded word back
to `VSMLAL ...`. The toolchain knows the instruction on the way out but not on the
way in.

## AMD64: Plan 9 spellings that differ from Intel

- **`PMADDWD` does not assemble; the SSE2 form is spelled `PMADDWL`** ("long" for
  the 32-bit result). The VEX form keeps the Intel name, `VPMADDWD`.
- `PADDD`/`PADDL` and `PSRLDQ`/`PSRLO` are accepted as aliases.
- `PSHUFD`, `MOVWLSX`, `CMOVQLT` all assemble under their familiar names.

## Verifying a new WORD encoding without aarch64 binutils

`aarch64-linux-gnu-as` / `objdump` are not installed on every dev box. The working
substitute:

1. Write the GNU-syntax instruction in a `.s` file, assemble with
   `clang -c -arch arm64`, disassemble with LLVM `objdump -d`. That yields the
   authoritative encoding.
2. Cross-check the word through the repo's own `asmcheck.Verify(hex, comment)` and
   confirm `Status == Match`.

Worth knowing about `asmcheck_test.go`: it defers any directive whose comment
contains `.8H` to an objdump cross-check ONLY when `arm64asm` cannot decode it, and
without objdump those are then accepted UNCHECKED. Integer `SMLAL`/`SMLAL2` with
`.8H` operands decode fine and are checked directly, so a green run is meaningful
for them; `SDOT` genuinely needs the objdump path. If you add a `.8H` encoding,
confirm which side of that line it falls on rather than assuming the passing test
covered it.

To confirm quickly whether the assembler accepts a mnemonic at all, drop it into a
one-line `TEXT` block and `GOOS=linux GOARCH=arm64 go build` (or `GOARCH=amd64`):
an `unrecognized instruction` or `illegal combination` error is the answer.

## Do not trust an LLM's Plan 9 operand-order reasoning

Plan 9 reverses operand order from Intel, and reversed-convention reasoning is a
reliable way to get a confident, wrong answer. During the #143 review an LLM rated
a length clamp a critical out-of-bounds bug, arguing that Go's `CMPQ src, dst`
evaluates `dst - src` so `CMPQ DX, CX` computes `CX - DX`. It is the reverse. The
shipped sequence computed `min` correctly, and the suggested `CMOVQGT` "fix" would
have *introduced* the out-of-bounds read it warned about. Three independent checks
(a hardware probe returning the computed `n`, a disassembly showing
`cmp %rdx,%rcx` + `cmovl %rcx,%rdx`, and guard-page tests where any over-read
segfaults) all agreed the original was right.

Verify operand semantics by disassembly or execution, never by recalled
convention.
