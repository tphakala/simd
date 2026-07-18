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
VLD1/VLD1.P/VST1/VST1.P
```

`VADDV` and `VUMAX` are the two worth calling out: both look like they might need
encoding and both do not. Reach for a `WORD` only after confirming the mnemonic is
actually rejected.

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
