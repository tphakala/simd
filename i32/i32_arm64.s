//go:build arm64

#include "textflag.h"

// int32 (de)interleave on ARM64 (NEON / ASIMD).
//
// ZIP1/ZIP2/UZP1/UZP2 on .4S operands and the VLD1/VST1 .S4 loads/stores all
// move 32-bit lanes by bit pattern, so these kernels are the int32 counterparts
// of interleave2NEON / deinterleave2NEON in ../f32/f32_arm64.s with identical
// vector bodies; only the scalar tails differ (integer MOVW instead of FMOVS).
// The ZIP/UZP instructions are hand-encoded as WORD because the Go assembler
// lacks mnemonics for them; the trailing comment is the decoded form and is
// cross-checked by asmcheck_test.go.

// func interleave2NEON(dst, a, b []int32)
// Interleaves: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // V0 = [a0, a1, a2, a3]
    VLD1.P 16(R2), [V1.S4]     // V1 = [b0, b1, b2, b3]
    WORD $0x4E813802           // ZIP1 V2.4S, V0.4S, V1.4S -> [a0, b0, a1, b1]
    WORD $0x4E817803           // ZIP2 V3.4S, V0.4S, V1.4S -> [a2, b2, a3, b3]
    VST1.P [V2.S4], 16(R0)     // Store [a0, b0, a1, b1]
    VST1.P [V3.S4], 16(R0)     // Store [a2, b2, a3, b3]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop4

interleave2_neon_remainder:
    AND $3, R3
    CBZ R3, interleave2_neon_done

interleave2_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    MOVW R5, (R0)
    MOVW R6, 4(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon_loop1

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []int32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop4:
    VLD1.P 16(R2), [V0.S4]     // V0 = [a0, b0, a1, b1]
    VLD1.P 16(R2), [V1.S4]     // V1 = [a2, b2, a3, b3]
    WORD $0x4E811802           // UZP1 V2.4S, V0.4S, V1.4S -> [a0, a1, a2, a3]
    WORD $0x4E815803           // UZP2 V3.4S, V0.4S, V1.4S -> [b0, b1, b2, b3]
    VST1.P [V2.S4], 16(R0)     // Store a
    VST1.P [V3.S4], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop4

deinterleave2_neon_remainder:
    AND $3, R3
    CBZ R3, deinterleave2_neon_done

deinterleave2_neon_loop1:
    MOVW (R2), R5
    MOVW 4(R2), R6
    MOVW R5, (R0)
    MOVW R6, (R1)
    ADD $8, R2
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon_loop1

deinterleave2_neon_done:
    RET

// Arithmetic, decorrelation and fixed-predictor kernels (NEON / ASIMD).
//
// These do integer ALU work on .4S vectors (4 int32/iter): ADD/SUB/SSHR/SHL and
// the AND/ORR/MOVI used for the mid/side parity bit. The Go assembler has no
// mnemonics for these vector ops, so they are hand-encoded as WORD with the
// decoded GNU form in the trailing comment (cross-checked by asmcheck_test.go).
// Each vector lane is 32 bits, so the SIMD path wraps exactly like the int32 Go
// reference; the scalar tails use the W-register (32-bit) ALU forms (ADDW/SUBW/
// ASRW/...) so they wrap identically. Dispatched from i32_arm64.go gated on the
// NEON CPU feature, with the pure-Go reference as the fallback.

// func addNEON(dst, a, b []int32)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, add_neon_remainder

add_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x4EA18402           // ADD V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, add_neon_loop4

add_neon_remainder:
    AND $3, R3
    CBZ R3, add_neon_done

add_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    ADDW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, add_neon_loop1

add_neon_done:
    RET

// func subNEON(dst, a, b []int32)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, sub_neon_remainder

sub_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x6EA18402           // SUB V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, sub_neon_loop4

sub_neon_remainder:
    AND $3, R3
    CBZ R3, sub_neon_done

sub_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    SUBW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, sub_neon_loop1

sub_neon_done:
    RET

// func minMaxNEON(res []int32) (minVal, maxVal int32)
// Signed int32 min and max over res in one pass. The dispatch gates len(res) >=
// 4, so at least one full 4-element (.4S) block exists: the min and max
// accumulators start from block 0 and fold the remaining full blocks with
// SMIN/SMAX, then SMINV/SMAXV reduce each accumulator across its 4 lanes to a
// scalar and a scalar tail folds the (n mod 4) remainder. SMIN/SMAX/SMINV/SMAXV
// have no Go assembler mnemonic, so they are hand-encoded WORD directives (the
// trailing comment is the decoded form, cross-checked by asmcheck_test.go).
// Every compare is signed, matching minMaxGo exactly.
TEXT ·minMaxNEON(SB), NOSPLIT, $0-32
    MOVD res_base+0(FP), R2
    MOVD res_len+8(FP), R3
    LSR  $2, R3, R4                  // R4 = full 4-element blocks (>=1)
    VLD1 (R2), [V0.S4]               // V0 = block 0 (min acc), no advance
    VLD1.P 16(R2), [V1.S4]           // V1 = block 0 (max acc), advance to block 1
    SUB  $1, R4                      // blocks remaining after block 0
    CBZ  R4, mm_neon_reduce          // single block: accumulators hold it; R2 at tail
mm_neon_loop:
    VLD1.P 16(R2), [V2.S4]           // load block + advance
    WORD $0x4EA26C00                 // SMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA26421                 // SMAX V1.4S, V1.4S, V2.4S
    SUB  $1, R4
    CBNZ R4, mm_neon_loop

mm_neon_reduce:
    WORD $0x4EB1A803                 // SMINV S3, V0.4S
    WORD $0x4EB0A824                 // SMAXV S4, V1.4S
    FMOVS F3, R5                     // R5 = running min (low 32 = int32)
    FMOVS F4, R6                     // R6 = running max (low 32 = int32)

    // scalar tail: (n mod 4) residuals (R2 already at &res[fullBlocks*4])
    AND  $3, R3, R4
    CBZ  R4, mm_neon_done
mm_neon_tail:
    MOVW.P 4(R2), R7                 // r (sign-extended; low 32 = int32)
    CMPW R5, R7                      // (R7 - R5), signed 32-bit
    CSEL LT, R7, R5, R5             // R5 = min(r, R5)
    CMPW R6, R7
    CSEL GT, R7, R6, R6             // R6 = max(r, R6)
    SUB  $1, R4
    CBNZ R4, mm_neon_tail

mm_neon_done:
    MOVW R5, minVal+24(FP)
    MOVW R6, maxVal+28(FP)
    RET
