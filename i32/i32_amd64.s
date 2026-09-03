//go:build amd64

#include "textflag.h"

// int32 (de)interleave on AMD64.
//
// Interleaving is pure 32-bit-lane data movement: no arithmetic happens, so the
// bit pattern of each lane is irrelevant and the AVX1 floating-point shuffles
// (VUNPCKLPS / VUNPCKHPS / VPERM2F128 / VSHUFPS / VUNPCKLPD / VUNPCKHPD) move
// int32 lanes exactly as they move float32 lanes. The chain is load -> shuffle
// -> store with no integer ALU op in between, so there is no FP/integer
// domain-crossing penalty. The scalar tails use the integer MOVL instead of the
// f32 VMOVSS. These kernels are the int32 counterparts of interleave2AVX /
// deinterleave2AVX in ../f32/f32_amd64.s and require only AVX (not AVX2).

// func interleave2AVX(dst, a, b []int32)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2AVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    // Process 8 pairs at a time (16 output elements)
    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   interleave2_avx_remainder

interleave2_avx_loop8:
    VMOVUPS (SI), Y0           // Y0 = [a0..a7]
    VMOVUPS (DI), Y1           // Y1 = [b0..b7]

    // Unpack within 128-bit lanes
    VUNPCKLPS Y1, Y0, Y2       // [a0,b0,a1,b1 | a4,b4,a5,b5]
    VUNPCKHPS Y1, Y0, Y3       // [a2,b2,a3,b3 | a6,b6,a7,b7]

    // Permute to get final order
    VPERM2F128 $0x20, Y3, Y2, Y4  // [a0,b0,a1,b1,a2,b2,a3,b3]
    VPERM2F128 $0x31, Y3, Y2, Y5  // [a4,b4,a5,b5,a6,b6,a7,b7]

    VMOVUPS Y4, (DX)
    VMOVUPS Y5, 32(DX)

    ADDQ $32, SI               // a += 8 * 4
    ADDQ $32, DI               // b += 8 * 4
    ADDQ $64, DX               // dst += 16 * 4
    DECQ AX
    JNZ  interleave2_avx_loop8

interleave2_avx_remainder:
    ANDQ $7, CX
    JZ   interleave2_avx_done

interleave2_avx_scalar:
    MOVL (SI), AX
    MOVL (DI), BX
    MOVL AX, (DX)
    MOVL BX, 4(DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  interleave2_avx_scalar

interleave2_avx_done:
    VZEROUPPER
    RET

// func deinterleave2AVX(a, b, src []int32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// 1. VSHUFPS $0x88 gathers the evens (a's), $0xDD gathers the odds (b's)
// 2. VPERM2F128 + VUNPCKLPD/VUNPCKHPD reorder the 64-bit chunks across lanes
TEXT ·deinterleave2AVX(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    // Process 8 pairs at a time
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   deinterleave2_avx_remainder

deinterleave2_avx_loop8:
    VMOVUPS (SI), Y0           // [a0,b0,a1,b1 | a2,b2,a3,b3]
    VMOVUPS 32(SI), Y1         // [a4,b4,a5,b5 | a6,b6,a7,b7]

    // Gather evens / odds within each 128-bit lane
    VSHUFPS $0x88, Y1, Y0, Y2  // [a0,a1,a4,a5 | a2,a3,a6,a7]
    VSHUFPS $0xDD, Y1, Y0, Y3  // [b0,b1,b4,b5 | b2,b3,b6,b7]

    // Reorder Y2 -> [a0,a1,a2,a3 | a4,a5,a6,a7]
    VPERM2F128 $0x01, Y2, Y2, Y4  // [a2,a3,a6,a7 | a0,a1,a4,a5]
    VUNPCKLPD Y4, Y2, Y5
    VUNPCKHPD Y4, Y2, Y6
    VPERM2F128 $0x20, Y6, Y5, Y7  // a's in order

    // Reorder Y3 -> [b0,b1,b2,b3 | b4,b5,b6,b7]
    VPERM2F128 $0x01, Y3, Y3, Y4
    VUNPCKLPD Y4, Y3, Y5
    VUNPCKHPD Y4, Y3, Y6
    VPERM2F128 $0x20, Y6, Y5, Y4  // b's in order

    VMOVUPS Y7, (DX)           // store a
    VMOVUPS Y4, (R8)           // store b

    ADDQ $64, SI               // src += 16 * 4
    ADDQ $32, DX               // a += 8 * 4
    ADDQ $32, R8               // b += 8 * 4
    DECQ AX
    JNZ  deinterleave2_avx_loop8

deinterleave2_avx_remainder:
    ANDQ $7, CX
    JZ   deinterleave2_avx_done

deinterleave2_avx_scalar:
    MOVL (SI), AX
    MOVL 4(SI), BX
    MOVL AX, (DX)
    MOVL BX, (R8)
    ADDQ $8, SI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  deinterleave2_avx_scalar

deinterleave2_avx_done:
    VZEROUPPER
    RET

// Arithmetic, decorrelation and fixed-predictor kernels.
//
// Unlike the (de)interleave kernels above (pure 32-bit-lane data movement on
// AVX1 float shuffles), these do integer ALU work on 256-bit lanes, which
// requires AVX2 (VPADDD/VPSUBD/VPSRAD/VPSLLD/VPAND/VPOR). They are dispatched
// from i32_amd64.go gated on the AVX2 CPU feature, with the pure-Go reference as
// the fallback. Each processes 8 int32 (one YMM) per iteration with a scalar
// tail. The Go assembler's 3-operand order is dst-last: VPSUBD a, b, c is
// c = b - a.

// func addAVX2(dst, a, b []int32)
TEXT ·addAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add_remainder

add_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPADDD  Y1, Y0, Y2         // Y2 = a + b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_loop8

add_remainder:
    ANDQ $7, CX
    JZ   add_done

add_scalar:
    MOVL (SI), AX
    ADDL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add_scalar

add_done:
    VZEROUPPER
    RET

// func subAVX2(dst, a, b []int32)
TEXT ·subAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub_remainder

sub_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPSUBD  Y1, Y0, Y2         // Y2 = a - b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_loop8

sub_remainder:
    ANDQ $7, CX
    JZ   sub_done

sub_scalar:
    MOVL (SI), AX
    SUBL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub_scalar

sub_done:
    VZEROUPPER
    RET

// func minMaxAVX2(res []int32) (minVal, maxVal int32)
// Signed int32 min and max over res in one pass. The dispatch gates len(res) >=
// 8, so at least one full 8-element block exists: the min and max accumulators
// start from block 0 and fold the remaining full blocks with VPMINSD/VPMAXSD,
// then a horizontal reduce collapses the 8 lanes to a scalar and a scalar tail
// folds the (n mod 8) remainder. Every compare is signed (VPMINSD/VPMAXSD and
// the CMPL tail), so the int32 sign bit is honored exactly like minMaxGo.
TEXT ·minMaxAVX2(SB), NOSPLIT, $0-32
    MOVQ res_base+0(FP), R8
    MOVQ res_len+8(FP), CX           // n (>=8)
    MOVQ CX, R9
    SHRQ $3, R9                      // R9 = full 8-element blocks (>=1)
    VMOVDQU (R8), Y0                 // min acc = block 0
    VMOVDQU (R8), Y1                 // max acc = block 0
    MOVQ R9, AX                      // block counter
    DECQ AX                          // blocks remaining after block 0
    JZ   mm_reduce                   // single block: accumulators already hold it
    LEAQ 32(R8), SI                  // working res ptr at block 1
mm_loop:
    VMOVDQU (SI), Y2
    VPMINSD Y2, Y0, Y0               // Y0 = lanewise min(Y2, Y0)
    VPMAXSD Y2, Y1, Y1               // Y1 = lanewise max(Y2, Y1)
    ADDQ $32, SI
    DECQ AX
    JNZ  mm_loop

mm_reduce:

    // horizontal min of Y0 -> AX (low 32 bits), max of Y1 -> DX (low 32 bits)
    VEXTRACTI128 $1, Y0, X3
    VPMINSD X3, X0, X0               // fold lanes 4..7 into 0..3
    VPSHUFD $0x4E, X0, X3           // swap 64-bit halves
    VPMINSD X3, X0, X0
    VPSHUFD $0xB1, X0, X3           // swap 32-bit within pairs
    VPMINSD X3, X0, X0              // lane0 = min over all 8 lanes
    MOVQ X0, AX                      // EAX = running min

    VEXTRACTI128 $1, Y1, X3
    VPMAXSD X3, X1, X1
    VPSHUFD $0x4E, X1, X3
    VPMAXSD X3, X1, X1
    VPSHUFD $0xB1, X1, X3
    VPMAXSD X3, X1, X1
    MOVQ X1, DX                      // EDX = running max

    // scalar tail: (n mod 8) residuals at &res[fullBlocks*8]
    MOVQ CX, BX
    ANDQ $7, BX
    JZ   mm_done
    MOVQ R9, R10
    SHLQ $5, R10                     // fullBlocks * 32 bytes
    ADDQ R8, R10                     // tail ptr
mm_tail:
    MOVL (R10), R11                  // r (32-bit)
    CMPL R11, AX
    JGE  mm_tail_max
    MOVL R11, AX                     // r < min -> new min
mm_tail_max:
    CMPL R11, DX
    JLE  mm_tail_next
    MOVL R11, DX                     // r > max -> new max
mm_tail_next:
    ADDQ $4, R10
    DECQ BX
    JNZ  mm_tail

mm_done:
    MOVL AX, minVal+24(FP)
    MOVL DX, maxVal+28(FP)
    VZEROUPPER
    RET

// Tier-3 reduction and element-wise kernels (AVX2).
//
// Both receive pre-clamped slices from the public API, so a_len (or dst_len)
// is the trusted element count and no in-assembly clamp exists.

// func sumAVX2(a []int32) int32
// Wrapping int32 sum: VPADDD folds 8-lane blocks into a vector accumulator, a
// horizontal add reduces it, and a 32-bit scalar tail adds the (n mod 8)
// remainder. Every add wraps in a 32-bit lane, and wrapping addition is
// associative, so the lane split and reduction order are bit-identical to
// sumGo for every input, including forced overflow.
TEXT ·sumAVX2(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y0, Y0, Y0           // int32 accumulator = 0

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   sum_avx2_reduce

sum_avx2_loop8:
    VMOVDQU (SI), Y1
    VPADDD Y1, Y0, Y0          // accumulate (wrapping)
    ADDQ $32, SI
    DECQ AX
    JNZ  sum_avx2_loop8

sum_avx2_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPADDD X1, X0, X0          // fold 8 -> 4 int32
    VPSHUFD $0x4E, X0, X1      // swap 64-bit halves
    VPADDD X1, X0, X0
    VPSHUFD $0xB1, X0, X1      // swap 32-bit within pairs
    VPADDD X1, X0, X0
    MOVQ X0, AX                // low int32 = vector total (in EAX)

    ANDQ $7, CX
    JZ   sum_avx2_done

sum_avx2_scalar:
    MOVL (SI), BX
    ADDL BX, AX                // 32-bit add: wraps like sumGo
    ADDQ $4, SI
    DECQ CX
    JNZ  sum_avx2_scalar

sum_avx2_done:
    MOVL AX, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX2(dst, a []int32)
// Wrapping absolute value, 8 lanes per iteration: VPABSD wraps at the type
// minimum (abs(MinInt32) = MinInt32 in a 32-bit lane), which is absGo's
// contract. The scalar tail negates only the negative elements; NEGL of
// 0x80000000 wraps to itself in 32-bit, so the tail needs no special case.
TEXT ·absAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   abs_avx2_tail

abs_avx2_loop8:
    VPABSD (SI), Y0
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_avx2_loop8

abs_avx2_tail:
    ANDQ $7, CX
    JZ   abs_avx2_done

abs_avx2_scalar:
    MOVL (SI), AX
    TESTL AX, AX
    JGE  abs_avx2_store
    NEGL AX                    // |v|; NEGL of MinInt32 wraps to itself
abs_avx2_store:
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs_avx2_scalar

abs_avx2_done:
    VZEROUPPER
    RET

// func negWhereNegAVX2(dst, mag []int32, sign []float32)
// Branchless conditional negate, 8 lanes per iteration. VPSRAD $31 on the raw
// float32 sign lanes broadcasts each sign bit to a full-width int32 mask m
// (all-ones iff the sign bit is set, so -0.0/-Inf/-NaN negate); VPXOR then
// VPSUBD apply (mag ^ m) - m, which is -mag when m = -1 (MinInt32 wraps to
// itself) and mag when m = 0. Bit-identical to negWhereNegGo. The scalar tail
// does the same on 32-bit GPRs with SARL/XORL/SUBL, where NEGL-equivalent wrap
// of MinInt32 stays MinInt32. Frame is 3 slice headers: dst+0, mag+24, sign+48.
TEXT ·negWhereNegAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ mag_base+24(FP), SI
    MOVQ sign_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   negwhereneg_avx2_tail

negwhereneg_avx2_loop8:
    VMOVDQU (DI), Y1           // sign lanes (raw float32 bits)
    VPSRAD  $31, Y1, Y1        // Y1 = sign >> 31 = mask m
    VMOVDQU (SI), Y0           // mag
    VPXOR   Y1, Y0, Y0         // mag ^ m
    VPSUBD  Y1, Y0, Y0         // (mag ^ m) - m
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  negwhereneg_avx2_loop8

negwhereneg_avx2_tail:
    ANDQ $7, CX
    JZ   negwhereneg_avx2_done

negwhereneg_avx2_scalar:
    MOVL (DI), BX              // sign bits
    SARL $31, BX               // BX = mask m
    MOVL (SI), AX              // mag
    XORL BX, AX                // mag ^ m
    SUBL BX, AX                // (mag ^ m) - m
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  negwhereneg_avx2_scalar

negwhereneg_avx2_done:
    VZEROUPPER
    RET

// Fixed-point scale-by-scalar kernels (AVX2).
//
// AVX2 has VPMULDQ (signed 32x32 -> 64) but no 64-bit ARITHMETIC shift (VPSRAQ is
// AVX-512). The kernels lean on one fact: for a 64-bit product p with |p| <= 2^62
// and result = int32(p >> s), the LOW 32 bits of a LOGICAL shift (VPSRLQ $s) equal
// the low 32 bits of the arithmetic shift. Result bit i (0..31) reads p bit (i+s),
// and i+s <= 31+31 = 62 <= 63 is always a real bit of p for both shift kinds; the
// two differ only in bits >= 32 (sign-fill vs zero-fill), which the int32() cast
// discards. So VPSRLQ then keep the low 32 bits is exact, including the
// a=k=MinInt32 wrap (2^62 >> 31 = 2^31 -> MinInt32). No EVEX is emitted: VPMULDQ,
// VPSRLQ, VPSLLQ and VPBLENDD are all AVX2, and k reaches the vector via VMOVD to
// an xmm then the AVX2 xmm->ymm VPBROADCASTD (VEX). The GP-register-source
// VPBROADCASTD is AVX-512-only and would SIGILL on an AVX2-only core, so it is
// avoided; k is sign-extended to int32 first (int32 for Q31, int16 for Q15) since
// VPMULDQ is a signed 32x32 multiply.
//
// Per 8 int32: VPMULDQ folds the even 32-bit lanes (0,2,4,6) into 4 int64
// products; VPSRLQ $32 slides the odd lanes (1,3,5,7) into even positions for a
// second VPMULDQ; each product set is VPSRLQ $s so its low 32 bits hold the
// result lane; VPSLLQ $32 lifts the odd results to positions 1,3,5,7; and
// VPBLENDD $0xAA merges even (mask bit 0 -> second source) with odd (mask bit 1 ->
// first source). The Go operand order is VPBLENDD $imm, YfromMask1, YfromMask0,
// Ydst, so the shifted-odd register is listed FIRST. The scalar tail does the same
// in 64-bit GPRs (MOVLQSX, IMULQ, SARQ $s, MOVL store). dst may alias a exactly:
// each block/lane reads a before it stores dst. Frame is two slice headers plus
// the scalar k: dst+0, a+24, k+48.

// func scaleQ31AVX2(dst, a []int32, k int32)
TEXT ·scaleQ31AVX2(SB), NOSPLIT, $0-52
    MOVQ    dst_base+0(FP), DX
    MOVQ    dst_len+8(FP), CX
    MOVQ    a_base+24(FP), SI
    MOVLQSX k+48(FP), BX          // BX = int64(k) for the scalar tail
    VMOVD   BX, X3                // low 32 bits = int32(k)
    VPBROADCASTD X3, Y3           // AVX2 xmm->ymm broadcast: k in all 8 int32 lanes

    MOVQ CX, AX
    SHRQ $3, AX                   // AX = n / 8
    JZ   scaleq31_avx2_tail

scaleq31_avx2_loop8:
    VMOVDQU  (SI), Y0             // a[i..i+7]
    VPMULDQ  Y3, Y0, Y4           // even-lane products a0,a2,a4,a6 * k (4x int64)
    VPSRLQ   $32, Y0, Y1          // slide odd lanes a1,a3,a5,a7 into even positions
    VPMULDQ  Y3, Y1, Y5           // odd-lane products a1,a3,a5,a7 * k (4x int64)
    VPSRLQ   $31, Y4, Y4          // low 32 of each int64 = even result lanes
    VPSRLQ   $31, Y5, Y5          // low 32 of each int64 = odd result lanes
    VPSLLQ   $32, Y5, Y5          // lift odd results to 32-bit positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y2    // lanes 1,3,5,7 <- Y5 (odd), 0,2,4,6 <- Y4 (even)
    VMOVDQU  Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scaleq31_avx2_loop8

scaleq31_avx2_tail:
    ANDQ $7, CX
    JZ   scaleq31_avx2_done

scaleq31_avx2_scalar:
    MOVLQSX (SI), AX              // int64(a[i])
    IMULQ   BX, AX                // a[i] * k (64-bit, no overflow: |p| <= 2^62)
    SARQ    $31, AX               // arithmetic shift right 31
    MOVL    AX, (DX)              // low 32 bits: wraps like int32()
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  scaleq31_avx2_scalar

scaleq31_avx2_done:
    VZEROUPPER
    RET

// func scaleQ15AVX2(dst, a []int32, k int16)
// Identical recombine to scaleQ31AVX2 with a shift of 15. k is a signed int16, so
// it is sign-extended to int32 before the broadcast (VPMULDQ is signed 32x32) and
// to int64 for the tail; |k * a[i]| <= 2^46, well inside the int64 product.
TEXT ·scaleQ15AVX2(SB), NOSPLIT, $0-50
    MOVQ    dst_base+0(FP), DX
    MOVQ    dst_len+8(FP), CX
    MOVQ    a_base+24(FP), SI
    MOVWQSX k+48(FP), BX          // BX = int64(k), sign-extended int16 (also tail k)
    VMOVD   BX, X3                // low 32 bits = int32(k)
    VPBROADCASTD X3, Y3           // AVX2 xmm->ymm broadcast of int32(k)

    MOVQ CX, AX
    SHRQ $3, AX                   // AX = n / 8
    JZ   scaleq15_avx2_tail

scaleq15_avx2_loop8:
    VMOVDQU  (SI), Y0             // a[i..i+7]
    VPMULDQ  Y3, Y0, Y4           // even-lane products a0,a2,a4,a6 * k (4x int64)
    VPSRLQ   $32, Y0, Y1          // slide odd lanes a1,a3,a5,a7 into even positions
    VPMULDQ  Y3, Y1, Y5           // odd-lane products a1,a3,a5,a7 * k (4x int64)
    VPSRLQ   $15, Y4, Y4          // low 32 of each int64 = even result lanes
    VPSRLQ   $15, Y5, Y5          // low 32 of each int64 = odd result lanes
    VPSLLQ   $32, Y5, Y5          // lift odd results to 32-bit positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y2    // lanes 1,3,5,7 <- Y5 (odd), 0,2,4,6 <- Y4 (even)
    VMOVDQU  Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scaleq15_avx2_loop8

scaleq15_avx2_tail:
    ANDQ $7, CX
    JZ   scaleq15_avx2_done

scaleq15_avx2_scalar:
    MOVLQSX (SI), AX              // int64(a[i])
    IMULQ   BX, AX                // k * a[i] (64-bit, |p| <= 2^46)
    SARQ    $15, AX               // arithmetic shift right 15
    MOVL    AX, (DX)              // low 32 bits: wraps like int32()
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  scaleq15_avx2_scalar

scaleq15_avx2_done:
    VZEROUPPER
    RET

// func gainQ31AVX2(dst, a []int32, gain int32, preShift, postShift int)
// Fused Q31 gain: dst[i] = PSHR32(MULT32_32_Q31(SHL32(a[i], preShift), gain), postShift).
// The MULT32_32_Q31 core is scaleQ31AVX2 verbatim (VPMULDQ even/odd, VPSRLQ $31,
// VPBLENDD recombine); this kernel wraps it in a pre-shift and a rounding post-shift:
//   - VPSLLD by the runtime preShift count (an XMM operand) applies SHL32 to the
//     8 int32 lanes BEFORE the widen, so the shifted value stays in int32 range and
//     |x*gain| <= 2^62. The VPSRLQ $31 low-32 trick therefore still holds exactly as
//     in scaleQ31AVX2 (bit i+31 <= 62 is a real product bit), and the odd-lane
//     VPSRLQ $32 slide reads the already-shifted Y0.
//   - VPADDD adds the broadcast rounding bias (int32(1)<<postShift)>>1, wrapping in
//     int32 lanes exactly like libopus PSHR32; bias is 0 when postShift == 0.
//   - VPSRAD by the runtime postShift count (an XMM operand) is the arithmetic
//     (sign-preserving, truncating) right shift; the bias already gave round-half-up.
// bias is formed in a 32-bit GPR (SHLL then arithmetic SARL $1) so postShift == 31
// sign-extends to 0xC0000000, matching (int32(1)<<31)>>1. The scalar tail SHLL/
// MOVLQSX (sign-extend AFTER the wrapping shift), IMULQ, SARQ $31, ADDL bias,
// SARL to close out lengths that are not a multiple of 8. Registers: DX/SI/DI the
// slices, BX=int64(gain), R8=preShift, R9=postShift, R10=bias, CX stages the CL
// shift counts, Y3=gain, Y5=bias, X6/X7 the vector shift counts; no R14/R15/BP.
// preShift and postShift must be in [0, 31]. Frame: dst+0, a+24, gain+48, preShift+56,
// postShift+64.
TEXT ·gainQ31AVX2(SB), NOSPLIT, $0-72
    MOVQ    dst_base+0(FP), DX
    MOVQ    a_base+24(FP), SI
    MOVLQSX gain+48(FP), BX          // BX = int64(gain) for the scalar tail
    MOVQ    preShift+56(FP), R8
    MOVQ    postShift+64(FP), R9

    MOVL    $1, R10                  // bias = (int32(1)<<postShift)>>1, in a 32-bit GPR
    MOVQ    R9, CX
    SHLL    CX, R10                  // 1 << postShift (wraps to 0x80000000 at 31)
    SARL    $1, R10                  // arithmetic >>1: 0xC0000000 at 31, else 1<<(postShift-1)

    VMOVD   BX, X3
    VPBROADCASTD X3, Y3              // gain in all 8 int32 lanes
    MOVQ    R8, X6                   // preShift count (VPSLLD, low 64 bits)
    MOVQ    R9, X7                   // postShift count (VPSRAD, low 64 bits)
    VMOVD   R10, X5
    VPBROADCASTD X5, Y5              // bias in all 8 int32 lanes

    MOVQ    dst_len+8(FP), CX
    MOVQ    CX, DI                   // DI = n (for the tail)
    SHRQ    $3, CX                   // CX = n / 8
    JZ      gainq31_avx2_tail

gainq31_avx2_loop8:
    VMOVDQU  (SI), Y0                // a[i..i+7]
    VPSLLD   X6, Y0, Y0             // SHL32: int32 lanes << preShift (wrapping)
    VPMULDQ  Y3, Y0, Y4            // even-lane products x0,x2,x4,x6 * gain (4x int64)
    VPSRLQ   $32, Y0, Y1           // slide shifted odd lanes into even positions
    VPMULDQ  Y3, Y1, Y2           // odd-lane products x1,x3,x5,x7 * gain (4x int64)
    VPSRLQ   $31, Y4, Y4          // low 32 = even MULT32_32_Q31 lanes
    VPSRLQ   $31, Y2, Y2          // low 32 = odd MULT32_32_Q31 lanes
    VPSLLQ   $32, Y2, Y2          // lift odd results to positions 1,3,5,7
    VPBLENDD $0xAA, Y2, Y4, Y4    // recombine even/odd int32 lanes
    VPADDD   Y5, Y4, Y4           // + bias (wrapping int32 lanes)
    VPSRAD   X7, Y4, Y4           // arithmetic >> postShift per int32 lane
    VMOVDQU  Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ CX
    JNZ  gainq31_avx2_loop8

gainq31_avx2_tail:
    ANDQ $7, DI
    JZ   gainq31_avx2_done

gainq31_avx2_scalar:
    MOVL    (SI), AX                 // a[i]
    MOVQ    R8, CX
    SHLL    CX, AX                   // SHL32: wraps in 32 bits (count in CL)
    MOVLQSX AX, AX                   // sign-extend the WRAPPED x to int64
    IMULQ   BX, AX                   // x * gain (|p| <= 2^62)
    SARQ    $31, AX                  // MULT32_32_Q31
    ADDL    R10, AX                  // + bias, 32-bit wrapping add
    MOVQ    R9, CX
    SARL    CX, AX                   // arithmetic >> postShift in 32 bits
    MOVL    AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ DI
    JNZ  gainq31_avx2_scalar

gainq31_avx2_done:
    VZEROUPPER
    RET

// func butterflyAVX2(lo, hi []int32)
// In-place radix-2 butterfly, 8 int32 per iteration: each block loads lo and hi,
// forms the wrapping sum (VPADDD) and difference (VPSUBD) into fresh registers
// BEFORE storing either, then writes lo = lo+hi and hi = lo-hi. Reading the whole
// block of both inputs before either store is what makes the in-place update safe
// and is also why lo and hi must not overlap. The scalar tail does the same on
// 32-bit GPRs, computing the sum in R8 while AX still holds lo so SUBL forms the
// difference from the unmodified lo. Every add/sub wraps in a 32-bit lane, matching
// butterflyGo exactly. Frame is two slice headers: lo+0, hi+24.
TEXT ·butterflyAVX2(SB), NOSPLIT, $0-48
    MOVQ lo_base+0(FP), SI
    MOVQ lo_len+8(FP), CX
    MOVQ hi_base+24(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   butterfly_remainder

butterfly_loop8:
    VMOVDQU (SI), Y0          // lo block
    VMOVDQU (DI), Y1          // hi block
    VPADDD  Y1, Y0, Y2        // Y2 = lo + hi
    VPSUBD  Y1, Y0, Y3        // Y3 = lo - hi
    VMOVDQU Y2, (SI)          // lo = lo + hi
    VMOVDQU Y3, (DI)          // hi = lo - hi
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  butterfly_loop8

butterfly_remainder:
    ANDQ $7, CX
    JZ   butterfly_done

butterfly_scalar:
    MOVL (SI), AX             // lo
    MOVL (DI), BX             // hi
    MOVL AX, R8
    ADDL BX, R8              // R8 = lo + hi (wraps in 32-bit)
    SUBL BX, AX              // AX = lo - hi (AX still holds lo)
    MOVL R8, (SI)            // lo = lo + hi
    MOVL AX, (DI)            // hi = lo - hi
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  butterfly_scalar

butterfly_done:
    VZEROUPPER
    RET

// func maxAbsAVX2(a []int32) int32
// Peak magnitude with celtMaxabs32 semantics: the same signed int32 min/max
// reduction as minMaxAVX2 (VMOVDQU block 0 into both accumulators, VPMINSD/
// VPMAXSD fold, horizontal reduce to a scalar min in AX and max in DX, CMPL
// scalar tail), then a combine to max(maxVal, -minVal). NEGL forms -minVal with
// the two's-complement wrap (-MinInt32 == MinInt32), and the signed CMPL/JGE
// picks the larger of maxVal and -minVal. The dispatch gates len(a) >= 8, so at
// least one full 8-element block exists; every compare is signed, matching
// maxAbsGo exactly. Frame is one slice header plus the int32 return: a+0, ret+24.
TEXT ·maxAbsAVX2(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), R8
    MOVQ a_len+8(FP), CX             // n (>=8)
    MOVQ CX, R9
    SHRQ $3, R9                      // R9 = full 8-element blocks (>=1)
    VMOVDQU (R8), Y0                 // min acc = block 0
    VMOVDQU (R8), Y1                 // max acc = block 0
    MOVQ R9, AX                      // block counter
    DECQ AX                          // blocks remaining after block 0
    JZ   maxabs_reduce               // single block: accumulators already hold it
    LEAQ 32(R8), SI                  // working ptr at block 1
maxabs_loop:
    VMOVDQU (SI), Y2
    VPMINSD Y2, Y0, Y0               // Y0 = lanewise min(Y2, Y0)
    VPMAXSD Y2, Y1, Y1               // Y1 = lanewise max(Y2, Y1)
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs_loop

maxabs_reduce:

    // horizontal min of Y0 -> AX (low 32 bits), max of Y1 -> DX (low 32 bits)
    VEXTRACTI128 $1, Y0, X3
    VPMINSD X3, X0, X0               // fold lanes 4..7 into 0..3
    VPSHUFD $0x4E, X0, X3            // swap 64-bit halves
    VPMINSD X3, X0, X0
    VPSHUFD $0xB1, X0, X3            // swap 32-bit within pairs
    VPMINSD X3, X0, X0               // lane0 = min over all 8 lanes
    MOVQ X0, AX                      // EAX = running min

    VEXTRACTI128 $1, Y1, X3
    VPMAXSD X3, X1, X1
    VPSHUFD $0x4E, X1, X3
    VPMAXSD X3, X1, X1
    VPSHUFD $0xB1, X1, X3
    VPMAXSD X3, X1, X1
    MOVQ X1, DX                      // EDX = running max

    // scalar tail: (n mod 8) residuals at &a[fullBlocks*8]
    MOVQ CX, BX
    ANDQ $7, BX
    JZ   maxabs_combine
    MOVQ R9, R10
    SHLQ $5, R10                     // fullBlocks * 32 bytes
    ADDQ R8, R10                     // tail ptr
maxabs_tail:
    MOVL (R10), R11                  // r (32-bit)
    CMPL R11, AX
    JGE  maxabs_tail_max
    MOVL R11, AX                     // r < min -> new min
maxabs_tail_max:
    CMPL R11, DX
    JLE  maxabs_tail_next
    MOVL R11, DX                     // r > max -> new max
maxabs_tail_next:
    ADDQ $4, R10
    DECQ BX
    JNZ  maxabs_tail

maxabs_combine:
    NEGL AX                          // AX = -min (wrapping; -MinInt32 == MinInt32)
    CMPL DX, AX                      // signed max vs -min
    JGE  maxabs_ret_max              // max >= -min: result = max (DX)
    MOVL AX, DX                      // else result = -min
maxabs_ret_max:
    MOVL DX, ret+24(FP)
    VZEROUPPER
    RET

// func sumSqShiftedQ31AVX2(a []int32, shift int) int32
// Wrapping int32 sum of pre-shifted Q31 squares, 8 int32 per iteration. Each block
// left-shifts the eight int32 lanes by the runtime shift count (VPSLLD with an XMM
// count applies SHL32 in 32-bit lanes, so the value stays in int32 range BEFORE the
// widen), then squares them with the scaleQ31AVX2 recombine: VPMULDQ gives the four
// even-lane squares x0,x2,x4,x6 as int64, VPSRLQ $32 slides the already-shifted odd
// lanes down for a second VPMULDQ, each product is shifted right 31 (VPSRLQ $31 --
// the square is non-negative so bit 63 is clear and the logical shift equals the
// arithmetic one, low 32 = the MULT32_32_Q31 term), and VPSLLQ $32 + VPBLENDD $0xAA
// reassemble the eight int32 terms. VPADDD accumulates them into an 8-lane int32
// accumulator (wrapping). After the loop the accumulator folds to a single int32
// exactly like sumAVX2 (VEXTRACTI128 + VPADDD + two VPSHUFD folds), and the scalar
// tail closes out the (n mod 8) residuals: SHLL wraps x in 32 bits, MOVLQSX sign-
// extends it, IMULQ forms x*x (|x| <= 2^31 so |p| <= 2^62 fits int64), SARQ $31
// takes the term, and ADDL accumulates it wrapping. Because wrapping int32 addition
// is associative, the 8-lane split plus the fold is bit-identical to
// sumSqShiftedQ31Go for every input. The dispatch guarantees shift in [0,31], so
// VPSLLD (saturates to 0 past 31) and SHLL (masks the count to 5 bits) never
// disagree on an out-of-range count. Registers SI/DI/CX/AX/DX/R8 and
// X6/Y0/Y1/Y2/Y4/Y5/Y7; no R14/R15/BP. Frame: a+0, shift+24, ret+32.
TEXT ·sumSqShiftedQ31AVX2(SB), NOSPLIT, $0-36
    MOVQ a_base+0(FP), SI
    MOVQ shift+24(FP), R8
    VMOVD R8, X6                  // VPSLLD count = shift; VEX move (not legacy MOVQ) so a
                                  // caller with dirty upper YMM state pays no AVX-SSE assist
    VPXOR Y7, Y7, Y7              // wrapping int32 accumulator = 0

    MOVQ a_len+8(FP), DI          // DI = n (also the tail count)
    MOVQ DI, CX
    SHRQ $3, CX                   // CX = n / 8
    JZ   sumsq_avx2_reduce

sumsq_avx2_loop8:
    VMOVDQU  (SI), Y0             // a[i..i+7]
    VPSLLD   X6, Y0, Y0           // x = a << shift (wrapping in 32-bit lanes)
    VPMULDQ  Y0, Y0, Y4          // even lanes x0,x2,x4,x6 squared -> 4x int64
    VPSRLQ   $32, Y0, Y1          // slide shifted odd lanes into even positions
    VPMULDQ  Y1, Y1, Y5         // odd lanes x1,x3,x5,x7 squared -> 4x int64
    VPSRLQ   $31, Y4, Y4         // low 32 of each int64 = even Q31 terms
    VPSRLQ   $31, Y5, Y5         // low 32 of each int64 = odd Q31 terms
    VPSLLQ   $32, Y5, Y5         // lift odd terms to int32 positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y2   // 8 int32 terms: 0,2,4,6 <- Y4, 1,3,5,7 <- Y5
    VPADDD   Y2, Y7, Y7          // accumulate (wrapping int32 lanes)
    ADDQ $32, SI
    DECQ CX
    JNZ  sumsq_avx2_loop8

sumsq_avx2_reduce:
    VEXTRACTI128 $1, Y7, X1
    VPADDD X1, X7, X7            // fold 8 -> 4 int32
    VPSHUFD $0x4E, X7, X1        // swap 64-bit halves
    VPADDD X1, X7, X7
    VPSHUFD $0xB1, X7, X1        // swap 32-bit within pairs
    VPADDD X1, X7, X7
    MOVQ X7, DX                  // EDX = vector total (low int32)

    ANDQ $7, DI
    JZ   sumsq_avx2_done

sumsq_avx2_scalar:
    MOVL    (SI), AX
    MOVQ    R8, CX
    SHLL    CX, AX              // x = a << shift (wraps in 32 bits, count in CL)
    MOVLQSX AX, AX              // sign-extend the WRAPPED x to int64
    IMULQ   AX, AX              // x * x (|x| <= 2^31 so |p| <= 2^62)
    SARQ    $31, AX             // MULT32_32_Q31 (square >= 0, arithmetic == logical)
    ADDL    AX, DX              // + term, wrapping int32 add
    ADDQ $4, SI
    DECQ DI
    JNZ  sumsq_avx2_scalar

sumsq_avx2_done:
    MOVL DX, ret+32(FP)
    VZEROUPPER
    RET

// func firValidQ15AVX2(dst, x []int32, taps []int16)
// int32 valid convolution in correlation orientation with int16 Q15 taps,
// vectorized over the OUTPUT index: 8 outputs per iteration. For output block i
// the accumulator Y2 starts at zero and, for each tap j, loads the sliding window
// x[i+j .. i+j+7] (one unaligned VMOVDQU that supplies the tap-j contribution for
// all 8 outputs at once, since output i+k reads lane k of the window), broadcasts
// taps[j] sign-extended to int32, forms the 8 Q15-TRUNCATED products with the
// exact scaleQ15AVX2 recombine (VPMULDQ even + VPSRLQ $32 slide + VPMULDQ odd,
// each VPSRLQ $15 to keep the low 32 = arithmetic-shift result, VPSLLQ $32 +
// VPBLENDD $0xAA to reassemble), and VPADDD-accumulates them (wrapping int32). The
// window pointer BX slides by 4 bytes per tap; the taps pointer R10 by 2. After
// all taps the 8 outputs are stored. The scalar-output tail runs the full inner
// tap loop per remaining output (MOVLQSX/MOVWQSX, IMULQ, SARQ $15, ADDL), so the
// per-product truncation is preserved and the result is bit-exact with
// firValidQ15Go. The dispatch gates n = len(dst) >= 8, and the kernel reads x only
// up to index n-1+len(taps)-1 <= len(x)-1, so there is no over-read. dst must not
// overlap x. Frame is dst+x+taps slice headers: dst+0, x+24, taps+48.
TEXT ·firValidQ15AVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX        // n = number of outputs
    MOVQ x_base+24(FP), SI        // x base = window base for output block 0
    MOVQ taps_base+48(FP), DI     // taps base (constant)
    MOVQ taps_len+56(FP), R8      // number of taps (>=1)

    MOVQ CX, R9
    SHRQ $3, R9                   // R9 = n / 8 = full 8-output blocks
    JZ   fir_avx2_tail

fir_avx2_block:
    VPXOR Y2, Y2, Y2             // acc = 0
    MOVQ  SI, BX                 // BX = window pointer = block base
    MOVQ  DI, R10                // R10 = taps pointer
    MOVQ  R8, R11                // R11 = tap counter
fir_avx2_tap:
    MOVWQSX (R10), AX           // AX = int64(taps[j]), sign-extended int16
    VMOVD   AX, X3
    VPBROADCASTD X3, Y3          // taps[j] in all 8 int32 lanes
    VMOVDQU  (BX), Y0            // window x[i+j .. i+j+7]
    VPMULDQ  Y3, Y0, Y4          // even-lane products (4x int64)
    VPSRLQ   $32, Y0, Y1         // slide odd lanes into even positions
    VPMULDQ  Y3, Y1, Y5          // odd-lane products (4x int64)
    VPSRLQ   $15, Y4, Y4         // low 32 of each = even result lanes
    VPSRLQ   $15, Y5, Y5         // low 32 of each = odd result lanes
    VPSLLQ   $32, Y5, Y5         // lift odd results to positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y6   // merge: 1,3,5,7 <- Y5 (odd), 0,2,4,6 <- Y4 (even)
    VPADDD   Y6, Y2, Y2          // acc += 8 Q15-truncated products (wrapping)
    ADDQ  $4, BX                 // slide window by 1 int32
    ADDQ  $2, R10                // next tap (int16)
    DECQ  R11
    JNZ   fir_avx2_tap
    VMOVDQU Y2, (DX)             // store 8 outputs
    ADDQ  $32, SI                // next block window base
    ADDQ  $32, DX                // next dst block
    DECQ  R9
    JNZ   fir_avx2_block

fir_avx2_tail:
    ANDQ $7, CX                  // CX = n mod 8 = scalar-output tail count
    JZ   fir_avx2_done
fir_avx2_tail_out:
    XORL R11, R11               // acc32 = 0
    MOVQ SI, BX                 // BX = window pointer for this output
    MOVQ DI, R10                // R10 = taps pointer
    MOVQ R8, R12                // R12 = tap counter
fir_avx2_tail_tap:
    MOVWQSX (R10), AX          // int64(taps[j])
    MOVLQSX (BX), R13          // int64(x[i+j])
    IMULQ   R13, AX            // taps[j] * x[i+j] (|p| <= 2^46)
    SARQ    $15, AX            // Q15 truncating arithmetic shift
    ADDL    AX, R11            // acc32 += low 32 (wrapping)
    ADDQ    $4, BX             // slide window by 1 int32
    ADDQ    $2, R10            // next tap
    DECQ    R12
    JNZ     fir_avx2_tail_tap
    MOVL    R11, (DX)          // store output
    ADDQ    $4, SI             // next output window base
    ADDQ    $4, DX             // next dst
    DECQ    CX
    JNZ     fir_avx2_tail_out

fir_avx2_done:
    VZEROUPPER
    RET

// func firSymValidQ15AVX2(dst, x []int32, center int16, pairs []int16)
// int32 valid convolution in correlation orientation with a SYMMETRIC Q15 tap set
// (one center tap plus K = len(pairs) mirror pairs, window length 2K+1),
// vectorized over the OUTPUT index: 8 outputs per iteration. For output block i
// the center sample sits at c = i+K. The accumulator Y2 starts at the center
// contribution: load the center window x[c .. c+7] (VMOVDQU), broadcast center,
// form the 8 Q15-TRUNCATED products with the exact scaleQ15AVX2 recombine (VPMULDQ
// even + VPSRLQ $32 slide + VPMULDQ odd, VPSRLQ $15 each, VPSLLQ $32 + VPBLENDD
// $0xAA to reassemble), then add. For each pair k in [1,K] it loads the LEFT
// window x[c-k .. c-k+7] and the RIGHT window x[c+k .. c+k+7], adds them with
// VPADDD (wrapping int32) BEFORE the multiply, the single-truncation-per-pair
// contract that makes this bit-exact with libopus comb_filter_const_c, then
// broadcasts pairs[k-1], forms the 8 truncated products with the same recombine,
// and VPADDD-accumulates. The center base R10 slides +32 per block; the left
// pointer BX slides -4 per pair, the right pointer R13 +4, the coeff pointer R11
// +2. After all pairs the 8 outputs are stored. The scalar-output tail runs the
// full center+pairs computation per remaining output, adding each mirror pair as
// a wrapping int32 sum (MOVL/ADDL) then sign-extending to int64 for the multiply
// (MOVLQSX/IMULQ/SARQ $15/ADDL), so the per-pair fold-then-truncate is preserved
// and the result is bit-exact with firSymValidQ15Go. The dispatch gates n =
// len(dst) >= 8, and the kernel reads x only up to index (n-1)+2K <= len(x)-1 (and
// down to x[0] at i=0,k=K), so there is no over-read. dst must not overlap x.
// Frame is dst+x+center+pairs: dst+0, x+24, center+48, pairs+56 (bytes 50-55 are
// alignment padding before the 8-byte-aligned pairs slice header).
TEXT ·firSymValidQ15AVX2(SB), NOSPLIT, $0-80
    MOVQ    dst_base+0(FP), DX
    MOVQ    dst_len+8(FP), CX       // n = number of outputs
    MOVQ    x_base+24(FP), SI       // x base
    MOVWQSX center+48(FP), AX       // center tap, sign-extended int16
    VMOVD   AX, X7
    VPBROADCASTD X7, Y7             // center in all 8 int32 lanes (constant)
    MOVQ    pairs_base+56(FP), DI   // pairs base (constant)
    MOVQ    pairs_len+64(FP), R8    // K = number of mirror pairs (>=0)

    MOVQ    R8, R10
    SHLQ    $2, R10                 // K*4 bytes
    ADDQ    SI, R10                 // R10 = &x[K] = center window base for block 0

    MOVQ    CX, R9
    SHRQ    $3, R9                  // R9 = n / 8 = full 8-output blocks
    JZ      sym_avx2_tail

sym_avx2_block:
    // center contribution -> acc
    VMOVDQU  (R10), Y0              // center window x[c .. c+7]
    VPMULDQ  Y7, Y0, Y4            // even-lane products
    VPSRLQ   $32, Y0, Y1           // slide odd lanes into even positions
    VPMULDQ  Y7, Y1, Y5           // odd-lane products
    VPSRLQ   $15, Y4, Y4          // low 32 of each = even result lanes
    VPSRLQ   $15, Y5, Y5          // low 32 of each = odd result lanes
    VPSLLQ   $32, Y5, Y5          // lift odd results to positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y6    // merge even/odd
    VPXOR    Y2, Y2, Y2           // acc = 0
    VPADDD   Y6, Y2, Y2           // acc = center products (wrapping)

    MOVQ  R8, R12                  // R12 = pair counter = K
    TESTQ R12, R12
    JZ    sym_avx2_store           // K == 0: center only
    MOVQ  R10, BX                  // BX = left window pointer
    SUBQ  $4, BX                   // pair k=1 left = center - 1 int32
    MOVQ  R10, R13                 // R13 = right window pointer
    ADDQ  $4, R13                  // pair k=1 right = center + 1 int32
    MOVQ  DI, R11                  // R11 = pairs coeff pointer
sym_avx2_pair:
    MOVWQSX (R11), AX             // pairs[k-1], sign-extended int16
    VMOVD   AX, X3
    VPBROADCASTD X3, Y3           // coeff in all 8 lanes
    VMOVDQU (BX), Y0             // left window x[c-k .. c-k+7]
    VMOVDQU (R13), Y1            // right window x[c+k .. c+k+7]
    VPADDD  Y1, Y0, Y0          // left+right, wrapping int32, BEFORE the multiply
    VPMULDQ  Y3, Y0, Y4
    VPSRLQ   $32, Y0, Y1
    VPMULDQ  Y3, Y1, Y5
    VPSRLQ   $15, Y4, Y4
    VPSRLQ   $15, Y5, Y5
    VPSLLQ   $32, Y5, Y5
    VPBLENDD $0xAA, Y5, Y4, Y6
    VPADDD   Y6, Y2, Y2          // acc += 8 Q15-truncated folded products (wrapping)
    SUBQ  $4, BX                  // next pair: left moves down
    ADDQ  $4, R13                 // right moves up
    ADDQ  $2, R11                 // next coeff
    DECQ  R12
    JNZ   sym_avx2_pair
sym_avx2_store:
    VMOVDQU Y2, (DX)              // store 8 outputs
    ADDQ  $32, R10                // next block center base
    ADDQ  $32, DX                 // next dst block
    DECQ  R9
    JNZ   sym_avx2_block

sym_avx2_tail:
    ANDQ $7, CX                   // CX = n mod 8 = scalar-output tail count
    JZ   sym_avx2_done
sym_avx2_tail_out:
    // center contribution
    MOVWQSX center+48(FP), AX    // center, sign-extended int64
    MOVLQSX (R10), R13           // int64(x[c])
    IMULQ   R13, AX              // center * x[c] (|p| <= 2^46)
    SARQ    $15, AX              // Q15 truncating arithmetic shift
    MOVL    AX, R9               // acc32 = center product (low 32)

    MOVQ  R8, R12                 // R12 = pair counter = K
    TESTQ R12, R12
    JZ    sym_avx2_tail_store     // K == 0
    MOVQ  R10, BX                 // left pointer = center
    SUBQ  $4, BX
    MOVQ  R10, R13                // right pointer = center
    ADDQ  $4, R13
    MOVQ  DI, R11                 // coeff pointer
sym_avx2_tail_pair:
    MOVWQSX (R11), AX            // int64(pairs[k-1])
    MOVL    (BX), SI            // left (low 32)
    ADDL    (R13), SI           // SI = left + right, wrapping int32, BEFORE truncation
    MOVLQSX SI, SI              // sign-extend the int32 sum to int64
    IMULQ   SI, AX              // coeff * sum (|p| <= 2^46)
    SARQ    $15, AX             // Q15 truncating arithmetic shift
    ADDL    AX, R9              // acc32 += low 32 (wrapping)
    SUBQ    $4, BX              // next pair
    ADDQ    $4, R13
    ADDQ    $2, R11
    DECQ    R12
    JNZ     sym_avx2_tail_pair
sym_avx2_tail_store:
    MOVL R9, (DX)                // store output
    ADDQ $4, R10                 // next output center base
    ADDQ $4, DX                  // next dst
    DECQ CX
    JNZ  sym_avx2_tail_out

sym_avx2_done:
    VZEROUPPER
    RET
