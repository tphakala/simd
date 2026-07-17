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
