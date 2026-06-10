//go:build amd64

#include "textflag.h"

// func toFloat32SliceF16C(dst []float32, src []Float16)
// Widens packed Float16 to float32 with VCVTPH2PS, 8 elements per iteration.
// The dispatch in f16_amd64.go calls this only with len(dst) == len(src) a
// non-zero multiple of 8; any sub-8 tail is handled in Go.
// Registers: DI/SI/CX and X0/Y1 only (no reserved register touched).
TEXT ·toFloat32SliceF16C(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), CX
	MOVQ src_base+24(FP), SI

	SHRQ $3, CX            // CX = number of 8-element blocks
	JZ   to_f16c_done

to_f16c_loop:
	VMOVDQU (SI), X0       // load 8 packed Float16 (128 bits)
	VCVTPH2PS X0, Y1       // widen to 8 float32 (256 bits)
	VMOVUPS Y1, (DI)
	ADDQ $16, SI           // 8 * 2 bytes consumed
	ADDQ $32, DI           // 8 * 4 bytes written
	DECQ CX
	JNZ  to_f16c_loop

to_f16c_done:
	VZEROUPPER
	RET

// func fromFloat32SliceF16C(dst []Float16, src []float32)
// Narrows float32 to Float16 with VCVTPS2PH, immediate 0 selecting
// round-to-nearest-even, matching fromFloat32Go. 8 elements per iteration.
// The dispatch calls this only with a non-zero multiple of 8.
TEXT ·fromFloat32SliceF16C(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), CX
	MOVQ src_base+24(FP), SI

	SHRQ $3, CX            // CX = number of 8-element blocks
	JZ   from_f16c_done

from_f16c_loop:
	VMOVUPS (SI), Y0       // load 8 float32 (256 bits)
	VCVTPS2PH $0, Y0, X1   // narrow to 8 Float16, round-to-nearest-even
	VMOVDQU X1, (DI)
	ADDQ $32, SI           // 8 * 4 bytes consumed
	ADDQ $16, DI           // 8 * 2 bytes written
	DECQ CX
	JNZ  from_f16c_loop

from_f16c_done:
	VZEROUPPER
	RET
