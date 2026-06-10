//go:build amd64

#include "textflag.h"

// func cpuid(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)
// Raw CPUID wrapper, needed because golang.org/x/sys/cpu does not expose F16C.
// Modeled on x/sys/cpu's own cpu_x86.s. CPUID clobbers AX/BX/CX/DX, all of which
// are caller-saved general registers in the Go ABI (no reserved register touched).
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL subleaf+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET
