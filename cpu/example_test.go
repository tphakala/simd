package cpu_test

import (
	"fmt"

	"github.com/tphakala/simd/cpu"
)

func ExampleInfo() {
	// Returns a string like "AMD64 AVX+FMA" or "ARM64 NEON"
	info := cpu.Info()
	fmt.Printf("CPU supports: %s\n", info)
}

func ExampleHasAVX() {
	if cpu.HasAVX() {
		fmt.Println("AVX is available")
	} else {
		fmt.Println("AVX is not available")
	}
}

func ExampleHasNEON() {
	if cpu.HasNEON() {
		fmt.Println("NEON is available")
	} else {
		fmt.Println("NEON is not available")
	}
}

func Example() {
	// Check available SIMD features
	fmt.Println("CPU Info:", cpu.Info())
	fmt.Println("AVX:", cpu.HasAVX())
	fmt.Println("AVX2:", cpu.HasAVX2())
	fmt.Println("FMA:", cpu.HasFMA())
	fmt.Println("NEON:", cpu.HasNEON())
}
