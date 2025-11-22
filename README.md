# simd

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/simd.svg)](https://pkg.go.dev/github.com/tphakala/simd)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/simd)](https://goreportcard.com/report/github.com/tphakala/simd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance SIMD (Single Instruction, Multiple Data) library for Go providing vectorized operations on float64 and float32 slices.

## Features

- **Pure Go assembly** - Native Go assembler, simple cross-compilation
- **Runtime CPU detection** - Automatically selects optimal implementation (AVX-512, AVX+FMA, SSE2, NEON, or pure Go)
- **Zero allocations** - All operations work on pre-allocated slices
- **23 operations** - Arithmetic, reduction, statistical, vector, and signal processing operations
- **Multi-architecture** - AMD64 (AVX-512/AVX+FMA/SSE2) and ARM64 (NEON) with pure Go fallback
- **Thread-safe** - All functions are safe for concurrent use

## Installation

```bash
go get github.com/tphakala/simd
```

Requires Go 1.25+

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/tphakala/simd/pkg/simd/cpu"
    "github.com/tphakala/simd/pkg/simd/f64"
)

func main() {
    fmt.Println("SIMD:", cpu.Info())

    // Vector operations
    a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
    b := []float64{8, 7, 6, 5, 4, 3, 2, 1}

    // Dot product
    dot := f64.DotProduct(a, b)
    fmt.Println("Dot product:", dot) // 120

    // Element-wise operations
    dst := make([]float64, len(a))
    f64.Add(dst, a, b)
    fmt.Println("Sum:", dst) // [9, 9, 9, 9, 9, 9, 9, 9]

    // Statistical operations
    mean := f64.Mean(a)
    stddev := f64.StdDev(a)
    fmt.Printf("Mean: %.2f, StdDev: %.2f\n", mean, stddev)

    // Vector operations
    f64.Normalize(dst, a)
    fmt.Println("Normalized:", dst)

    // Distance calculation
    dist := f64.EuclideanDistance(a, b)
    fmt.Println("Distance:", dist)
}
```

## Packages

### `cpu` - CPU Feature Detection

```go
import "github.com/tphakala/simd/pkg/simd/cpu"

fmt.Println(cpu.Info())      // "AMD64 AVX-512", "AMD64 AVX+FMA", "AMD64 SSE2", or "ARM64 NEON"
fmt.Println(cpu.HasAVX())    // true/false
fmt.Println(cpu.HasAVX512()) // true/false
fmt.Println(cpu.HasNEON())   // true/false
```

### `f64` - float64 Operations

| Category        | Function                  | Description                 | SIMD Width                  |
| --------------- | ------------------------- | --------------------------- | --------------------------- |
| **Arithmetic**  | `Add(dst, a, b)`          | Element-wise addition       | 8x (AVX-512) / 4x (AVX) / 2x (NEON) |
|                 | `Sub(dst, a, b)`          | Element-wise subtraction    | 8x / 4x / 2x                |
|                 | `Mul(dst, a, b)`          | Element-wise multiplication | 8x / 4x / 2x                |
|                 | `Div(dst, a, b)`          | Element-wise division       | 8x / 4x / 2x                |
|                 | `Scale(dst, a, s)`        | Multiply by scalar          | 8x / 4x / 2x                |
|                 | `AddScalar(dst, a, s)`    | Add scalar                  | 8x / 4x / 2x                |
|                 | `FMA(dst, a, b, c)`       | Fused multiply-add: a\*b+c  | 8x / 4x / 2x                |
| **Unary**       | `Abs(dst, a)`             | Absolute value              | 8x / 4x / 2x                |
|                 | `Neg(dst, a)`             | Negation                    | 8x / 4x / 2x                |
|                 | `Sqrt(dst, a)`            | Square root                 | 8x / 4x / 2x                |
|                 | `Reciprocal(dst, a)`      | Reciprocal (1/x)            | 8x / 4x / 2x                |
| **Reduction**   | `DotProduct(a, b)`        | Dot product                 | 8x / 4x / 2x                |
|                 | `Sum(a)`                  | Sum of elements             | 8x / 4x / 2x                |
|                 | `Min(a)`                  | Minimum value               | 8x / 4x / 2x                |
|                 | `Max(a)`                  | Maximum value               | 8x / 4x / 2x                |
| **Statistical** | `Mean(a)`                 | Arithmetic mean             | 8x / 4x / 2x                |
|                 | `Variance(a)`             | Population variance         | 8x / 4x / 2x                |
|                 | `StdDev(a)`               | Standard deviation          | 8x / 4x / 2x                |
| **Vector**      | `EuclideanDistance(a, b)` | L2 distance                 | 8x / 4x / 2x                |
|                 | `Normalize(dst, a)`       | Unit vector normalization   | 8x / 4x / 2x                |
|                 | `CumulativeSum(dst, a)`   | Running sum                 | Sequential                  |
| **Range**       | `Clamp(dst, a, min, max)` | Clamp to range              | 8x / 4x / 2x                |
| **Batch**       | `DotProductBatch(r, rows, v)` | Multiple dot products   | 8x / 4x / 2x                |
| **Signal**      | `ConvolveValid(dst, sig, k)` | FIR filter / convolution | 8x / 4x / 2x                |

### `f32` - float32 Operations

Same API as `f64` but for `float32` with wider SIMD:

| Architecture     | SIMD Width  |
| ---------------- | ----------- |
| AMD64 (AVX-512)  | 16x float32 |
| AMD64 (AVX+FMA)  | 8x float32  |
| AMD64 (SSE2)     | 4x float32  |
| ARM64 (NEON)     | 4x float32  |

## Performance

### AMD64 (Intel Core i7-1260P, AVX+FMA)

#### float64 Operations

| Operation         | Size | Time     | Throughput |
| ----------------- | ---- | -------- | ---------- |
| DotProduct        | 277  | 35.1 ns  | 126 GB/s   |
| DotProduct        | 1000 | 160 ns   | 100 GB/s   |
| Add               | 1000 | 97 ns    | 247 GB/s   |
| Mul               | 1000 | 88 ns    | 273 GB/s   |
| FMA               | 1000 | 120 ns   | 268 GB/s   |
| Sum               | 1000 | 86 ns    | 93 GB/s    |
| Mean              | 1000 | 92 ns    | 87 GB/s    |
| Variance          | 1000 | 483 ns   | 17 GB/s    |
| EuclideanDistance | 100  | 14.1 ns  | 114 GB/s   |
| Normalize         | 100  | 42 ns    | 38 GB/s    |
| Sqrt              | 100  | 66 ns    | 24 GB/s    |
| Reciprocal        | 100  | 43 ns    | 37 GB/s    |

#### float32 Operations

| Operation  | Size | Time    | Throughput |
| ---------- | ---- | ------- | ---------- |
| DotProduct | 100  | 9.9 ns  | 81 GB/s    |
| DotProduct | 1000 | 69 ns   | 116 GB/s   |
| Add        | 1000 | 49 ns   | 246 GB/s   |
| Mul        | 1000 | 44 ns   | 274 GB/s   |
| FMA        | 1000 | 59 ns   | 270 GB/s   |

#### Comparison vs Pure Go (1000 elements)

| Operation    | SIMD   | Pure Go | Speedup  |
| ------------ | ------ | ------- | -------- |
| DotProduct   | 160 ns | 128 ns  | ~1x      |
| Add          | 97 ns  | 138 ns  | **1.4x** |
| Mul          | 88 ns  | 202 ns  | **2.3x** |
| FMA          | 120 ns | 127 ns  | ~1x      |
| Sum          | 86 ns  | 126 ns  | **1.5x** |

### ARM64 (Raspberry Pi 5, NEON)

#### float64 Operations

| Operation         | Size | Time     | Throughput |
| ----------------- | ---- | -------- | ---------- |
| DotProduct        | 277  | 151 ns   | 29 GB/s    |
| DotProduct        | 1000 | 513 ns   | 31 GB/s    |
| Add               | 1000 | 775 ns   | 31 GB/s    |
| Mul               | 1000 | 727 ns   | 33 GB/s    |
| FMA               | 1000 | 890 ns   | 36 GB/s    |
| Sum               | 1000 | 635 ns   | 13 GB/s    |
| Mean              | 1000 | 677 ns   | 12 GB/s    |

#### float32 Operations

| Operation  | Size  | Time     | Throughput |
| ---------- | ----- | -------- | ---------- |
| DotProduct | 100   | 37 ns    | 21 GB/s    |
| DotProduct | 1000  | 263 ns   | 30 GB/s    |
| DotProduct | 10000 | 2.78 µs  | 29 GB/s    |
| Add        | 1000  | 389 ns   | 31 GB/s    |
| Mul        | 1000  | 390 ns   | 31 GB/s    |
| FMA        | 1000  | 479 ns   | 33 GB/s    |

#### Comparison vs Pure Go

| Operation        | Size | SIMD    | Pure Go  | Speedup  |
| ---------------- | ---- | ------- | -------- | -------- |
| DotProduct (f32) | 100  | 37 ns   | 137 ns   | **3.7x** |
| DotProduct (f32) | 1000 | 262 ns  | 1350 ns  | **5.2x** |
| DotProduct (f64) | 100  | 62 ns   | 138 ns   | **2.2x** |
| DotProduct (f64) | 1000 | 513 ns  | 1353 ns  | **2.6x** |
| Add (f32)        | 1000 | 389 ns  | 2015 ns  | **5.2x** |
| Sum (f32)        | 1000 | 343 ns  | 1327 ns  | **3.9x** |

### Performance Notes

- **AMD64**: On modern x86-64 CPUs, Go 1.25's auto-vectorization handles simple loops well, but explicit SIMD provides consistent performance and significant speedups for complex operations like EuclideanDistance (**6.4x**).

- **ARM64**: NEON SIMD provides substantial speedups over pure Go across all operations:
  - float32: **3.7x - 5.2x** faster (4 elements per 128-bit vector)
  - float64: **2.2x - 2.6x** faster (2 elements per 128-bit vector)

- **CumulativeSum** is inherently sequential (each element depends on the previous) and uses pure Go on all platforms.

## Architecture Support

| Architecture | Instruction Set | Status            |
| ------------ | --------------- | ----------------- |
| AMD64        | AVX-512         | Full SIMD support |
| AMD64        | AVX + FMA       | Full SIMD support |
| AMD64        | SSE2            | Full SIMD support |
| ARM64        | NEON/ASIMD      | Full SIMD support |
| ARM64        | SVE/SVE2        | Planned           |
| Other        | -               | Pure Go fallback  |

## Design Principles

1. **Pure Go assembly** - Native Go assembler for maximum portability and easy cross-compilation
2. **Runtime dispatch** - CPU features detected once at init time, zero runtime overhead
3. **Zero allocations** - No heap allocations in hot paths
4. **Safe defaults** - Gracefully falls back to pure Go on unsupported CPUs
5. **Boundary safe** - Handles any slice length, not just SIMD-aligned sizes

## Testing

The library includes comprehensive tests with pure Go reference implementations for validation:

```bash
# Run all tests
go test ./...

# Run tests with verbose output
task test

# Run benchmarks
task bench

# Compare SIMD vs pure Go performance
task bench:compare

# Show CPU SIMD capabilities
task cpu
```

See [Taskfile.yml](Taskfile.yml) for all available tasks.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
