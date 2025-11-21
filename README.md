# simd

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/simd.svg)](https://pkg.go.dev/github.com/tphakala/simd)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/simd)](https://goreportcard.com/report/github.com/tphakala/simd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance SIMD (Single Instruction, Multiple Data) library for Go providing vectorized operations on float64 and float32 slices.

## Features

- **Pure Go assembly** - No CGO required, simple cross-compilation
- **Runtime CPU detection** - Automatically selects optimal implementation (AVX+FMA, NEON, or pure Go)
- **Zero allocations** - All operations work on pre-allocated slices
- **23 operations** - Arithmetic, reduction, statistical, vector, and signal processing operations
- **Multi-architecture** - AMD64 (AVX+FMA) and ARM64 (NEON) with pure Go fallback
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

fmt.Println(cpu.Info())      // "AMD64 AVX2+FMA" or "ARM64 NEON"
fmt.Println(cpu.HasAVX())    // true/false
fmt.Println(cpu.HasNEON())   // true/false
```

### `f64` - float64 Operations

| Category        | Function                  | Description                 | SIMD Width           |
| --------------- | ------------------------- | --------------------------- | -------------------- |
| **Arithmetic**  | `Add(dst, a, b)`          | Element-wise addition       | 4x (AVX) / 2x (NEON) |
|                 | `Sub(dst, a, b)`          | Element-wise subtraction    | 4x / 2x              |
|                 | `Mul(dst, a, b)`          | Element-wise multiplication | 4x / 2x              |
|                 | `Div(dst, a, b)`          | Element-wise division       | 4x / 2x              |
|                 | `Scale(dst, a, s)`        | Multiply by scalar          | 4x / 2x              |
|                 | `AddScalar(dst, a, s)`    | Add scalar                  | 4x / 2x              |
|                 | `FMA(dst, a, b, c)`       | Fused multiply-add: a\*b+c  | 4x / 2x              |
| **Unary**       | `Abs(dst, a)`             | Absolute value              | 4x / 2x              |
|                 | `Neg(dst, a)`             | Negation                    | 4x / 2x              |
|                 | `Sqrt(dst, a)`            | Square root                 | 4x / 2x              |
|                 | `Reciprocal(dst, a)`      | Reciprocal (1/x)            | 4x / 2x              |
| **Reduction**   | `DotProduct(a, b)`        | Dot product                 | 4x / 2x              |
|                 | `Sum(a)`                  | Sum of elements             | 4x / 2x              |
|                 | `Min(a)`                  | Minimum value               | 4x / 2x              |
|                 | `Max(a)`                  | Maximum value               | 4x / 2x              |
| **Statistical** | `Mean(a)`                 | Arithmetic mean             | 4x / 2x              |
|                 | `Variance(a)`             | Population variance         | 4x / 2x              |
|                 | `StdDev(a)`               | Standard deviation          | 4x / 2x              |
| **Vector**      | `EuclideanDistance(a, b)` | L2 distance                 | 4x / 2x              |
|                 | `Normalize(dst, a)`       | Unit vector normalization   | 4x / 2x              |
|                 | `CumulativeSum(dst, a)`   | Running sum                 | Sequential           |
| **Range**       | `Clamp(dst, a, min, max)` | Clamp to range              | 4x / 2x              |
| **Batch**       | `DotProductBatch(r, rows, v)` | Multiple dot products   | 4x / 2x              |
| **Signal**      | `ConvolveValid(dst, sig, k)` | FIR filter / convolution | 4x / 2x              |

### `f32` - float32 Operations

Same API as `f64` but for `float32` with wider SIMD:

| Architecture | SIMD Width |
| ------------ | ---------- |
| AMD64 (AVX)  | 8x float32 |
| ARM64 (NEON) | 4x float32 |

## Performance

Benchmarks on AMD64 with AVX+FMA (Intel/AMD processor):

### float64 Operations

| Operation         | Size | Time    | Throughput |
| ----------------- | ---- | ------- | ---------- |
| DotProduct        | 277  | 34.9 ns | 127 GB/s   |
| DotProduct        | 1000 | 159 ns  | 100 GB/s   |
| Add               | 1000 | 106 ns  | 226 GB/s   |
| Mul               | 1000 | 109 ns  | 220 GB/s   |
| FMA               | 1000 | 121 ns  | 263 GB/s   |
| Sum               | 1000 | 90.6 ns | 88 GB/s    |
| Mean              | 1000 | 89.6 ns | 89 GB/s    |
| Variance          | 1000 | 540 ns  | 15 GB/s    |
| EuclideanDistance | 100  | 90 ns   | 18 GB/s    |
| Normalize         | 100  | 42.4 ns | 38 GB/s    |
| Sqrt              | 100  | 130 ns  | 12 GB/s    |
| Reciprocal        | 100  | 86.6 ns | 18 GB/s    |

### float32 Operations

| Operation  | Size | Time    | Throughput |
| ---------- | ---- | ------- | ---------- |
| DotProduct | 100  | 7.2 ns  | 111 GB/s   |
| DotProduct | 1000 | 70 ns   | 114 GB/s   |
| Add        | 1000 | 47.3 ns | 253 GB/s   |
| Mul        | 1000 | 47.5 ns | 252 GB/s   |
| FMA        | 1000 | 63.5 ns | 252 GB/s   |

### Comparison vs Pure Go

Benchmarks on Intel Core i7-1260P (AVX+FMA):

#### float64 Operations (1000 elements unless noted)

| Operation         | SIMD    | Pure Go | Speedup  |
| ----------------- | ------- | ------- | -------- |
| DotProduct        | 165 ns  | 128 ns  | ~1x      |
| Add               | 105 ns  | 138 ns  | **1.3x** |
| Mul               | 104 ns  | 202 ns  | **1.9x** |
| FMA               | 124 ns  | 127 ns  | ~1x      |
| Sum               | 82 ns   | 126 ns  | **1.5x** |
| Mean              | 83 ns   | 126 ns  | **1.5x** |
| Variance          | 538 ns  | 122 ns  | 0.2x     |
| Sqrt (100)        | 129 ns  | 24 ns   | 0.2x     |
| Reciprocal (100)  | 87 ns   | 24 ns   | 0.3x     |
| EuclideanDist(100)| 93 ns   | 24 ns   | 0.3x     |
| Normalize (100)   | 39 ns   | 51 ns   | **1.3x** |
| CumulativeSum     | 443 ns  | 129 ns  | 0.3x     |
| DotProductBatch   | 67 ns   | -       | -        |
| ConvolveValid     | 7.3 µs  | -       | -        |

#### float32 Operations (1000 elements)

| Operation    | SIMD   | Pure Go | Speedup  |
| ------------ | ------ | ------- | -------- |
| DotProduct   | 70 ns  | 124 ns  | **1.8x** |
| Add          | 47 ns  | 131 ns  | **2.8x** |
| Mul          | 50 ns  | 207 ns  | **4.1x** |
| FMA          | 59 ns  | 126 ns  | **2.1x** |

**Note:** Modern Go compilers (1.25+) perform aggressive auto-vectorization on simple loops.
SIMD benefits are more pronounced on older Go versions and for complex operations.
Throughput numbers above (in Performance section) are the primary performance metric.

## Architecture Support

| Architecture | Instruction Set | Status            |
| ------------ | --------------- | ----------------- |
| AMD64        | AVX + FMA       | Full SIMD support |
| AMD64        | SSE2 only       | Pure Go fallback  |
| ARM64        | NEON/ASIMD      | Full SIMD support |
| ARM64        | SVE/SVE2        | Planned           |
| Other        | -               | Pure Go fallback  |

## Design Principles

1. **No CGO** - Pure Go assembly for maximum portability and easy cross-compilation
2. **Runtime dispatch** - CPU features detected once at init time, zero runtime overhead
3. **Zero allocations** - No heap allocations in hot paths
4. **Safe defaults** - Gracefully falls back to pure Go on unsupported CPUs
5. **Boundary safe** - Handles any slice length, not just SIMD-aligned sizes

## Testing

The library includes comprehensive tests validated against a C reference implementation:

```bash
# Run all tests
go test ./...

# Run benchmarks
go test ./pkg/simd/f64 -bench=. -benchmem

# Generate test expectations from C reference
cd testdata && gcc -O2 -march=native -o generate_expectations generate_expectations.c -lm
./generate_expectations
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
