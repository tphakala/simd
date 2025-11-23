# Contributing to simd

Thank you for your interest in contributing to the SIMD library! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/simd.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `go test ./...`
6. Run linter: `golangci-lint run`
7. Commit your changes with a descriptive message
8. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Go 1.25 or later
- GCC (for C reference implementation tests)
- golangci-lint (for linting)

### Building and Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run benchmarks
go test ./f64 -bench=. -benchmem
go test ./f32 -bench=. -benchmem

# Run linter
golangci-lint run

# Generate test expectations from C reference
cd testdata
gcc -O2 -march=native -o generate_expectations generate_expectations.c -lm
./generate_expectations
```

## Code Organization

```
simd/
├── cpu/               # CPU feature detection
│   ├── cpu.go         # Common types and functions
│   ├── cpu_amd64.go
│   ├── cpu_arm64.go
│   └── cpu_other.go
├── f64/               # float64 operations
│   ├── f64.go         # Public API
│   ├── f64_go.go      # Pure Go implementations
│   ├── f64_amd64.go   # AMD64 dispatchers
│   ├── f64_amd64.s    # AMD64 assembly
│   ├── f64_arm64.go   # ARM64 dispatchers
│   ├── f64_arm64.s    # ARM64 assembly
│   ├── f64_other.go   # Fallback dispatchers
│   └── *_test.go      # Tests and benchmarks
├── f32/               # float32 operations (same structure)
├── c128/              # complex128 operations (same structure)
├── testdata/          # C reference implementation
├── .githooks/         # Git pre-commit hooks
├── Taskfile.yml       # Task runner configuration
├── doc.go             # Package documentation
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── go.mod
```

## Adding New Operations

When adding a new SIMD operation, follow these steps:

### 1. Add the Public API (`f64.go` / `f32.go`)

```go
// OperationName performs [description].
// Processes min(len(dst), len(a)) elements.
func OperationName(dst, a []float64) {
    n := min(len(dst), len(a))
    if n == 0 {
        return
    }
    operationName64(dst[:n], a[:n])
}
```

### 2. Add Pure Go Implementation (`f64_go.go`)

```go
func operationNameGo(dst, a []float64) {
    for i := range dst {
        dst[i] = // operation
    }
}
```

### 3. Add Dispatchers

**For AMD64 (`f64_amd64.go`):**

```go
func operationName64(dst, a []float64) {
    if hasAVX && len(dst) >= 4 {
        operationNameAVX(dst, a)
        return
    }
    operationNameGo(dst, a)
}

//go:noescape
func operationNameAVX(dst, a []float64)
```

**For ARM64 (`f64_arm64.go`):**

```go
func operationName64(dst, a []float64) {
    if hasNEON && len(dst) >= 2 {
        operationNameNEON(dst, a)
        return
    }
    operationNameGo(dst, a)
}

//go:noescape
func operationNameNEON(dst, a []float64)
```

**For other architectures (`f64_other.go`):**

```go
func operationName64(dst, a []float64) { operationNameGo(dst, a) }
```

### 4. Add Assembly Implementation

Add to `f64_amd64.s` and `f64_arm64.s`. For initial development, you can add a stub:

```asm
TEXT ·operationNameAVX(SB), $0-48
    JMP ·operationNameGo(SB)
```

### 5. Add Tests

Add tests in `f64_test.go` or create a new test file:

```go
func TestOperationName(t *testing.T) {
    tests := []struct {
        name string
        a    []float64
        want []float64
    }{
        {"empty", nil, nil},
        {"single", []float64{1}, []float64{1}},
        // Add more test cases
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            dst := make([]float64, len(tt.a))
            OperationName(dst, tt.a)
            // Assert results
        })
    }
}
```

### 6. Update C Reference

Add the operation to `testdata/generate_expectations.c` for validation.

### 7. Update Documentation

Update `README.md` to include the new operation in the API table.

## Code Style

- Follow standard Go formatting (`gofmt`)
- Use meaningful variable names
- Add doc comments for all public functions
- Keep functions focused and small
- Avoid magic numbers - use named constants

## Assembly Guidelines

- Use Go's plan9 assembly syntax
- Include comments explaining the algorithm
- Handle remainder elements after SIMD loops
- Use `VZEROUPPER` at the end of AVX functions (AMD64)
- Test with various slice sizes to exercise all code paths

## Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what the PR does and why
3. **Tests**: Include tests for new functionality
4. **Benchmarks**: Include benchmarks for performance-critical code
5. **Documentation**: Update README if adding new features

## Reporting Issues

When reporting issues, please include:

- Go version (`go version`)
- Operating system and architecture
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
