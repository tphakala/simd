package asmencoding

// This file adds an optional cross-check that decodes hand-encoded ARM64 words
// with an external aarch64 objdump (GNU binutils). It exists because
// golang.org/x/arch/arm64/arm64asm cannot decode ARMv8.2 half-precision (.8H)
// FP16 SIMD instructions, so those WORD directives are invisible to the pure-Go
// path in this package. objdump closes that gap when binutils is installed;
// callers skip the check cleanly when it is not.

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	// bytesPerWord is the size in bytes of one ARM64 instruction word.
	bytesPerWord = 4
	// objdumpTimeout bounds a single objdump invocation so a wedged tool
	// cannot hang the test.
	objdumpTimeout = 20 * time.Second
	// probeWord is a known FP16 (.8H) instruction used to confirm a candidate
	// objdump actually targets aarch64. It is FMUL V2.8H, V0.8H, V1.8H.
	probeWord uint32 = 0x6E411C02
	// probeExpect is probeWord's expected disassembly.
	probeExpect = "FMUL V2.8H, V0.8H, V1.8H"
)

// SIMDObjdumpEnv names an environment variable that, when set, overrides the
// objdump program used for the aarch64 cross-check (highest priority).
const SIMDObjdumpEnv = "SIMD_OBJDUMP"

// objdumpLineRe matches one disassembly line from `objdump -D`, capturing the
// 32-bit instruction word and the instruction text. Example line:
//
//	0:	6e411c02 	fmul	v2.8h, v0.8h, v1.8h
var objdumpLineRe = regexp.MustCompile(`(?m)^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+(\S.*?)\s*$`)

// objdumpCandidates returns objdump program names to try, in priority order.
// SIMD_OBJDUMP wins when set. aarch64-linux-gnu-objdump is the cross-binutils
// name on non-arm64 hosts. The host "objdump" is tried only on a GNU/Linux
// arm64 host, where it natively targets aarch64; FindObjdump's probe rejects it
// anywhere else (for example llvm-objdump on macOS).
func objdumpCandidates() []string {
	const maxCandidates = 3
	c := make([]string, 0, maxCandidates)
	if env := os.Getenv(SIMDObjdumpEnv); env != "" {
		c = append(c, env)
	}
	c = append(c, "aarch64-linux-gnu-objdump")
	if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		c = append(c, "objdump")
	}
	return c
}

// FindObjdump returns the path to an objdump that can disassemble aarch64 FP16
// (.8H) instructions, or "" if none is usable. Each candidate is probed on a
// known instruction, so a wrong-architecture objdump (or a non-GNU one such as
// llvm-objdump that does not understand -m aarch64) is rejected rather than
// trusted.
func FindObjdump() string {
	for _, name := range objdumpCandidates() {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		decoded, err := DisassembleWords(context.Background(), path, []uint32{probeWord})
		if err != nil {
			continue
		}
		if VerifyDecoded(decoded[probeWord], probeExpect).Status == Match {
			return path
		}
	}
	return ""
}

// DisassembleWords disassembles each 32-bit word with the given objdump,
// returning a map from instruction word to its GNU-syntax disassembly. Words
// are written little-endian (aarch64 byte order) and decoded with
// `-D -b binary -m aarch64`. Identical words collapse to a single map entry.
func DisassembleWords(ctx context.Context, tool string, words []uint32) (map[uint32]string, error) {
	f, err := os.CreateTemp("", "asmencoding-*.bin")
	if err != nil {
		return nil, fmt.Errorf("create temp: %w", err)
	}
	defer func() { _ = os.Remove(f.Name()) }()

	buf := make([]byte, bytesPerWord*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(buf[i*bytesPerWord:], w)
	}
	if _, err = f.Write(buf); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("write temp: %w", err)
	}
	if err = f.Close(); err != nil {
		return nil, fmt.Errorf("close temp: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, objdumpTimeout)
	defer cancel()
	// -EL forces little-endian decoding so the check still runs on the rare
	// host whose binutils defaults to AArch64 big-endian, rather than skipping.
	cmd := exec.CommandContext(ctx, tool, "-D", "-b", "binary", "-m", "aarch64", "-EL", f.Name())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err = cmd.Run(); err != nil {
		return nil, fmt.Errorf("run %s: %w: %s", tool, err, strings.TrimSpace(stderr.String()))
	}

	out := make(map[uint32]string, len(words))
	for _, m := range objdumpLineRe.FindAllStringSubmatch(stdout.String(), -1) {
		v, perr := strconv.ParseUint(m[1], 16, 32)
		if perr != nil {
			continue
		}
		out[uint32(v)] = strings.Join(strings.Fields(m[2]), " ")
	}
	return out, nil
}
