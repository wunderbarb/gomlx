// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// trimcheckpoint reads a GoMLX MNIST checkpoint (.json + .bin) and produces a trimmed
// checkpoint containing only the information needed for inference: model weights and
// the hyperparameters that affect graph construction.
//
// Usage:
//
//	go run trimcheckpoint.go -checkpoint_dir <dir> [-output_dir <dir>]
//
// If -output_dir is not given, it defaults to <checkpoint_dir>/inference.
package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/klog/v2"
)

// inferenceParams lists the hyperparameter keys that affect model graph construction
// and are therefore required at inference time.
var inferenceParams = map[string]bool{
	"model":             true,
	"cnn_normalization": true,
	"cnn_dropout_rate":  true,
	"dropout_rate":      true,
}

// optimizerScopes lists scope prefixes for optimizer variables that should be excluded.
var optimizerScopes = []string{
	"var:/optimizers/",
	"var:/AdamOptimizer/",
}

// serializedData mirrors the JSON structure of a GoMLX checkpoint.
type serializedData struct {
	Params    []serializedParam `json:"Params"`
	Variables []serializedVar   `json:"Variables"`
	BinFormat string            `json:"BinFormat"`
}

type serializedParam struct {
	Scope     string `json:"Scope"`
	Key       string `json:"Key"`
	Value     any    `json:"Value"`
	ValueType string `json:"ValueType"`
}

type serializedVar struct {
	ParameterName string `json:"ParameterName"`
	Dimensions    []int  `json:"Dimensions"`
	DType         string `json:"DType"`
	Pos           int    `json:"Pos"`
	Length        int    `json:"Length"`
}

var (
	flagCheckpointDir = flag.String("checkpoint_dir", "", "Directory containing the checkpoint to trim.")
	flagOutputDir     = flag.String("output_dir", "", "Output directory for trimmed checkpoint. Defaults to <checkpoint_dir>/inference.")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagCheckpointDir == "" {
		klog.Fatal("-checkpoint_dir is required")
	}

	outputDir := *flagOutputDir
	if outputDir == "" {
		outputDir = filepath.Join(*flagCheckpointDir, "inference")
	}

	// Find the latest checkpoint in the directory.
	baseName, err := latestCheckpoint(*flagCheckpointDir)
	if err != nil {
		klog.Fatalf("Finding checkpoint: %+v", err)
	}
	fmt.Printf("Reading checkpoint: %s\n", baseName)

	// Read the JSON metadata.
	jsonPath := filepath.Join(*flagCheckpointDir, baseName+".json")
	jsonData, err := os.ReadFile(jsonPath)
	if err != nil {
		klog.Fatalf("Reading %s: %+v", jsonPath, err)
	}
	var checkpoint serializedData
	if err := json.Unmarshal(jsonData, &checkpoint); err != nil {
		klog.Fatalf("Parsing JSON: %+v", err)
	}

	// Read and decompress the binary data.
	binPath := filepath.Join(*flagCheckpointDir, baseName+".bin")
	binData, err := readBinFile(binPath)
	if err != nil {
		klog.Fatalf("Reading %s: %+v", binPath, err)
	}

	// Filter params: keep only those needed for inference.
	var trimmedParams []serializedParam
	for _, p := range checkpoint.Params {
		if inferenceParams[p.Key] {
			trimmedParams = append(trimmedParams, p)
			fmt.Printf("  Keeping param: scope=%q key=%q value=%v\n", p.Scope, p.Key, p.Value)
		}
	}
	fmt.Printf("  Params: %d → %d\n", len(checkpoint.Params), len(trimmedParams))

	// Filter variables: keep only model weights (exclude optimizer state, global_step, rng state).
	var trimmedVars []serializedVar
	var trimmedBinBuf bytes.Buffer
	pos := 0
	for _, v := range checkpoint.Variables {
		if shouldExcludeVariable(v.ParameterName) {
			fmt.Printf("  Excluding variable: %s\n", v.ParameterName)
			continue
		}
		// Extract this variable's bytes from the decompressed binary.
		if v.Pos+v.Length > len(binData) {
			klog.Fatalf("Variable %s: pos=%d length=%d exceeds binary data size %d",
				v.ParameterName, v.Pos, v.Length, len(binData))
		}
		varBytes := binData[v.Pos : v.Pos+v.Length]

		newVar := serializedVar{
			ParameterName: v.ParameterName,
			Dimensions:    v.Dimensions,
			DType:         v.DType,
			Pos:           pos,
			Length:        v.Length,
		}
		trimmedVars = append(trimmedVars, newVar)
		trimmedBinBuf.Write(varBytes)
		pos += v.Length
		fmt.Printf("  Keeping variable: %s (dims=%v, dtype=%s, %d bytes)\n",
			v.ParameterName, v.Dimensions, v.DType, v.Length)
	}
	fmt.Printf("  Variables: %d → %d\n", len(checkpoint.Variables), len(trimmedVars))
	fmt.Printf("  Binary data: %d → %d bytes\n", len(binData), trimmedBinBuf.Len())

	// Build the trimmed checkpoint.
	trimmed := serializedData{
		Params:    trimmedParams,
		Variables: trimmedVars,
		BinFormat: "gzip",
	}

	// Write output files.
	if err := os.MkdirAll(outputDir, 0770); err != nil {
		klog.Fatalf("Creating output dir: %+v", err)
	}

	outJSONPath := filepath.Join(outputDir, "checkpoint.json")
	outJSON, err := json.MarshalIndent(&trimmed, "", "\t")
	if err != nil {
		klog.Fatalf("Marshaling JSON: %+v", err)
	}
	// json.MarshalIndent doesn't add trailing newline; json.Encoder.Encode does.
	outJSON = append(outJSON, '\n')
	if err := os.WriteFile(outJSONPath, outJSON, 0660); err != nil {
		klog.Fatalf("Writing %s: %+v", outJSONPath, err)
	}

	outBinPath := filepath.Join(outputDir, "checkpoint.bin")
	if err := writeGzipBinFile(outBinPath, trimmedBinBuf.Bytes()); err != nil {
		klog.Fatalf("Writing %s: %+v", outBinPath, err)
	}

	fmt.Printf("\nTrimmed checkpoint written to:\n  %s\n  %s\n", outJSONPath, outBinPath)
}

// shouldExcludeVariable returns true if the variable is not needed for inference.
func shouldExcludeVariable(parameterName string) bool {
	// Exclude optimizer variables.
	for _, prefix := range optimizerScopes {
		if strings.HasPrefix(parameterName, prefix) {
			return true
		}
	}
	// Exclude global_step.
	if parameterName == "var:/global_step" {
		return true
	}
	// Exclude RNG state.
	if parameterName == "var:/#rngState" {
		return true
	}
	return false
}

// latestCheckpoint finds the most recent checkpoint base name in the directory.
func latestCheckpoint(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("reading directory %q: %w", dir, err)
	}
	var checkpoints []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "checkpoint-") && strings.HasSuffix(name, ".json") {
			baseName := name[:len(name)-len(".json")]
			checkpoints = append(checkpoints, baseName)
		}
	}
	if len(checkpoints) == 0 {
		return "", fmt.Errorf("no checkpoints found in %q", dir)
	}
	sort.Strings(checkpoints)
	return checkpoints[len(checkpoints)-1], nil
}

const binHeader = "gomlx_checkpoints"

// readBinFile reads a GoMLX .bin file, handling both gzip-compressed and uncompressed formats.
// Returns the raw decompressed tensor data.
func readBinFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %q: %w", path, err)
	}
	defer f.Close()

	// Check for the gomlx_checkpoints header.
	header := make([]byte, len(binHeader))
	_, err = io.ReadFull(f, header)
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	if string(header) != binHeader {
		// Uncompressed format: rewind and read all.
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seeking: %w", err)
		}
		return io.ReadAll(f)
	}

	// Read compression type.
	var compLen uint8
	if err := binary.Read(f, binary.BigEndian, &compLen); err != nil {
		return nil, fmt.Errorf("reading compression type length: %w", err)
	}
	compType := make([]byte, compLen)
	if _, err := io.ReadFull(f, compType); err != nil {
		return nil, fmt.Errorf("reading compression type: %w", err)
	}
	if string(compType) != "gzip" {
		return nil, fmt.Errorf("unsupported compression type: %q", string(compType))
	}

	// Decompress gzip data.
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("creating gzip reader: %w", err)
	}
	defer gz.Close()
	return io.ReadAll(gz)
}

// writeGzipBinFile writes tensor data as a GoMLX gzip-compressed .bin file.
func writeGzipBinFile(path string, data []byte) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %q: %w", path, err)
	}
	defer f.Close()

	// Write header.
	if _, err := f.Write([]byte(binHeader)); err != nil {
		return fmt.Errorf("writing header: %w", err)
	}
	if err := binary.Write(f, binary.BigEndian, uint8(len("gzip"))); err != nil {
		return fmt.Errorf("writing compression length: %w", err)
	}
	if _, err := f.Write([]byte("gzip")); err != nil {
		return fmt.Errorf("writing compression type: %w", err)
	}

	// Write gzip-compressed data.
	gz := gzip.NewWriter(f)
	if _, err := gz.Write(data); err != nil {
		gz.Close()
		return fmt.Errorf("writing compressed data: %w", err)
	}
	if err := gz.Close(); err != nil {
		return fmt.Errorf("closing gzip writer: %w", err)
	}
	return nil
}
