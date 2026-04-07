# Changelog

All notable changes to this project will be documented in this file.

## [Released]

## [3.0.0] - 2026-04-01

### Features
- Full batch support across all dataflows (OS, WS, IS)
- Robust topology parser with case-insensitive `Batch` and `Batch Size` headers
- Batch-aware IFMAP and OFMAP address generation
- Correct MAC operation counting across batched workloads
- GEMM batch support with transformed dimensions
- Mixed batch size validation and rejection
- Comprehensive test coverage for batch operations
- Backward compatibility with legacy single-batch topologies

## [2.0.2] - 2024-02-07

### Features
- Bug fixes
- Faster compute simulation
- Solved a memory leak issue
  
## [2.0.1] - 2021-04-16

### Features
- Modular codebase
- Python package
