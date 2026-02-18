# Performance Benchmarks

This directory contains benchmark scripts for measuring file discovery performance.

## Scripts

### `benchmark_patterns.py`
Tests file discovery performance with default ignore patterns (71 patterns).

**Usage**:
```bash
source .venv/bin/activate
python benchmarks/benchmark_patterns.py
```

**Expected Results**:
- Throughput: >10,000 files/sec
- Patterns: ~71 (default ignore patterns)

### `benchmark_vendor_patterns.py`
Tests file discovery performance with vendor library patterns (847 patterns).

**Usage**:
```bash
source .venv/bin/activate
python benchmarks/benchmark_vendor_patterns.py
```

**Expected Results**:
- Throughput: >15,000 files/sec
- Patterns: 847 (71 default + 776 vendor patterns)

### `test_pattern_correctness.py`
Verifies that optimized pattern matching produces correct results.

**Usage**:
```bash
source .venv/bin/activate
python benchmarks/test_pattern_correctness.py
```

**Expected Results**:
- All pattern matching tests pass ✅
- All directory filtering tests pass ✅

## Performance Targets

| Scenario | Target | Optimized |
|----------|--------|-----------|
| Default patterns | >100 files/sec | ~15,000 files/sec ✅ |
| Vendor patterns | >100 files/sec | ~20,000 files/sec ✅ |

## Optimization Details

See [../PERFORMANCE_OPTIMIZATION.md](../PERFORMANCE_OPTIMIZATION.md) for implementation details and performance analysis.
