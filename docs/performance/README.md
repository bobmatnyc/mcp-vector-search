# Performance Documentation

Optimization work, benchmarks, and performance analysis for the MCP Vector Search indexing and search pipelines.

## Contents

- [Batch Size Auto-Tuning Implementation](./batch-size-auto-tuning.md) — Dynamic batch size selection based on available memory and throughput
- [Knowledge Graph Batch Insert Optimization](./kg_batch_optimization.md) — Bulk insert optimizations for the knowledge graph backend
- [Directory Filtering Performance Improvements](./performance-improvements.md) — Speed improvements for file discovery and directory exclusion logic
- [Vendor Patterns Performance Optimization](./performance-optimization.md) — Performance tuning for vendor pattern matching and filtering
- [Performance Optimizations - February 15, 2026](./performance-optimizations-2026-02-15.md) — Summary of indexing pipeline optimizations from February 15, 2026
- [Performance Optimizations: O(n²) → O(n) Indexing](./performance-optimizations-2026-02-23.md) — Algorithmic improvement reducing indexing complexity from quadratic to linear
- [Performance Optimizations for Indexing Pipeline (Issue #107)](./performance-optimizations-summary.md) — Comprehensive summary of all optimizations addressing issue #107
- [Search Optimizations](./search-optimizations.md) — Query-time search speed improvements and caching strategies

## Related Documentation

- [Parent Index](../README.md)
- [Architecture Documentation](../architecture/README.md)
- [Research Notes](../research/README.md)
