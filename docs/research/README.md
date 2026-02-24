# Research Notes

Investigations, feasibility studies, and experimental analysis for MCP Vector Search. These are point-in-time research documents and may contain experimental ideas or outdated information.

## AI and Code Review

- [AI Code Review Integration Feasibility](./ai-code-review-integration-feasibility-2026-02-22.md) — Feasibility analysis for AI-powered code review integration
- [Code Review Infrastructure Analysis](./code-review-infrastructure-analysis-2026-02-22.md) — Analysis of code review infrastructure requirements
- [PR/MR Review Techniques](./pr-mr-review-techniques-2026-02-23.md) — Research on pull request and merge request review approaches
- [Issue #89 Phase 2 Scope](./issue-89-phase-2-scope-2026-02-22.md) — Scope definition for phase 2 of code review features

## Embedding Models

- [Embedding Model Evaluation: CodeBERT vs MiniLM](./embedding-model-evaluation-codebert-vs-minilm-2026-02-19.md) — Comparative evaluation of embedding models
- [Embedding Model Upgrade Re-Embed Analysis](./embedding-model-upgrade-re-embed-analysis-2026-02-19.md) — Impact analysis of upgrading embedding models on existing indexes
- [Code Embedding Models: No Trust Remote Code](./code-embedding-models-no-trust-remote-code-2026-01-23.md) — Security analysis of embedding model loading
- [Code Embedding Models Research](./code-embedding-models-research-2026-02-20.md) — Survey of available code embedding models
- [Transformer Models for Code Search](./transformer-models-code-search-2025-2026.md) — Research on transformer-based code search models
- [GraphCodeBERT Default Changes](./GRAPHCODEBERT_DEFAULT_CHANGES.md) — Analysis of switching to GraphCodeBERT as default model
- [Chunking and Embedding Techniques](./chunking-embedding-techniques-research-2026-02-24.md) — Research on chunking strategies and embedding methods
- [Vector Search Quantization and ANN Optimizations](./vector-search-quantization-ann-optimizations-2026-02-24.md) — Approximate nearest neighbor and quantization techniques

## Indexing and Performance

- [Indexing Performance Architecture Analysis](./indexing-performance-architecture-analysis-2026-02-15.md) — Architecture-level analysis of indexing performance
- [Indexing Performance Bottleneck Analysis](./indexing-performance-bottleneck-analysis-2026-02-18.md) — Root cause analysis of indexing bottlenecks
- [Indexing Performance Optimization Implementation](./indexing-performance-optimization-implementation-2026-02-02.md) — Implementation details for indexing optimizations
- [Indexing Parallelization Analysis](./indexing-parallelization-analysis-2026-02-02.md) — Analysis of parallelization opportunities in indexing
- [Pipeline Parallelism Analysis](./pipeline-parallelism-analysis-2026-02-18.md) — Research on pipeline-level parallelism
- [Pipeline Parallelism Bottleneck](./pipeline-parallelism-bottleneck-2026-02-20.md) — Bottleneck identification in parallel pipeline stages
- [Parallel Pipeline Race Condition Analysis](./parallel-pipeline-race-condition-analysis-2026-02-22.md) — Race condition investigation in parallel indexing
- [Two-Phase Indexing Architecture](./two-phase-indexing-architecture-2026-02-04.md) — Research into two-phase indexing design
- [Two-Phase Architecture Summary](./two-phase-architecture-summary.md) — Summary of two-phase architecture decisions
- [Progressive Indexing Architecture](./progressive-indexing-architecture-2025-02-15.md) — Progressive indexing for large codebases
- [Batch Embedding Implementation Analysis](./batch-embedding-implementation-analysis.md) — Analysis of batch embedding strategies
- [M4 Max Performance Optimizations](./m4-max-performance-optimizations-2026-02-02.md) — Apple M4 Max specific performance tuning
- [MVS Index Scale Performance Investigation](./mvs-index-scale-performance-investigation-2026-02-23.md) — Performance at scale for large indexes
- [GPU Utilization Analysis](./gpu-utilization-analysis-2026-02-19.md) — GPU usage analysis for embedding generation

## AWS and Remote Infrastructure

- [AWS GPU Starvation Analysis](./aws-gpu-starvation-analysis-2026-02-21.md) — Analysis of GPU resource starvation on AWS
- [AWS T4 GPU Idle Bottleneck Analysis](./aws-t4-gpu-idle-bottleneck-analysis-2026-02-21.md) — T4 GPU idle time investigation
- [AWS Pipeline Verification](./aws-pipeline-verification-2026-02-20.md) — Verification of the AWS indexing pipeline
- [Duetto AWS GPU Instance Setup](./duetto-aws-gpu-instance-setup-2026-02-19.md) — Setting up GPU instances for Duetto
- [Duetto Code Intelligence API Audit](./duetto-code-intelligence-api-audit-2026-02-20.md) — Audit of Duetto's code intelligence APIs
- [Duetto Incremental Indexing Capabilities](./duetto-incremental-indexing-capabilities-2026-02-19.md) — Incremental indexing analysis for Duetto
- [Duetto Indexing Performance Investigation](./duetto-indexing-performance-investigation-2026-02-04.md) — Performance investigation for Duetto indexing
- [Duetto No Vectors Investigation](./duetto-no-vectors-investigation-2026-02-22.md) — Investigation of missing vectors in Duetto
- [MPS Device Decision Analysis](./mps-device-decision-analysis-2026-02-20.md) — Decision analysis for Apple MPS device support

## Knowledge Graph and Ontology

- [KG Ontology Research](./kg-ontology-research-2026-02-15.md) — Research into knowledge graph ontology design
- [AST-Based KG Generation Architecture](./ast-based-kg-generation-architecture-2026-02-16.md) — Architecture for AST-driven KG generation
- [Tree-Based KG Generation Without Parsing](./tree-based-kg-generation-without-parsing-2026-02-16.md) — KG generation using tree structures without full parsing
- [AST Extraction Pipeline Analysis](./ast-extraction-pipeline-analysis-2026-02-15.md) — Analysis of AST extraction pipeline design
- [Code Ontology Standards](./code-ontology-standards-2026-02-16.md) — Standards and best practices for code ontologies
- [Work Entities Ontology Research](./work-entities-ontology-research-2026-02-15.md) — Ontology research for work and task entities
- [Document Ontology Feasibility](./document-ontology-feasibility-2026-02-24.md) — Feasibility of a document-level ontology
- [Semantic Doc-Code Matching](./semantic-doc-code-matching-2025-02-15.md) — Matching documentation to code semantically
- [Imports Relationship Fix](./imports-relationship-fix-2026-02-16.md) — Fix for import relationship extraction in KG
- [KG Person Node Integration](./kg-person-node-integration-2026-02-20.md) — Adding person/author nodes to the knowledge graph
- [Lazy Relationship Implementation Status](./lazy-relationship-implementation-status.md) — Status of lazy-loaded relationship computation
- [Semantic Relationship Parallelization Analysis](./semantic-relationship-parallelization-analysis-2025-12-20.md) — Parallelizing semantic relationship extraction

## Visualization

- [Visualization Architecture Analysis](./visualization-architecture-analysis-2025-12-06.md) — Analysis of visualization system architecture
- [Visualization Performance Analysis](./visualization-performance-analysis-2026-01-27.md) — Performance analysis of the visualization layer
- [Visualization Analysis for Large DB Optimizations](./visualization-analysis-large-db-optimizations-2026-02-03.md) — Optimizations for visualizing large databases
- [Visualization Layout Configuration Analysis](./visualization-layout-configuration-analysis-2025-12-08.md) — Layout configuration options research
- [Visualization Report Buttons Analysis](./visualization-report-buttons-analysis-2025-12-15.md) — Analysis of report button UX
- [Visualization Styling Analysis](./visualization-styling-analysis-2026-02-20.md) — Research on visualization styling options
- [Visualize Index Command Research](./visualize-index-command-research-2026-02-20.md) — Research for the visualize-index command
- [Visualizer Issue Investigation](./visualizer-issue-investigation-2026-02-20.md) — Investigation of visualizer bugs and limitations
- [Multi-Root Visualization Architecture](./multi-root-visualization-architecture-2025-02-16.md) — Architecture for multi-root visualization
- [Treemap Sunburst Drill-Down Color Issues](./treemap-sunburst-drill-down-color-issues-2026-02-20.md) — Color issues in treemap/sunburst drill-down
- [D3 Automatic Spacing Research](./d3-automatic-spacing-research-2025-12-05.md) — D3.js automatic node spacing research
- [Code Graph Visualization Best Practices](./code-graph-visualization-best-practices-2025-12-05.md) — Best practices for code graph visualization
- [Phase 4 Visualization Export Plan](./phase-4-visualization-export-plan.md) — Planning for visualization export features
- [Phase 3 Cross-File Analysis Requirements](./phase3-cross-file-analysis-requirements.md) — Requirements for cross-file dependency analysis

## ChromaDB and Backend

- [ChromaDB Alternatives Evaluation](./chromadb-alternatives-evaluation-2026-02-02.md) — Evaluation of alternatives to ChromaDB
- [ChromaDB Baseline Storage Investigation](./chromadb-baseline-storage-investigation.md) — Storage baseline measurements for ChromaDB
- [ChromaDB Rust Panic - Indexing Analysis](./chromadb-rust-panic-indexing-analysis-2025-12-10.md) — Analysis of Rust panics during indexing
- [ChromaDB Rust Panic Investigation](./chromadb-rust-panic-investigation-2025-12-10.md) — Root cause investigation of ChromaDB Rust panics
- [ChromaDB Segfault Investigation](./chromadb-segfault-investigation-2026-01-19.md) — Investigation of ChromaDB segmentation faults
- [BM25 Vectors Backend Warning](./bm25-vectors-backend-warning-2026-02-22.md) — Analysis of BM25 backend warning messages
- [Rust Detection Investigation](./rust-detection-investigation-2026-02-04.md) — Investigation of Rust code detection
- [Rust Migration Performance Analysis](./rust-migration-performance-analysis-2026-02-02.md) — Performance analysis for potential Rust migration
- [Rust Rewrite Evaluation](./rust-rewrite-evaluation-2026-01-23.md) — Feasibility evaluation for Rust rewrite

## Code Analysis Features

- [Dead Code Analysis Feature Requirements](./dead-code-analysis-feature-requirements-2026-01-21.md) — Requirements for dead code detection
- [Issue #14: Code Smell Detection Requirements](./issue-14-code-smell-detection-requirements.md) — Requirements for code smell detection
- [Issue #17: Diff-Aware Analysis Research](./issue-17-diff-aware-analysis-research.md) — Research for diff-aware code analysis
- [Issue #18: Baseline Comparison Requirements](./issue-18-baseline-comparison-requirements.md) — Requirements for baseline comparison features
- [SARIF Output Format Requirements](./sarif-output-format-requirements-2025-12-11.md) — Requirements for SARIF-format output
- [Index Code Architecture](./index-code-architecture-2026-02-20.md) — Architecture of the index code module
- [Index Code Search Integration](./index-code-search-integration-2026-02-20.md) — Integration of index code with search
- [Secondary Code Index Architecture](./secondary-code-index-architecture-2026-02-20.md) — Architecture for a secondary code index
- [Code Index Footprint Analysis](./code-index-footprint-analysis-2026-02-20.md) — Disk and memory footprint analysis
- [Analyze Command Implementation Research](./analyze-command-implementation-research-2024-12-10.md) — Research for the analyze command
- [Structural Analysis - Apex/Kartik Path Bug](./apex-kartik-path-bug-2026-02-21.md) — Bug analysis for Apex path resolution
- [Augment Code Search Comparison](./augment-code-search-comparison-2026-02-21.md) — Comparison with Augment Code search capabilities

## LLM and Chat Integration

- [Chat Enhancement Architecture](./chat-enhancement-architecture-2026-02-20.md) — Architecture for chat command enhancements
- [LLM Controller Architecture Analysis](./llm-controller-architecture-analysis.md) — Analysis of the LLM controller design
- [OpenRouter Integration Analysis](./openrouter-integration-analysis-2026-02-03.md) — Research into OpenRouter API integration
- [Attention Interface Error Investigation](./attention-interface-error-investigation-2026-02-06.md) — Investigation of attention mechanism errors
- [Frontmatter Tag Support Analysis](./frontmatter-tag-support-analysis-2026-02-20.md) — Analysis of frontmatter tag handling in search

## Configuration and CLI

- [CLI Flags and Default Exclusions Analysis](./cli-flags-and-default-exclusions-analysis-2026-01-18.md) — Analysis of CLI flag design and default exclusion patterns
- [Default Extensions Analysis](./default-extensions-analysis-2026-01-18.md) — Research on default file extension support
- [Configuration Defaults Investigation](./configuration-defaults-investigation-2026-01-19.md) — Investigation of configuration default values
- [Dotfile Configuration Research](./dotfile-configuration-research-2025-12-08.md) — Research on dotfile-based configuration
- [Background Indexing CLI Structure](./background-indexing-cli-structure-2025-12-20.md) — CLI structure for background indexing
- [Setup Command Design](./setup-command-design-2025-11-25.md) — Design research for the setup command
- [Automatic Setup Command Design](./automatic-setup-command-design-2025-11-30.md) — Design for zero-config automatic setup
- [Setup and Installation MCP Integration Analysis](./setup-installation-mcp-integration-analysis-2025-11-30.md) — MCP integration analysis during setup

## Other Research

- [Story Module Codebase Patterns](./story-module-codebase-patterns-2026-02-20.md) — Patterns identified in the story module
- [GitStory Analysis](./gitstory-analysis-2026-02-20.md) — Analysis of git history storytelling
- [Messaging System Bug Analysis](./messaging-system-bug-analysis-2026-02-23.md) — Bug analysis for the messaging system
- [GitHub Actions Pipeline Analysis](./github-actions-pipeline-analysis-2026-01-07.md) — Analysis of CI pipeline setup
- [Gitignore Auto-Update Implementation](./gitignore-auto-update-implementation-2025-11-25.md) — Implementation of automatic gitignore updates
- [Gitignore Bug Analysis](./gitignore-bug-analysis-2026-01-16.md) — Bug analysis for gitignore handling
- [Homebrew Token Investigation](./homebrew-token-investigation-2025-11-25.md) — Investigation of Homebrew token handling
- [Claude Desktop Documentation Review](./claude-desktop-documentation-review-2025-12-02.md) — Review of Claude Desktop documentation
- [Claude Desktop vs VS Code Installer Analysis](./claude-desktop-vs-code-installer-analysis-2025-12-02.md) — Comparison of installer approaches
- [Pre-Release Readiness Check](./pre-release-readiness-check-2025-12-02.md) — Pre-release quality and readiness checks
- [Test Quality Examples](./test-quality-examples.md) — Examples of high-quality test patterns
- [Performance Optimization: Indexing and Visualization](./performance-optimization-indexing-visualization-2025-12-16.md) — Combined performance research
- [MCP Vector Search Structural Analysis Design](./mcp-vector-search-structural-analysis-design.md) — Design document for structural analysis

## Related Documentation

- [Parent Index](../README.md)
- [Internal Documentation](../internal/README.md)
- [Development Documentation](../development/README.md)
- [Architecture Documentation](../architecture/README.md)
