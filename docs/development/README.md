# Development Documentation

Documentation for contributors, maintainers, and developers working on MCP Vector Search.

## Core Developer Docs

- [Development Setup](./setup.md) — Get your development environment ready
- [Contributing Guide](./contributing.md) — How to contribute, code standards, and PR process
- [Technical Architecture Guide](./architecture.md) — System design, layers, and data flow
- [Internal API Reference](./api.md) — Internal API documentation for core modules
- [Testing Guide](./testing.md) — Testing strategies, structure, and best practices
- [Code Quality](./code-quality.md) — Linting, formatting, and type checking standards
- [Versioning Guidelines](./versioning.md) — Semantic versioning and release workflow
- [Project Organization](./project-organization.md) — File organization standards and naming conventions

## Sprint Planning

- [Sprint Quickstart Guide](./sprint-quickstart.md) — Get started with active sprint issues
- [Sprint Plan Summary](./sprint-plan-summary.md) — Quick reference for sprint meetings
- [Structural Code Analysis - Sprint Plan](./sprint-plan.md) — Full sprint breakdown
- [Sprint Board](./sprint-board.md) — Visual tracking board for active work
- [GitHub Milestones and Issue Dependencies Setup](./github-milestones-setup.md) — Setting up GitHub milestones
- [GitHub Milestones - Quick Start](./MILESTONES_QUICKSTART.md) — Minimal milestones setup guide
- [PR Workflow Guide](./pr-workflow-guide.md) — Pull request workflow and conventions

## Implementation Summaries

- [MCP Auto-Installation Implementation Summary](./MCP_AUTO_INSTALLATION_IMPLEMENTATION.md) — Auto-install implementation details
- [OpenAI API Integration](./OPENAI_API_INTEGRATION.md) — OpenAI embedding integration
- [OpenRouter Setup Enhancement](./openrouter-setup-enhancement.md) — OpenRouter API key setup improvements
- [OpenRouter Setup Enhancement - Visual Demo](./openrouter-setup-demo.md) — Demo of OpenRouter setup flow
- [Phase 1 Complete: State Management System](./PHASE1_STATE_MANAGEMENT_COMPLETE.md) — Phase 1 state management
- [Phase 5 D3.js Visualization Enhancement](./phase-5-implementation-summary.md) — Phase 5 visualization implementation
- [Multiprocess File Parsing Implementation](./multiprocess-parsing-implementation.md) — Parallel file parsing
- [Progressive Loading Implementation](./progressive-loading-implementation.md) — Lazy-load visualization nodes
- [Streaming JSON Implementation](./STREAMING_JSON_IMPLEMENTATION.md) — Streaming JSON for large graphs
- [Git Integration Implementation](./git-integration-implementation.md) — Git history integration
- [LanceDB Backend](./LANCEDB_BACKEND.md) — LanceDB vector backend implementation
- [LanceDB Streaming Iterator](./lancedb-streaming-iterator.md) — Streaming iterator for LanceDB
- [LLM Client: Claude Sonnet 4.6 on AWS Bedrock](./llm-bedrock-sonnet-4.6-upgrade.md) — Bedrock LLM integration upgrade
- [Two-Phase Architecture Migration](./MIGRATION_TWO_PHASE.md) — Migration guide to two-phase indexing
- [Two-Phase Indexing Architecture Refactoring](./TWO_PHASE_INDEXING_REFACTORING.md) — Refactoring details
- [Async Relationship Computation](./async-relationships.md) — Background async relationship building
- [Non-Blocking Knowledge Graph Build](./nonblocking-kg-build.md) — Non-blocking KG construction
- [Tree-Based KG Generation](./tree-based-kg-generation.md) — Tree-based knowledge graph generation
- [Memory-Aware Worker Spawning](./resource_manager.md) — Dynamic worker count based on memory
- [Schema Versioning](./schema-versioning.md) — Database schema version management
- [Vendor Patterns Integration](./VENDOR_PATTERNS.md) — Third-party vendor pattern support
- [CLI Command Hierarchy Refactor](./cli-command-hierarchy-refactor.md) — CLI restructuring
- [D3.js Tree Layout Integration](./d3js-tree-integration.md) — D3.js tree visualization integration
- [Hybrid Visualization System](./hybrid-visualization-frontend-integration.md) — Frontend integration for hybrid viz

## Visualization Architecture

- [Visualization Architecture V2.0](./VISUALIZATION_ARCHITECTURE_V2.md) — Full V2 architecture document
- [Visualization Architecture V2.0 - Executive Summary](./VISUALIZATION_ARCHITECTURE_V2_SUMMARY.md) — Summary of V2 architecture
- [Visualization V2.0 - Documentation Index](./VISUALIZATION_V2_INDEX.md) — Index of all V2 visualization docs
- [Visualization V2.0 - Implementation Checklist](./VISUALIZATION_V2_CHECKLIST.md) — Checklist for V2 implementation
- [Visualization V2.0 - Visual Reference](./VISUALIZATION_V2_DIAGRAMS.md) — Diagrams for V2 visualization
- [Visualization V2.0 Implementation Summary](./VISUALIZATION_V2_IMPLEMENTATION_SUMMARY.md) — Implementation outcomes
- [Full Directory Tree Fan Visualization](./FULL_TREE_FAN_VISUALIZATION.md) — Fan-layout tree implementation
- [Rightward Tree Layout Implementation](./TREE_LAYOUT_IMPLEMENTATION.md) — Rightward tree layout
- [Two-Phase Visualization Layout](./TWO_PHASE_LAYOUT_IMPLEMENTATION.md) — Two-phase layout for large graphs
- [Tree Navigation Test Guide](./tree-navigation-test-guide.md) — Testing tree navigation features

## Bug Fixes (Visualization and MCP)

- [Bug Fix: Visualization Data Initialization for Large Graphs](./BUGFIX_VISUALIZATION_DATA_INIT.md)
- [Bug Fix: Progressive Loading Node Expansion](./bugfix-progressive-loading.md)
- [Fix: Parent-Child Linking Issue in File Filtering](./fix-filter-tree-hierarchy.md)
- [Fix: visibleNodes Initialization Bug for Large Graphs](./fix-visibleNodes-initialization.md)
- [Fix: Visualization Filter Expansion Issue](./fix-visualization-filter-expansion.md)
- [Duplicate Node Rendering Fix](./duplicate-node-rendering-fix.md)
- [Root Breadcrumb Navigation Fix](./ROOT_BREADCRUMB_FIX.md)
- [Root Node Filtering Fix](./root-node-filtering-fix.md)
- [Progressive Loading Fix - Root Cause Analysis](./progressive-loading-fix.md)
- [Dimension Mismatch Fix Summary](./DIMENSION_FIX_SUMMARY.md)
- [Fix: Vector Dimension Mismatch (768D vs 384D)](./DIMENSION_MISMATCH_FIX.md)
- [AST-Based Circular Dependency Detection Fix](./ast_circular_dependency_fix.md)
- [ChromaDB Rust Panic Defense Implementation](./chromadb-rust-panic-defense.md)
- [ChromaDB Rust Panic Recovery Implementation](./chromadb-rust-panic-recovery.md)
- [MCP Installation Bug Fix - 2025-12-01](./mcp-installation-bug-fix-2025-12-01.md)
- [MCPInstaller Platform Forcing Fix](./mcp-installer-platform-forcing-fix.md)
- [Monorepo Detection Fix](./monorepo-detection-fix.md)
- [Glob Pattern Fix for `--files` Option](./glob-pattern-fix-summary.md)
- [Setup API Key Interactive Prompt Fix](./setup-api-key-interactive-prompt-fix.md)
- [Review Module Test Coverage Improvement Summary](./test-coverage-improvement-summary.md)

## Related Documentation

- [Parent Index](../README.md)
- [Architecture Documentation](../architecture/README.md)
- [Bug Fix Documentation](../bug-fixes/README.md)
- [Research Notes](../research/README.md)
