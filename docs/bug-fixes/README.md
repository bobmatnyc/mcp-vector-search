# Bug Fix Documentation

Documentation for specific bugs that were identified, investigated, and resolved in MCP Vector Search.

## Contents

- [Bug Fixes: Reindex and KeyboardInterrupt Handling](./bug-fixes-reindex-keyboard-interrupt.md) — Fixes for reindex command failures and unhandled keyboard interrupt crashes
- [Bugfix: Schema Evolution Destroying Embeddings](./bugfix-schema-evolution-2026-02-23.md) — Fix for schema migrations that inadvertently wiped stored embedding data
- [ChromaDB Segfault Protection](./chromadb-segfault-protection.md) — Defensive patterns to prevent segmentation faults from ChromaDB native code
- [Fix: "Too many open files" Error During Large Reindexing](./fix-file-descriptor-limit-2026-02-23.md) — Resolution for file descriptor exhaustion during large repository indexing
- [macOS SIGBUS Crash Fix](./macos-sigbus-fix.md) — Fix for bus error crashes specific to macOS memory alignment issues
- [SIGILL (Illegal Hardware Instruction) Fix](./SIGILL_FIX.md) — Fix for illegal hardware instruction crashes caused by CPU instruction set incompatibilities

## Related Documentation

- [Parent Index](../README.md)
- [Development Documentation](../development/README.md)
- [QA Verification Reports](../qa/)
