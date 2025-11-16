#!/bin/bash

# Documentation Reorganization Script
# This script moves documentation files to their new locations
# Uses git mv to preserve history

set -e  # Exit on error

PROJECT_ROOT="/Users/masa/Projects/mcp-vector-search"
DOCS_DIR="$PROJECT_ROOT/docs"

echo "================================================"
echo "Documentation Reorganization Script"
echo "================================================"
echo ""
echo "This script will reorganize the documentation structure."
echo "All moves use 'git mv' to preserve file history."
echo ""
echo "⚠️  WARNING: This will move/rename many files."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

cd "$PROJECT_ROOT"

echo ""
echo "================================================"
echo "Phase 1: Simple Moves (Direct Renames)"
echo "================================================"
echo ""

# To getting-started/
echo "Moving to getting-started/..."
if [ -f "$DOCS_DIR/CONFIGURATION.md" ]; then
    git mv "$DOCS_DIR/CONFIGURATION.md" "$DOCS_DIR/getting-started/configuration.md"
    echo "✓ Moved CONFIGURATION.md → getting-started/configuration.md"
fi

# To guides/
echo ""
echo "Moving to guides/..."
if [ -f "$DOCS_DIR/CLI_FEATURES.md" ]; then
    git mv "$DOCS_DIR/CLI_FEATURES.md" "$DOCS_DIR/guides/cli-usage.md"
    echo "✓ Moved CLI_FEATURES.md → guides/cli-usage.md"
fi

if [ -f "$DOCS_DIR/MCP_FILE_WATCHING.md" ]; then
    git mv "$DOCS_DIR/MCP_FILE_WATCHING.md" "$DOCS_DIR/guides/file-watching.md"
    echo "✓ Moved MCP_FILE_WATCHING.md → guides/file-watching.md"
fi

if [ -f "$DOCS_DIR/mcp-integration.md" ]; then
    git mv "$DOCS_DIR/mcp-integration.md" "$DOCS_DIR/guides/mcp-integration.md"
    echo "✓ Moved mcp-integration.md → guides/mcp-integration.md"
fi

# To reference/
echo ""
echo "Moving to reference/..."
if [ -f "$DOCS_DIR/FEATURES.md" ]; then
    git mv "$DOCS_DIR/FEATURES.md" "$DOCS_DIR/reference/features.md"
    echo "✓ Moved FEATURES.md → reference/features.md"
fi

if [ -f "$DOCS_DIR/STRUCTURE.md" ]; then
    git mv "$DOCS_DIR/STRUCTURE.md" "$DOCS_DIR/reference/architecture.md"
    echo "✓ Moved STRUCTURE.md → reference/architecture.md"
fi

# To development/
echo ""
echo "Moving to development/..."
if [ -f "$DOCS_DIR/DEVELOPMENT.md" ]; then
    git mv "$DOCS_DIR/DEVELOPMENT.md" "$DOCS_DIR/development/setup.md"
    echo "✓ Moved DEVELOPMENT.md → development/setup.md"
fi

if [ -f "$DOCS_DIR/developer/DEVELOPER.md" ]; then
    git mv "$DOCS_DIR/developer/DEVELOPER.md" "$DOCS_DIR/development/architecture.md"
    echo "✓ Moved developer/DEVELOPER.md → development/architecture.md"
fi

if [ -f "$DOCS_DIR/developer/API.md" ]; then
    git mv "$DOCS_DIR/developer/API.md" "$DOCS_DIR/development/api.md"
    echo "✓ Moved developer/API.md → development/api.md"
fi

if [ -f "$DOCS_DIR/developer/CONTRIBUTING.md" ]; then
    git mv "$DOCS_DIR/developer/CONTRIBUTING.md" "$DOCS_DIR/development/contributing.md"
    echo "✓ Moved developer/CONTRIBUTING.md → development/contributing.md"
fi

if [ -f "$DOCS_DIR/developer/LINTING.md" ]; then
    git mv "$DOCS_DIR/developer/LINTING.md" "$DOCS_DIR/development/code-quality.md"
    echo "✓ Moved developer/LINTING.md → development/code-quality.md"
fi

if [ -f "$DOCS_DIR/developer/TESTING.md" ]; then
    git mv "$DOCS_DIR/developer/TESTING.md" "$DOCS_DIR/development/testing.md"
    echo "✓ Moved developer/TESTING.md → development/testing.md"
fi

if [ -f "$DOCS_DIR/reference/PROJECT_ORGANIZATION.md" ]; then
    git mv "$DOCS_DIR/reference/PROJECT_ORGANIZATION.md" "$DOCS_DIR/development/project-organization.md"
    echo "✓ Moved reference/PROJECT_ORGANIZATION.md → development/project-organization.md"
fi

if [ -f "$DOCS_DIR/VERSIONING.md" ]; then
    git mv "$DOCS_DIR/VERSIONING.md" "$DOCS_DIR/development/versioning.md"
    echo "✓ Moved VERSIONING.md → development/versioning.md"
fi

# To architecture/
echo ""
echo "Moving to architecture/..."
if [ -f "$DOCS_DIR/architecture/REINDEXING_WORKFLOW.md" ]; then
    git mv "$DOCS_DIR/architecture/REINDEXING_WORKFLOW.md" "$DOCS_DIR/architecture/indexing-workflow.md"
    echo "✓ Moved architecture/REINDEXING_WORKFLOW.md → architecture/indexing-workflow.md"
fi

if [ -f "$DOCS_DIR/performance/CONNECTION_POOLING.md" ]; then
    git mv "$DOCS_DIR/performance/CONNECTION_POOLING.md" "$DOCS_DIR/architecture/performance.md"
    echo "✓ Moved performance/CONNECTION_POOLING.md → architecture/performance.md"
fi

# To internal/
echo ""
echo "Moving to internal/..."
if [ -f "$DOCS_DIR/IMPROVEMENTS_SUMMARY.md" ]; then
    git mv "$DOCS_DIR/IMPROVEMENTS_SUMMARY.md" "$DOCS_DIR/internal/improvements.md"
    echo "✓ Moved IMPROVEMENTS_SUMMARY.md → internal/improvements.md"
fi

if [ -f "$DOCS_DIR/developer/REFACTORING_ANALYSIS.md" ]; then
    git mv "$DOCS_DIR/developer/REFACTORING_ANALYSIS.md" "$DOCS_DIR/internal/refactoring-analysis.md"
    echo "✓ Moved developer/REFACTORING_ANALYSIS.md → internal/refactoring-analysis.md"
fi

if [ -f "$DOCS_DIR/reference/INSTALL_COMMAND_ENHANCEMENTS.md" ]; then
    git mv "$DOCS_DIR/reference/INSTALL_COMMAND_ENHANCEMENTS.md" "$DOCS_DIR/internal/install-enhancements.md"
    echo "✓ Moved reference/INSTALL_COMMAND_ENHANCEMENTS.md → internal/install-enhancements.md"
fi

# Archive old reference docs that are now internal
if [ -f "$DOCS_DIR/reference/ENGINEER_TASK.md" ]; then
    git mv "$DOCS_DIR/reference/ENGINEER_TASK.md" "$DOCS_DIR/_archive/ENGINEER_TASK.md"
    echo "✓ Archived reference/ENGINEER_TASK.md → _archive/"
fi

echo ""
echo "================================================"
echo "Phase 2: Cleanup Empty Directories"
echo "================================================"
echo ""

# Remove empty directories (if empty)
for dir in "$DOCS_DIR/developer" "$DOCS_DIR/performance" "$DOCS_DIR/analysis" "$DOCS_DIR/debugging" "$DOCS_DIR/technical" "$DOCS_DIR/optimizations"; do
    if [ -d "$dir" ] && [ -z "$(ls -A $dir)" ]; then
        rmdir "$dir"
        echo "✓ Removed empty directory: $(basename $dir)/"
    fi
done

echo ""
echo "================================================"
echo "Phase 3: Summary"
echo "================================================"
echo ""
echo "Files have been reorganized successfully!"
echo ""
echo "Next steps:"
echo "1. Create consolidated documentation files (installation, testing, etc.)"
echo "2. Create new content files (guides, advanced topics)"
echo "3. Update all internal links in moved files"
echo "4. Verify all links work"
echo "5. Update README.md and CLAUDE.md"
echo "6. Commit changes with: git commit -m 'docs: reorganize documentation structure'"
echo ""
echo "See docs/REORGANIZATION_SUMMARY.md for complete details."
echo ""
