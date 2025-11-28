#!/bin/bash
# Post-installation verification script for mcp-vector-search
# This script verifies that all dependencies are correctly installed

set -e

echo "🔍 Verifying MCP Vector Search Installation..."
echo

# Check if mcp-vector-search is available
if ! command -v mcp-vector-search &> /dev/null; then
    echo "❌ mcp-vector-search command not found"
    echo "   Please install: pip install mcp-vector-search"
    exit 1
fi

echo "✓ mcp-vector-search command found"

# Run the doctor command
echo
echo "Running dependency check..."
echo

if mcp-vector-search doctor; then
    echo
    echo "✅ Installation verified successfully!"
    echo
    echo "Next steps:"
    echo "  1. Navigate to your project directory"
    echo "  2. Run: mcp-vector-search setup"
    echo "  3. Start searching: mcp-vector-search search \"your query\""
    echo
else
    echo
    echo "⚠️  Some dependencies are missing"
    echo
    echo "Try reinstalling:"
    echo "  pip install --upgrade --force-reinstall mcp-vector-search"
    echo
    exit 1
fi
