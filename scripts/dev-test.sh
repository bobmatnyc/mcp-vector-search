#!/bin/bash
# Development testing script for Stage A

set -e

echo "🧪 Stage A: Development Testing"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

echo "📋 Running pre-commit checks..."
uv run pre-commit run --all-files || {
    echo "❌ Pre-commit checks failed. Please fix issues and try again."
    exit 1
}

echo "🧪 Running tests..."
uv run pytest -v --cov=src/mcp_vector_search || {
    echo "❌ Tests failed. Please fix failing tests."
    exit 1
}

echo "🔍 Running type checks..."
uv run mypy src/ || {
    echo "❌ Type checking failed. Please fix type issues."
    exit 1
}

echo "🚀 Testing CLI functionality..."
uv run mcp-vector-search version || {
    echo "❌ CLI version command failed."
    exit 1
}

uv run mcp-vector-search --help > /dev/null || {
    echo "❌ CLI help command failed."
    exit 1
}

echo "✅ All Stage A tests passed!"
echo ""
echo "🎯 Next steps:"
echo "   1. Test your changes manually with: uv run mcp-vector-search"
echo "   2. When ready, run: ./scripts/deploy-test.sh"
echo "   3. Finally publish with: ./scripts/publish.sh"
