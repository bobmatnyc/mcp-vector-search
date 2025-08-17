#!/bin/bash
# Quick workflow reference and helper

echo "🔄 MCP Vector Search - Development Workflow"
echo "==========================================="
echo ""
echo "📋 Three-Stage Development Process:"
echo ""
echo "🛠️  Stage A: Local Development & Testing"
echo "   ./scripts/dev-test.sh     - Run all development tests"
echo "   uv run mcp-vector-search  - Test CLI locally"
echo "   uv run pytest            - Run test suite"
echo ""
echo "📦 Stage B: Local Deployment Testing"
echo "   ./scripts/deploy-test.sh  - Build and test clean deployment"
echo "   cd ~/other-project        - Test on different codebase"
echo "   mcp-vector-search init    - Test initialization"
echo ""
echo "🚀 Stage C: PyPI Publication"
echo "   ./scripts/publish.sh      - Publish to PyPI"
echo "   pip install mcp-vector-search --upgrade  - Test published version"
echo ""
echo "🔧 Quick Commands:"
echo "   uv sync                   - Install dependencies"
echo "   uv run pre-commit install - Setup git hooks"
echo "   uv run pytest -v         - Run tests with verbose output"
echo "   uv run mypy src/          - Type checking"
echo "   uv run black src/ tests/  - Format code"
echo ""
echo "📖 For detailed instructions, see: DEVELOPMENT.md"
echo ""

# Check current status
if [ -f "pyproject.toml" ]; then
    CURRENT_VERSION=$(uv run python -c "from mcp_vector_search import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    echo "📋 Current version: $CURRENT_VERSION"
    
    # Check if development environment is set up
    if command -v uv &> /dev/null; then
        echo "✅ UV package manager available"
    else
        echo "❌ UV package manager not found - install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
    
    # Check if in development mode
    if uv pip show mcp-vector-search 2>/dev/null | grep -q "Editable install"; then
        echo "✅ Development mode active (editable install)"
    else
        echo "ℹ️  Not in development mode - run: uv pip install -e ."
    fi
else
    echo "❌ Not in project directory - navigate to mcp-vector-search root"
fi
