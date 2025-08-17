#!/bin/bash
# Local deployment testing script for Stage B

set -e

echo "📦 Stage B: Local Deployment Testing"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "🏗️ Building package..."
uv run python -m build || {
    echo "❌ Build failed. Please check your code."
    exit 1
}

echo "🔍 Checking package..."
uv run twine check dist/* || {
    echo "❌ Package check failed. Please fix package issues."
    exit 1
}

echo "📦 Installing clean version locally..."
pip uninstall mcp-vector-search -y 2>/dev/null || true

# Find the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "❌ No wheel file found in dist/"
    exit 1
fi

pip install "$WHEEL_FILE" || {
    echo "❌ Installation failed."
    exit 1
}

echo "🧪 Testing installed version..."
mcp-vector-search version || {
    echo "❌ Version command failed."
    exit 1
}

mcp-vector-search --help > /dev/null || {
    echo "❌ Help command failed."
    exit 1
}

echo "✅ Local deployment test passed!"
echo ""
echo "🎯 Manual testing checklist:"
echo "   1. cd to another project directory"
echo "   2. Run: mcp-vector-search init"
echo "   3. Run: mcp-vector-search index"
echo "   4. Run: mcp-vector-search search 'some query'"
echo "   5. Verify results are reasonable"
echo ""
echo "When manual testing is complete, run: ./scripts/publish.sh"
