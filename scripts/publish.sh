#!/bin/bash
# ============================================================================
# DEPRECATION NOTICE: This script is deprecated as of v4.0.3
# Please use the Makefile instead:
#   make publish        # Publish to PyPI
#   make publish-test   # Publish to test PyPI
#   make release-patch  # Full release workflow including publishing
# 
# This script will be removed in v5.0.0
# ============================================================================

# PyPI publication script for Stage C

set -e

echo "============================================================================"
echo "⚠️  DEPRECATION WARNING: This script is deprecated!"
echo "   Please use: make publish"
echo "============================================================================"
echo ""
echo "🚀 Stage C: PyPI Publication"
echo "============================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from mcp_vector_search import __version__; print(__version__)")
echo "📋 Current version: $CURRENT_VERSION"

# Confirm publication
echo ""
echo "⚠️  You are about to publish version $CURRENT_VERSION to PyPI"
echo "   This action cannot be undone!"
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Publication cancelled."
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

echo "📤 Uploading to PyPI..."
uv run twine upload dist/* || {
    echo "❌ Upload failed. Please check your credentials and network."
    exit 1
}

echo "✅ Successfully published version $CURRENT_VERSION to PyPI!"
echo ""
echo "🎯 Post-publication checklist:"
echo "   1. Wait 2-3 minutes for PyPI to process"
echo "   2. Test installation: pip install mcp-vector-search==$CURRENT_VERSION"
echo "   3. Create GitHub release: git tag v$CURRENT_VERSION && git push --tags"
echo "   4. Update documentation if needed"
echo "   5. Announce the release!"
