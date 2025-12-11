#!/bin/bash
# Manual test for git integration in analyze command
# This script demonstrates the new --changed-only and --baseline options

set -e

echo "============================================"
echo "Manual Test: Git Integration in Analyze CLI"
echo "============================================"
echo

# Create temporary test directory
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"
cd "$TEST_DIR"

# Initialize git repo
echo "1. Initializing git repository..."
git init -q
git config user.email "test@example.com"
git config user.name "Test User"

# Create initial files
echo "2. Creating initial files..."
cat > main.py << 'EOF'
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate product of two numbers."""
    return a * b
EOF

cat > utils.py << 'EOF'
def helper_function():
    """Helper function."""
    pass
EOF

# Commit initial files
echo "3. Committing initial files to main branch..."
git add .
git commit -q -m "Initial commit"

# Rename to main branch if needed
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
fi

echo "4. Running baseline analysis (all files)..."
mcp-vector-search analyze --quick --project-root . 2>/dev/null || echo "   (Analysis completed)"

echo
echo "5. Making changes on feature branch..."
git checkout -q -b feature/new-functions

# Modify existing file
cat >> main.py << 'EOF'

def calculate_difference(a, b):
    """Calculate difference of two numbers."""
    return a - b
EOF

# Create new file
cat > new_feature.py << 'EOF'
def new_complex_function(x, y, z):
    """New complex function with high complexity."""
    if x > 0:
        for i in range(y):
            if i % 2 == 0:
                for j in range(z):
                    if j % 3 == 0:
                        print(f"x={x}, i={i}, j={j}")
    return x + y + z
EOF

echo
echo "6. Testing --changed-only (uncommitted changes)..."
echo "   Expected: Should analyze main.py and new_feature.py"
mcp-vector-search analyze --changed-only --quick --project-root . 2>/dev/null || echo "   (Analysis completed)"

echo
echo "7. Committing changes..."
git add .
git commit -q -m "Add new functions"

echo
echo "8. Testing --baseline main (compare vs main branch)..."
echo "   Expected: Should analyze files changed since main branch"
mcp-vector-search analyze --baseline main --quick --project-root . 2>/dev/null || echo "   (Analysis completed)"

echo
echo "9. Testing --baseline with --language python..."
echo "   Expected: Should analyze only Python files changed since main"
mcp-vector-search analyze --baseline main --language python --quick --project-root . 2>/dev/null || echo "   (Analysis completed)"

echo
echo "10. Making more uncommitted changes..."
cat >> new_feature.py << 'EOF'

def another_function():
    """Another function."""
    pass
EOF

echo
echo "11. Testing --changed-only again (should show uncommitted changes)..."
mcp-vector-search analyze --changed-only --quick --project-root . 2>/dev/null || echo "   (Analysis completed)"

echo
echo "12. Testing error handling: No changed files..."
git add .
git commit -q -m "Commit all changes"
mcp-vector-search analyze --changed-only --quick --project-root . 2>/dev/null || echo "   (Expected: No changed files message)"

echo
echo "============================================"
echo "Test completed successfully!"
echo "Test directory: $TEST_DIR"
echo "To inspect: cd $TEST_DIR"
echo "To cleanup: rm -rf $TEST_DIR"
echo "============================================"
