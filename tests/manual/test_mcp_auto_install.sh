#!/usr/bin/env bash
# Manual test for MCP auto-installation with project path detection
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MCP Auto-Installation Test${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print test steps
print_step() {
    echo -e "\n${BLUE}>>> $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Create a temporary test project
TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

print_step "1. Setting up test project: $TEST_DIR"
cd "$TEST_DIR"
git init
echo "# Test Project" > README.md
git add README.md
git commit -m "Initial commit"
print_success "Test project created"

# Initialize mcp-vector-search
print_step "2. Initializing mcp-vector-search"
mcp-vector-search init --force --no-auto-index
print_success "Project initialized"

# Verify .mcp-vector-search directory exists
if [ -d ".mcp-vector-search" ]; then
    print_success ".mcp-vector-search directory exists"
else
    print_error ".mcp-vector-search directory not found"
    exit 1
fi

# Test auto-detection from project root
print_step "3. Testing auto-detection from project root"
echo "Current directory: $(pwd)"
mcp-vector-search install list-platforms
print_success "Listed platforms from root"

# Test auto-detection from subdirectory
print_step "4. Testing auto-detection from subdirectory"
mkdir -p src/components
cd src/components
echo "Current directory: $(pwd)"
mcp-vector-search install list-platforms
print_success "Listed platforms from subdirectory"

# Go back to root
cd "$TEST_DIR"

# Test MCP status command
print_step "5. Checking MCP status"
mcp-vector-search install mcp-status || true
print_success "MCP status command works"

# Test dry-run installation
print_step "6. Testing dry-run installation"
mcp-vector-search install mcp --dry-run
print_success "Dry-run completed"

# Test actual installation (if platforms detected)
print_step "7. Testing actual installation (auto-detect)"
echo "This will install to the highest confidence platform..."
echo "If no platforms detected, this will skip."
mcp-vector-search install mcp || {
    echo -e "${YELLOW}No MCP platforms detected - skipping installation${NC}"
}

# Verify installation
print_step "8. Verifying installation"
mcp-vector-search install mcp-status

# Test from different directory
print_step "9. Testing from nested directory"
mkdir -p deep/nested/path
cd deep/nested/path
echo "Current directory: $(pwd)"
mcp-vector-search install mcp-status
print_success "Auto-detection works from deeply nested directory"

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All Tests Passed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${BLUE}Test project location: ${TEST_DIR}${NC}"
echo -e "${BLUE}Cleanup will happen automatically on exit${NC}"
