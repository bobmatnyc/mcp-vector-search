#!/bin/bash
# Demo script for non-blocking KG build feature
#
# This script demonstrates:
# 1. Enabling background KG build
# 2. Indexing project (search available immediately)
# 3. Checking status while KG builds
# 4. Verifying search works without waiting

set -e

echo "============================================================"
echo "Non-Blocking KG Build Demo"
echo "============================================================"
echo ""

# Step 1: Enable background KG build
echo "Step 1: Enabling background KG build..."
export MCP_VECTOR_SEARCH_AUTO_KG=true
echo "✓ Set MCP_VECTOR_SEARCH_AUTO_KG=true"
echo ""

# Step 2: Run indexing (will trigger background KG)
echo "Step 2: Indexing project..."
echo "(Search will be available immediately after indexing)"
echo ""
time mcp-vector-search index
echo ""
echo "✓ Indexing complete! Search is now available."
echo ""

# Step 3: Check status (should show KG building)
echo "Step 3: Checking status..."
mcp-vector-search status | grep -A2 "Knowledge Graph"
echo ""

# Step 4: Test search (should work immediately)
echo "Step 4: Testing search (without waiting for KG)..."
mcp-vector-search search "async search" --limit 3
echo ""
echo "✓ Search works! No need to wait for KG."
echo ""

# Step 5: Wait a bit and check status again
echo "Step 5: Waiting 10 seconds for KG build..."
sleep 10
echo ""
echo "Checking status again..."
mcp-vector-search status | grep -A2 "Knowledge Graph"
echo ""

echo "============================================================"
echo "Demo Complete!"
echo "============================================================"
echo ""
echo "Key Takeaways:"
echo "  1. Search is available immediately after indexing"
echo "  2. KG builds in background without blocking"
echo "  3. Status shows KG build progress"
echo "  4. Search works with or without KG"
echo ""
echo "To disable background KG build:"
echo "  unset MCP_VECTOR_SEARCH_AUTO_KG"
echo ""
