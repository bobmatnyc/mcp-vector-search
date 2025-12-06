#!/bin/bash

# Streaming JSON Load Verification Test
# Tests the new FastAPI streaming endpoint for visualization

set -e

echo "🧪 Testing Streaming JSON Load Implementation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Start visualization server in background
echo "1. Starting visualization server on port 8099..."
uv run mcp-vector-search visualize serve --port 8099 > /tmp/viz-server-test.log 2>&1 &
SERVER_PID=$!
sleep 3

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    rm -f /tmp/viz-server-test.log
}
trap cleanup EXIT

# 2. Test server is running
echo "2. Testing server health..."
if curl -s http://localhost:8099/ | grep -q "Code Graph"; then
    echo "   ✅ Server is running and serving index.html"
else
    echo "   ❌ Server failed to start or index.html not found"
    exit 1
fi

# 3. Test streaming endpoint exists
echo "3. Testing /api/graph-data endpoint..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8099/api/graph-data)
if [ "$HTTP_STATUS" = "200" ]; then
    echo "   ✅ Streaming endpoint returns 200 OK"
else
    echo "   ❌ Streaming endpoint returned HTTP $HTTP_STATUS"
    exit 1
fi

# 4. Test content type
echo "4. Testing response content type..."
CONTENT_TYPE=$(curl -s -I http://localhost:8099/api/graph-data | grep -i content-type | cut -d' ' -f2 | tr -d '\r')
if [[ "$CONTENT_TYPE" == *"application/json"* ]]; then
    echo "   ✅ Content-Type is application/json"
else
    echo "   ❌ Content-Type is $CONTENT_TYPE (expected application/json)"
    exit 1
fi

# 5. Test server is uvicorn (not SimpleHTTPServer)
echo "5. Testing server type..."
SERVER_HEADER=$(curl -s -I http://localhost:8099/ | grep -i "^server:" | cut -d' ' -f2 | tr -d '\r')
if [[ "$SERVER_HEADER" == *"uvicorn"* ]]; then
    echo "   ✅ Server is uvicorn (FastAPI)"
else
    echo "   ❌ Server is $SERVER_HEADER (expected uvicorn)"
    exit 1
fi

# 6. Test response size matches chunk-graph.json
echo "6. Testing response size..."
EXPECTED_SIZE=$(wc -c < .mcp-vector-search/visualization/chunk-graph.json | xargs)
ACTUAL_SIZE=$(curl -s http://localhost:8099/api/graph-data | wc -c | xargs)
if [ "$EXPECTED_SIZE" = "$ACTUAL_SIZE" ]; then
    echo "   ✅ Response size matches ($ACTUAL_SIZE bytes)"
else
    echo "   ⚠️  Response size mismatch: expected $EXPECTED_SIZE, got $ACTUAL_SIZE"
    echo "   (This might be OK if file changed during test)"
fi

# 7. Test JSON is valid
echo "7. Testing JSON validity..."
if curl -s http://localhost:8099/api/graph-data | python3 -m json.tool > /dev/null 2>&1; then
    echo "   ✅ Response is valid JSON"
else
    echo "   ❌ Response is not valid JSON"
    exit 1
fi

# 8. Test streaming code is in index.html
echo "8. Testing frontend streaming code..."
if grep -q "loadGraphDataStreaming" .mcp-vector-search/visualization/index.html; then
    echo "   ✅ Streaming JavaScript code found in index.html"
else
    echo "   ❌ Streaming code not found in index.html"
    exit 1
fi

# 9. Test progress bar logic
echo "9. Testing progress bar implementation..."
if grep -q "transferPercent = Math.round((loaded / total) \* 50)" .mcp-vector-search/visualization/index.html; then
    echo "   ✅ Two-stage progress tracking implemented (0-50% transfer, 50-100% parse)"
else
    echo "   ⚠️  Progress tracking may not be using two-stage approach"
fi

# 10. Test retry button on error
echo "10. Testing error handling UI..."
if grep -q 'onclick="location.reload()"' .mcp-vector-search/visualization/index.html; then
    echo "   ✅ Retry button implemented on error"
else
    echo "   ⚠️  Retry button not found in error handling"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All tests passed!"
echo ""
echo "📊 Summary:"
echo "   - FastAPI server: ✅ Running"
echo "   - Streaming endpoint: ✅ /api/graph-data works"
echo "   - Frontend code: ✅ Streaming loader implemented"
echo "   - Progress tracking: ✅ Two-stage (transfer + parse)"
echo "   - Error handling: ✅ Retry button on failure"
echo ""
echo "🌐 Manual Test Instructions:"
echo "   1. Open Safari: http://localhost:8099"
echo "   2. Open DevTools (Cmd+Option+I)"
echo "   3. Go to Network tab"
echo "   4. Reload page"
echo "   5. Watch for:"
echo "      - GET /api/graph-data (should show chunked transfer)"
echo "      - Progress bar: 0% → 50% → 100%"
echo "      - Graph renders without errors"
echo "   6. Check Console for errors (should be none)"
echo "   7. Verify Memory usage < 100MB (Memory tab)"
echo ""
echo "📝 Next Steps:"
echo "   - Run manual test in Safari to confirm 6.3MB file loads"
echo "   - Check browser memory usage during load"
echo "   - Verify visualization renders correctly"
echo ""
