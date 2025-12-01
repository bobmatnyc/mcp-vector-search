#!/bin/bash
# Wait for PyPI package availability and update Homebrew formula
# This script implements retry logic with exponential backoff

set -e

VERSION="${1:-0.14.3}"
MAX_ATTEMPTS=10
TIMEOUT=300  # 5 minutes total timeout
START_TIME=$(date +%s)

echo "=========================================="
echo "Homebrew Tap Update - Version $VERSION"
echo "=========================================="
echo ""
echo "This is a NON-BLOCKING operation"
echo "Will retry up to $MAX_ATTEMPTS times with exponential backoff"
echo "Total timeout: $TIMEOUT seconds"
echo ""

# Function to check if package is available on PyPI
check_pypi_availability() {
    local version=$1
    local url="https://pypi.org/pypi/mcp-vector-search/$version/json"

    if curl -s -f -o /dev/null "$url"; then
        return 0
    else
        return 1
    fi
}

# Function to calculate elapsed time
get_elapsed_time() {
    local current_time=$(date +%s)
    echo $((current_time - START_TIME))
}

# Retry loop with exponential backoff
attempt=1
wait_time=5

while [ $attempt -le $MAX_ATTEMPTS ]; do
    elapsed=$(get_elapsed_time)

    if [ $elapsed -ge $TIMEOUT ]; then
        echo "⏱️  Timeout reached after ${elapsed}s"
        break
    fi

    echo "Attempt $attempt/$MAX_ATTEMPTS: Checking if version $VERSION is available on PyPI..."

    if check_pypi_availability "$VERSION"; then
        echo "✓ Version $VERSION found on PyPI!"
        echo ""
        echo "Proceeding with Homebrew formula update..."
        echo ""

        # Check if HOMEBREW_TAP_TOKEN is set
        if [ -z "$HOMEBREW_TAP_TOKEN" ]; then
            echo "⚠️  WARNING: HOMEBREW_TAP_TOKEN not set"
            echo ""
            echo "Manual Fallback Instructions:"
            echo "=============================="
            echo "1. Export your GitHub token:"
            echo "   export HOMEBREW_TAP_TOKEN=<your-github-token>"
            echo ""
            echo "2. Run the update script:"
            echo "   python3 scripts/update_homebrew_formula.py --version $VERSION --verbose"
            echo ""
            echo "3. Or update manually:"
            echo "   cd /path/to/homebrew-mcp-vector-search"
            echo "   # Edit Formula/mcp-vector-search.rb"
            echo "   # Update version and sha256"
            echo "   git add Formula/mcp-vector-search.rb"
            echo "   git commit -m 'chore: update formula to $VERSION'"
            echo "   git push origin main"
            echo ""
            exit 1
        fi

        # Run the actual update
        if python3 scripts/update_homebrew_formula.py --version "$VERSION" --verbose; then
            echo ""
            echo "=========================================="
            echo "✓ Homebrew formula updated successfully!"
            echo "=========================================="
            exit 0
        else
            exit_code=$?
            echo ""
            echo "=========================================="
            echo "✗ Homebrew formula update failed"
            echo "=========================================="
            echo ""
            echo "Exit code: $exit_code"
            echo ""
            echo "Manual Fallback Instructions:"
            echo "=============================="
            echo "1. Check the error above and resolve any issues"
            echo "2. Run manually:"
            echo "   python3 scripts/update_homebrew_formula.py --version $VERSION --verbose"
            echo ""
            exit $exit_code
        fi
    fi

    # Package not available yet, wait with exponential backoff
    echo "✗ Version $VERSION not yet available on PyPI"
    echo "⏱️  Waiting ${wait_time}s before next attempt... (elapsed: ${elapsed}s/${TIMEOUT}s)"
    echo ""

    sleep $wait_time

    # Exponential backoff: 5, 10, 20, 40, 60, 60, 60...
    wait_time=$((wait_time * 2))
    if [ $wait_time -gt 60 ]; then
        wait_time=60
    fi

    attempt=$((attempt + 1))
done

# If we get here, we've exhausted retries or timed out
elapsed=$(get_elapsed_time)
echo "=========================================="
echo "⚠️  Could not update Homebrew formula"
echo "=========================================="
echo ""
echo "Version $VERSION is not yet available on PyPI after $attempt attempts (${elapsed}s)"
echo ""
echo "This is a NON-BLOCKING failure - continuing with release process"
echo ""
echo "Manual Fallback Instructions:"
echo "=============================="
echo "1. Wait for PyPI to process the package (may take 5-10 minutes)"
echo "2. Check availability: https://pypi.org/project/mcp-vector-search/$VERSION/"
echo "3. Once available, run:"
echo "   export HOMEBREW_TAP_TOKEN=<your-github-token>"
echo "   python3 scripts/update_homebrew_formula.py --version $VERSION --verbose"
echo ""
echo "4. Or update manually:"
echo "   cd \$(mktemp -d)"
echo "   git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git"
echo "   cd homebrew-mcp-vector-search"
echo ""
echo "   # Get SHA256 from PyPI"
echo "   PYPI_SHA256=\$(curl -s https://pypi.org/pypi/mcp-vector-search/$VERSION/json | python3 -c \"import sys, json; data = json.load(sys.stdin); sdist = [r for r in data['urls'] if r['packagetype'] == 'sdist'][0]; print(sdist['digests']['sha256'])\")"
echo ""
echo "   # Update formula"
echo "   sed -i '' \"s/version \\\"[^\\\"]*\\\"/version \\\"$VERSION\\\"/g\" Formula/mcp-vector-search.rb"
echo "   sed -i '' \"s/sha256 \\\"[^\\\"]*\\\"/sha256 \\\"\$PYPI_SHA256\\\"/g\" Formula/mcp-vector-search.rb"
echo ""
echo "   # Commit and push"
echo "   git add Formula/mcp-vector-search.rb"
echo "   git commit -m \"chore: update formula to $VERSION\""
echo "   git push origin main"
echo ""

# Exit with warning code (non-zero but distinguishable from hard failure)
exit 2
