#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/ci/run_audit.sh <target-repo-path> <policy-path> [output-dir]
#
# Required environment variables:
#   OPENROUTER_API_KEY or ANTHROPIC_API_KEY
#
# Optional environment variables:
#   MVS_AUDIT_GPG_KEY_ID — GPG key for signing
#   MVS_AUDIT_REQUIRE_KG_PATH — require KG evidence (default: false)
#   MVS_AUDIT_LLM_BACKEND — anthropic or openrouter (default: auto-detect)

TARGET="${1:?Usage: run_audit.sh <target-repo> <policy-path> [output-dir]}"
POLICY="${2:?Usage: run_audit.sh <target-repo> <policy-path> [output-dir]}"
OUTPUT_DIR="${3:-audits}"

echo "=== Privacy Policy Audit ==="
echo "Target:  $TARGET"
echo "Policy:  $POLICY"
echo "Output:  $OUTPUT_DIR"
echo ""

# Ensure target is indexed
echo "Step 1: Indexing target repository..."
mvs index --project-root "$TARGET" || echo "Warning: Indexing failed, continuing with existing index"

# Run audit
echo "Step 2: Running privacy audit..."
mvs audit run \
  --target "$TARGET" \
  --policy "$POLICY" \
  --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

# Report
echo ""
echo "=== Audit Complete ==="
case $EXIT_CODE in
  0) echo "Status: CERTIFIED" ;;
  1) echo "Status: FAILED" ;;
  2) echo "Status: CERTIFIED WITH EXCEPTIONS" ;;
  *) echo "Status: ERROR (exit code $EXIT_CODE)" ;;
esac

# Verify
LATEST=$(find "$OUTPUT_DIR" -name "certification.md" -type f | sort | tail -1)
if [ -f "$LATEST" ]; then
  echo "Certification: $LATEST"
  mvs audit verify "$(dirname "$LATEST")" && echo "Verification: PASSED" || echo "Verification: FAILED"
fi

exit $EXIT_CODE
