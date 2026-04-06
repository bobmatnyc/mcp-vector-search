# Privacy Policy Auditor

Audit codebases against stated privacy policies using semantic code search and knowledge graph analysis.

## Quick Start

### CLI (Local)

```bash
# Install auditor dependencies
pip install 'mcp-vector-search[auditor]'

# Run audit
mvs audit run --target /path/to/repo --policy /path/to/repo/PRIVACY.md

# Verify certification
mvs audit verify certifications/<target>/latest/

# List audit history
mvs audit list
```

### GitHub Action (CI)

Add to your repository or trigger manually from the Actions tab:

1. Go to **Actions** → **Privacy Policy Audit**
2. Click **Run workflow**
3. Fill in:
   - **Target repository**: URL or path to the repo to audit
   - **Policy path**: Relative path to the privacy policy (default: `PRIVACY.md`)
   - **Reviewer**: GitHub username for manual review assignments
   - **Require KG path**: Whether to require knowledge graph evidence
   - **LLM backend**: `openrouter` or `anthropic`

### Required Secrets

Configure these in your repository Settings → Secrets:
- `OPENROUTER_API_KEY` — OpenRouter API key (recommended)
- `ANTHROPIC_API_KEY` — Anthropic API key (alternative)
- `AUDIT_GPG_KEY_ID` — GPG key ID for signing (optional)

## How It Works

1. **Extract Claims**: Parses the privacy policy into testable assertions using text analysis + LLM
2. **Collect Evidence**: Queries the codebase using vector search, hybrid search, and knowledge graph
3. **Judge Verdicts**: LLM evaluates each claim against collected evidence
4. **Certify**: Produces a signed certification document with per-claim verdicts

## Verdict Types

| Verdict | Meaning |
|---------|---------|
| **PASS** | Claim supported by 2+ evidence items |
| **FAIL** | Evidence contradicts the claim |
| **INSUFFICIENT EVIDENCE** | Not enough evidence to verify |
| **MANUAL REVIEW** | Requires human assessment |

## Overall Status

| Status | Condition | Exit Code |
|--------|-----------|-----------|
| CERTIFIED | All claims PASS | 0 |
| CERTIFIED WITH EXCEPTIONS | No FAILs, some INSUFFICIENT/MANUAL | 2 |
| FAILED | Any claim FAILs | 1 |

## Certification Output

Each audit produces a directory with:
- `certification.md` — Human-readable report
- `certification.json` — Machine-readable full data
- `certification.json.sig` — GPG signature (if configured)
- `evidence/<claim_id>.json` — Per-claim evidence
- `manifest.json` — SHA-256 integrity hashes
- `policy-snapshot.md` — Policy text at audit time

## Configuration

Environment variables (prefix: `MVS_AUDIT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MVS_AUDIT_ANTHROPIC_API_KEY` | from `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `MVS_AUDIT_LLM_BACKEND` | auto-detect | `anthropic` or `openrouter` |
| `MVS_AUDIT_EXTRACTOR_MODEL` | `claude-haiku-4-5` | Model for claim extraction |
| `MVS_AUDIT_JUDGE_MODEL` | `claude-opus-4-6` | Model for verdict reasoning |
| `MVS_AUDIT_REQUIRE_KG_PATH` | `true` | Require KG evidence for PASS |
| `MVS_AUDIT_MIN_EVIDENCE_COUNT` | `2` | Minimum evidence items for PASS |
| `MVS_AUDIT_GPG_KEY_ID` | — | GPG key for signing |

## .audit-ignore.yml

Place in the target repo root to suppress specific claims:

```yaml
- claim_id: "a47dd0f37640"
  justification: "Google OAuth data collection is documented in our separate data processing agreement"
  approved_by: "privacy-team"
  expires: "2027-01-01"

- category: "retention"
  justification: "Retention policies are managed at the infrastructure level via Vercel Blob TTL"
  approved_by: "engineering-lead"
```
