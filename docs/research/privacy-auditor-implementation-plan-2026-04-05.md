# Privacy-Policy-vs-Code Audit Tool — Implementation Plan

**Repo:** mcp-vector-search
**Date:** 2026-04-05
**Status:** Approved — M1 in progress

## User-Locked Decisions

1. **Trigger**: On-demand only (CLI + GitHub Action workflow_dispatch)
2. **Evidence threshold**: PASS requires 2+ snippets AND knowledge graph path
3. **Ignore mechanism**: `.audit-ignore.yml` in target repo with justifications (min 20 chars)
4. **Drift detection**: Hash policy + code commit SHA, re-audit when either changes
5. **Manual review**: Auto-create GitHub issues for MANUAL_REVIEW / INSUFFICIENT_EVIDENCE
6. **Certification storage**: Same repo (this one), `certifications/` at root, public for transparency
7. **GPG key**: Dev key for M2 local, CI key deferred to M3
8. **Model choice**: Haiku for claim extraction, Opus for judge
9. **Target repo auth**: Local filesystem (M1-M2), GH App deferred
10. **Ignore file location**: Target repo only (user-controlled)
11. **Auto-indexing**: Runner ensures index+KG exist, auto-runs if missing (M2+)
12. **Compound claims**: Split atomically during extraction
13. **Overall status**: Any FAIL → FAILED; MANUAL_REVIEW/INSUFFICIENT only → CERTIFIED_WITH_EXCEPTIONS; all PASS → CERTIFIED

## File Layout

```
src/mcp_vector_search/auditor/
├── __init__.py
├── models.py
├── policy_extractor.py
├── claim_router.py
├── evidence_collector.py
├── judge.py
├── certifier.py
├── issue_creator.py        # M4
├── drift.py                # M5
├── ignore.py
├── config.py
├── prompts/
│   ├── extract_claims.md
│   └── judge_verdict.md
├── strategies/
│   ├── data_sharing.yaml
│   ├── encryption.yaml
│   ├── retention.yaml
│   ├── user_rights.yaml
│   ├── third_party.yaml
│   └── logging_pii.yaml
└── runner.py

src/mcp_vector_search/cli/commands/audit.py
.github/workflows/privacy-audit.yml    # M3
certifications/                        # M2+
```

## Milestones

**M1 — CLI prototype (MD output, no signing/GH)** — 9 tasks
**M2 — Signing + JSON sidecar + cert repo structure** — 3 tasks
**M3 — GitHub Action wrapper** — 2 tasks
**M4 — GH issue creation for manual review** — 2 tasks
**M5 — Drift detection + scheduled re-audit** — 3 tasks

## Claim-Category → Query Strategy

| Category | Tools + Patterns |
|---|---|
| data_sharing | search_hybrid + kg_query (outgoing HTTP), grep for requests.post/fetch/axios |
| encryption | search_code (AES/crypto/TLS/bcrypt/argon2), find_smells(crypto), kg_query |
| retention | search_hybrid (delete/purge/ttl/expire), kg_callers_at_commit, DELETE FROM |
| user_rights | search_hybrid (GDPR/DSAR/export), kg_query for /users/*/export endpoints |
| third_party | search_code (analytics/segment/amplitude), parse package.json/pyproject.toml |
| logging_pii | search_hybrid (log PII redact mask), find_smells(logging), grep logger.info |

## Certification Directory Structure (M2+)

```
certifications/
├── README.md
├── index.json
├── audit-log.jsonl
├── <target-slug>/
│   ├── latest -> <timestamp>
│   └── <YYYYMMDD-HHMMSS>/
│       ├── certification.md
│       ├── certification.json
│       ├── certification.json.sig
│       ├── policy-snapshot.md
│       ├── policy-snapshot.sha256
│       ├── evidence/
│       │   └── <claim_id>.json
│       └── manifest.json
```

## Overall Status Logic

```
if any(v.status == "FAIL" for v in verdicts):
    overall = "FAILED"
elif any(v.status in ("MANUAL_REVIEW","INSUFFICIENT_EVIDENCE") for v in verdicts):
    overall = "CERTIFIED_WITH_EXCEPTIONS"
else:
    overall = "CERTIFIED"
```

## Testing Strategy

- Unit tests under tests/unit/auditor/ with mocked LLM
- Integration tests under tests/integration/auditor/ with fixture repo + VCR
- Golden snapshot for MD rendering (exclude timestamps)
- Manual test target: ~/Projects/tripbot7 (already indexed, has privacy.md)
  - Known nuances: Vercel Analytics undisclosed, minimum-necessary vs full itinerary to AI

## Open Questions — Resolved

All 8 open questions resolved per user decisions above.
