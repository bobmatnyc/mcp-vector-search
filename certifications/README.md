# Privacy Certifications

Audit artifacts from the mcp-vector-search privacy auditor.

## Structure

- `<target-slug>/<YYYYMMDD-HHMMSS>/` — one directory per audit run
- `certification.md` — human-readable report
- `certification.json` — machine-readable full data
- `certification.json.sig` — GPG detached signature (if signed)
- `policy-snapshot.md` — policy text at audit time
- `policy-snapshot.sha256` — SHA-256 of the policy snapshot
- `evidence/<claim_id>.json` — per-claim evidence
- `manifest.json` — SHA-256 hashes of all files

## Verification

```
mvs audit verify certifications/<target>/<timestamp>/
```
