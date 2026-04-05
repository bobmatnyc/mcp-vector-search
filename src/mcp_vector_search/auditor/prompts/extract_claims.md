# Extract Privacy Policy Claims

You are a privacy-policy analyst. Your task is to extract atomic, testable claims from a privacy policy document so that each claim can be independently verified against source code.

## Instructions

1. **Atomize compound claims**: Split claims that cover multiple requirements.
   - Example: "We encrypt data in transit and at rest" → two separate claims.
2. **Normalize each claim** into a concrete, testable statement starting with a verb.
   - Example: "All network communications use TLS 1.2 or higher."
3. **Identify the policy section** where the claim appears (e.g., "Data Security", "Retention").
4. **Extract keywords** relevant to code search (function names, library names, protocol names).
5. **Mark non-testable claims**: If a claim cannot be verified via code (e.g., "We train our staff on privacy"), set `testable: false`.
6. **Capture source offsets**: Record the character start and end positions in the original text.
7. **Limit output**: Extract at most {{ max_claims }} claims. Prioritize the most security-critical.

## Categories

Assign each claim exactly one category:
- `data_sharing` — sending data to third parties or external services
- `encryption` — data encryption at rest or in transit
- `retention` — data retention periods and deletion
- `user_rights` — GDPR/CCPA user rights (access, deletion, portability)
- `third_party` — third-party SDKs, analytics, or integrations
- `logging_pii` — PII in logs (or commitment to exclude it)
- `consent` — consent mechanisms and opt-in/opt-out
- `security` — general security controls (access control, auditing)
- `access_control` — authentication, authorization, least-privilege

## Output Format

Return a JSON array of claim objects matching this schema:

```json
[
  {
    "category": "encryption",
    "text": "verbatim excerpt from policy",
    "normalized": "All API endpoints use TLS 1.2 or higher for data in transit.",
    "keywords": ["TLS", "HTTPS", "ssl", "encrypt"],
    "policy_section": "Data Security",
    "testable": true,
    "source_offsets": [1234, 1456]
  }
]
```

## Policy Text

{{ policy_text }}
