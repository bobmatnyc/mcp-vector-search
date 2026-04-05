"""Text-analysis-based privacy claim extractor — no LLM required.

Extracts PolicyClaim objects from policy text using regex pattern matching
and keyword classification.  Works entirely offline; no API key needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from .models import PolicyClaim

# ---------------------------------------------------------------------------
# Category keyword mapping
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "data_sharing": [
        "share",
        "sell",
        "disclose",
        "provide to",
        "transfer",
        "third party",
        "partner",
    ],
    "encryption": [
        "encrypt",
        "secure",
        "protect",
        "SSL",
        "TLS",
        "hash",
        "bcrypt",
    ],
    "retention": [
        "retain",
        "store",
        "keep",
        "delete",
        "purge",
        "expire",
        "TTL",
    ],
    "user_rights": [
        "request",
        "right to",
        "access your",
        "delete your",
        "export",
        "portability",
        "GDPR",
        "CCPA",
    ],
    "third_party": [
        "third-party",
        "service provider",
        "vendor",
        "processor",
        "sub-processor",
        "analytics",
        "tracking",
    ],
    "logging_pii": [
        "log",
        "record",
        "track",
        "monitor",
        "audit trail",
        "PII",
        "personal",
    ],
    "consent": [
        "consent",
        "agree",
        "opt-in",
        "opt-out",
        "permission",
        "authorize",
    ],
    "security": [
        "firewall",
        "authentication",
        "access control",
        "vulnerability",
        "penetration",
    ],
    "access_control": [
        "role",
        "permission",
        "admin",
        "authorize",
        "credential",
    ],
}

# ---------------------------------------------------------------------------
# Regex patterns for claim types
# ---------------------------------------------------------------------------

# Privacy-relevant action verbs used in compound-split detection
_PRIVACY_VERBS: re.Pattern[str] = re.compile(
    r"\b(encrypt|secure|protect|share|sell|disclose|collect|gather|receive|retain|store|"
    r"keep|delete|purge|log|record|track|monitor|provide|transfer|authorize|consent)\b",
    re.IGNORECASE,
)

# Patterns that capture the full phrase around privacy-relevant statements.
# Each tuple: (pattern, claim_type_hint)
_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Negation: "we do not/never/will not..."
    (
        re.compile(
            r"\b(?:we|our\s+(?:company|organization|service|app|application))\b"
            r"[^.]*?\b(?:do\s+not|don't|never|will\s+not|won't|does\s+not|doesn't)\b[^.]*",
            re.IGNORECASE,
        ),
        "negation",
    ),
    # Collection: "we collect/gather/receive..."
    (
        re.compile(
            r"\b(?:we|our\s+(?:company|organization|service|app|application))\b"
            r"[^.]*?\b(?:collect|gather|receive|obtain|acquire)\b[^.]*",
            re.IGNORECASE,
        ),
        "collection",
    ),
    # Sharing: "we share/disclose/provide...to..."
    (
        re.compile(
            r"\b(?:we|our\s+(?:company|organization|service|app|application))\b"
            r"[^.]*?\b(?:share|disclose|provide|sell|transfer|send)\b[^.]*",
            re.IGNORECASE,
        ),
        "sharing",
    ),
    # Security/encryption: "encrypted/secured/protected..."
    (
        re.compile(
            r"\b(?:data|information|communications?|traffic|passwords?|credentials?)\b"
            r"[^.]*?\b(?:encrypt(?:ed|ion)|secur(?:ed|ity)|protect(?:ed|ion)|"
            r"SSL|TLS|hash(?:ed)?|bcrypt)\b[^.]*",
            re.IGNORECASE,
        ),
        "security",
    ),
    # Reverse security pattern: "we encrypt/secure..."
    (
        re.compile(
            r"\b(?:we|our\s+(?:company|organization|service|app|application))\b"
            r"[^.]*?\b(?:encrypt|secur|protect|hash)\b[^.]*",
            re.IGNORECASE,
        ),
        "security",
    ),
    # Rights: "you can request/you have the right to/you may..."
    (
        re.compile(
            r"\b(?:you|users?)\b"
            r"[^.]*?\b(?:can\s+request|have\s+the\s+right\s+to|may\s+request|"
            r"are\s+entitled|can\s+access|can\s+delete|can\s+export|"
            r"have\s+the\s+ability|right\s+to\s+access|right\s+to\s+delete|"
            r"right\s+to\s+erasure|right\s+to\s+portability)\b[^.]*",
            re.IGNORECASE,
        ),
        "user_rights",
    ),
    # Retention: "we retain/store...for / deleted after..."
    (
        re.compile(
            r"\b(?:we|data|information|records?)\b"
            r"[^.]*?\b(?:retain(?:ed)?|stor(?:ed|e)|keep|kept|"
            r"delet(?:ed|ion)|purge[d]?|expir(?:es?|ation)|TTL)\b"
            r"[^.]*?\b(?:for|after|within|until|days?|months?|years?|period)\b[^.]*",
            re.IGNORECASE,
        ),
        "retention",
    ),
    # Third-party: "third-party / service providers / partners..."
    (
        re.compile(
            r"\b(?:third[- ]party|service\s+providers?|vendors?|"
            r"processors?|sub-?processors?|analytics\s+providers?|"
            r"advertising\s+partners?)\b[^.]*",
            re.IGNORECASE,
        ),
        "third_party",
    ),
    # Consent: "with your consent / you agree / opt-in / opt-out..."
    (
        re.compile(
            r"\b(?:with\s+your\s+consent|you\s+agree|opt[- ]in|opt[- ]out|"
            r"your\s+permission|you\s+authorize|explicit\s+consent)\b[^.]*",
            re.IGNORECASE,
        ),
        "consent",
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _RawMatch:
    """Intermediate representation of a regex hit before claim construction."""

    sentence: str
    hint: str
    start: int
    end: int
    confidence: float = 0.8


def _split_sections(policy_text: str) -> list[tuple[str, str]]:
    """Split policy into (section_title, section_body) tuples.

    Handles markdown headers (##, ###), numbered sections (1., 2.),
    and ALL-CAPS section headings.

    Returns a list of (title, body) pairs. Content before the first header
    is returned under the title "Preamble".
    """
    header_re = re.compile(
        r"^(?:#{1,4}\s+(.+)|(\d+(?:\.\d+)*)\.\s+(.+)|([A-Z][A-Z\s]{4,}[A-Z]))\s*$",
        re.MULTILINE,
    )

    sections: list[tuple[str, str]] = []
    last_end = 0
    last_title = "Preamble"

    for m in header_re.finditer(policy_text):
        # Save previous section body
        body = policy_text[last_end : m.start()].strip()
        if body:
            sections.append((last_title, body))

        # Determine new title
        if m.group(1):
            last_title = m.group(1).strip()
        elif m.group(3):
            last_title = m.group(3).strip()
        elif m.group(4):
            last_title = m.group(4).strip().title()
        last_end = m.end()

    # Append trailing section
    body = policy_text[last_end:].strip()
    if body:
        sections.append((last_title, body))

    return sections or [("Preamble", policy_text)]


def _extract_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences, returning (sentence, start, end) tuples."""
    # Split on sentence-ending punctuation followed by whitespace
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    sentences: list[tuple[str, int, int]] = []
    prev = 0
    for m in sentence_re.finditer(text):
        sentence = text[prev : m.start()].strip()
        if sentence:
            sentences.append((sentence, prev, m.start()))
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        sentences.append((tail, prev, len(text)))
    return sentences


def _classify_category(sentence: str) -> tuple[str, list[str]]:
    """Classify a sentence into a category by keyword matching.

    Returns (best_category, matched_keywords).
    Falls back to "security" if no category matches.
    """
    sentence_lower = sentence.lower()
    best_category = "security"
    best_count = 0
    best_keywords: list[str] = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        matched = [kw for kw in keywords if kw.lower() in sentence_lower]
        if len(matched) > best_count:
            best_count = len(matched)
            best_category = category
            best_keywords = matched

    return best_category, best_keywords


def _compute_confidence(sentence: str, hint: str, keywords: list[str]) -> float:
    """Assign an extraction confidence score.

    Rules:
    - Negation ("do not", "never") + keywords → 0.9
    - Clear assertion with 2+ keywords → 0.85
    - Clear assertion with 1 keyword → 0.75
    - Ambiguous/no keywords → 0.55
    """
    lower = sentence.lower()
    is_negation = bool(
        re.search(
            r"\b(?:do\s+not|don't|never|will\s+not|won't|does\s+not|doesn't)\b",
            lower,
        )
    )
    is_clear = bool(
        re.search(
            r"\b(?:we|our|you|users?|data|information)\b",
            lower,
        )
    )

    if is_negation and keywords:
        return 0.9
    if is_clear and len(keywords) >= 2:
        return 0.85
    if is_clear and len(keywords) == 1:
        return 0.75
    return 0.55


def _normalize_claim(sentence: str, category: str) -> str:
    """Rewrite a policy sentence as a simple testable assertion.

    Pattern: "The application [does/does not] [action]"
    """
    lower = sentence.lower()

    is_negation = bool(
        re.search(
            r"\b(?:do\s+not|don't|never|will\s+not|won't|does\s+not|doesn't)\b",
            lower,
        )
    )

    # Extract action verb phrase: use the first privacy verb found
    verb_match = _PRIVACY_VERBS.search(sentence)
    if verb_match:
        # Take text from the verb onward (trimmed to ~60 chars)
        action = sentence[verb_match.start() :].strip()
        # Trim to first punctuation or 80 chars
        action = re.split(r"[.,;]", action)[0].strip()[:80]
        verb_form = "does not" if is_negation else "does"
        return f"The application {verb_form} {action.lower()}"

    # Fallback: use sentence directly, truncated
    return sentence[:120]


def _split_compound_claim(
    sentence: str,
    section_title: str,
    sentence_start: int,
    sentence_end: int,
) -> list[_RawMatch] | None:
    """Split a compound sentence joined by 'and' into two claims if applicable.

    Splitting only occurs when:
    - The sentence contains " and "
    - Both halves contain at least one privacy-relevant verb
    - Both halves classify into *different* categories

    Returns None if the sentence should not be split.
    """
    parts = re.split(r"\s+and\s+", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None

    left, right = parts[0].strip(), parts[1].strip()

    # Both halves must have privacy verbs
    if not _PRIVACY_VERBS.search(left) or not _PRIVACY_VERBS.search(right):
        return None

    left_cat, left_kw = _classify_category(left)
    right_cat, right_kw = _classify_category(right)

    # Only split when categories differ
    if left_cat == right_cat:
        return None

    mid = sentence_start + len(left)
    return [
        _RawMatch(
            sentence=left,
            hint="compound_split",
            start=sentence_start,
            end=mid,
            confidence=_compute_confidence(left, "compound_split", left_kw),
        ),
        _RawMatch(
            sentence=right,
            hint="compound_split",
            start=mid,
            end=sentence_end,
            confidence=_compute_confidence(right, "compound_split", right_kw),
        ),
    ]


def _find_matches_in_text(
    text: str,
    text_offset: int,
    section_title: str,
) -> list[_RawMatch]:
    """Find all privacy-relevant matches within a block of text."""
    matches: list[_RawMatch] = []
    seen_sentences: set[str] = set()

    sentences = _extract_sentences(text)

    for sentence, rel_start, rel_end in sentences:
        abs_start = text_offset + rel_start
        abs_end = text_offset + rel_end

        if not sentence or len(sentence) < 10:
            continue

        # Check if any claim pattern fires on this sentence
        matched_hint: str | None = None
        for pattern, hint in _CLAIM_PATTERNS:
            if pattern.search(sentence):
                matched_hint = hint
                break

        if matched_hint is None:
            continue

        # Attempt compound splitting first
        splits = _split_compound_claim(sentence, section_title, abs_start, abs_end)
        if splits:
            for split in splits:
                norm = split.sentence.strip().lower()
                if norm not in seen_sentences:
                    seen_sentences.add(norm)
                    matches.append(split)
            continue

        # Single sentence match
        norm = sentence.strip().lower()
        if norm not in seen_sentences:
            seen_sentences.add(norm)
            _, keywords = _classify_category(sentence)
            confidence = _compute_confidence(sentence, matched_hint, keywords)
            matches.append(
                _RawMatch(
                    sentence=sentence,
                    hint=matched_hint,
                    start=abs_start,
                    end=abs_end,
                    confidence=confidence,
                )
            )

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_claims_text(policy_text: str) -> list[PolicyClaim]:
    """Extract privacy claims using text analysis only — no LLM required.

    Splits the policy into sections, applies regex/keyword pattern matching
    to find privacy-relevant sentences, classifies each into a category,
    normalizes it as a testable assertion, and returns PolicyClaim objects.

    Compound sentences joined by "and" with differing privacy categories are
    split into two separate claims.

    Args:
        policy_text: Full text of the privacy policy document.

    Returns:
        List of PolicyClaim objects with stable IDs and extraction_confidence.
    """
    sections = _split_sections(policy_text)
    logger.debug("Text extractor found %d sections", len(sections))

    raw_matches: list[tuple[_RawMatch, str]] = []  # (match, section_title)
    offset = 0

    for section_title, section_body in sections:
        # Compute offset of this section body in the original text
        body_start = policy_text.find(section_body, offset)
        if body_start == -1:
            body_start = offset
        section_matches = _find_matches_in_text(section_body, body_start, section_title)
        for m in section_matches:
            raw_matches.append((m, section_title))
        offset = body_start + len(section_body)

    claims: list[PolicyClaim] = []
    seen_normalized: set[str] = set()

    for raw, section_title in raw_matches:
        category, keywords = _classify_category(raw.sentence)
        normalized = _normalize_claim(raw.sentence, category)

        # Dedup by normalized text
        if normalized.lower() in seen_normalized:
            continue
        seen_normalized.add(normalized.lower())

        claim_id = PolicyClaim.compute_id(category, normalized)

        try:
            claim = PolicyClaim(
                id=claim_id,
                category=category,  # type: ignore[arg-type]
                text=raw.sentence,
                normalized=normalized,
                keywords=keywords,
                policy_section=section_title,
                testable=True,
                source_offsets=(raw.start, raw.end),
            )
            claims.append(claim)
        except Exception as exc:
            logger.warning(
                "Skipping malformed text-extracted claim: %s — %s",
                raw.sentence[:60],
                exc,
            )

    logger.info("Text extractor produced %d claims", len(claims))
    return claims
