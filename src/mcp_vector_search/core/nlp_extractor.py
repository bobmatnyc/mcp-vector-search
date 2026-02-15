"""NLP-based entity extraction from docstrings and comments.

Uses lightweight NLP techniques (no LLM required):
- YAKE for keyword extraction
- Regex patterns for code references
- Pattern matching for technical terms
"""

import re
from dataclasses import dataclass, field

# Try YAKE first (lightweight), fallback to simple extraction
try:
    import yake

    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False


@dataclass
class ExtractedEntities:
    """Entities extracted from text."""

    keywords: list[str] = field(default_factory=list)  # Top keywords from YAKE
    code_references: list[str] = field(default_factory=list)  # `backtick` references
    technical_terms: list[str] = field(
        default_factory=list
    )  # Capitalized terms, acronyms
    action_verbs: list[str] = field(
        default_factory=list
    )  # Returns, raises, creates, etc.


class NLPExtractor:
    """Extract semantic entities from docstrings/comments."""

    def __init__(self, max_keywords: int = 10):
        """Initialize NLP extractor.

        Args:
            max_keywords: Maximum number of keywords to extract
        """
        self.max_keywords = max_keywords
        if HAS_YAKE:
            self.kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=2,  # 1-2 word phrases
                dedupLim=0.7,
                top=max_keywords,
                features=None,
            )
        else:
            self.kw_extractor = None

    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entity types from text.

        Args:
            text: Text to extract entities from (docstring, comment, etc.)

        Returns:
            ExtractedEntities with all extracted entity types
        """
        if not text or len(text.strip()) < 10:
            return ExtractedEntities()

        return ExtractedEntities(
            keywords=self._extract_keywords(text),
            code_references=self._extract_code_refs(text),
            technical_terms=self._extract_technical_terms(text),
            action_verbs=self._extract_action_verbs(text),
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Use YAKE or fallback to simple extraction.

        Args:
            text: Text to extract keywords from

        Returns:
            List of extracted keywords
        """
        if self.kw_extractor:
            try:
                keywords = self.kw_extractor.extract_keywords(text)
                return [kw for kw, _score in keywords]
            except Exception:
                # Fallback if YAKE fails
                pass

        # Simple fallback: extract significant words
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        # Filter common words
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "will",
            "would",
            "could",
            "should",
            "their",
            "there",
            "they",
            "them",
            "these",
            "those",
            "when",
            "where",
            "which",
            "while",
        }
        return list({w for w in words if w not in stopwords})[: self.max_keywords]

    def _extract_code_refs(self, text: str) -> list[str]:
        """Extract backtick code references.

        Args:
            text: Text to extract from

        Returns:
            List of code references
        """
        return re.findall(r"`([^`]+)`", text)

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract CamelCase, ACRONYMS, and technical patterns.

        Args:
            text: Text to extract from

        Returns:
            List of technical terms
        """
        terms = []
        # CamelCase
        terms.extend(re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", text))
        # ACRONYMS (2+ uppercase letters)
        terms.extend(re.findall(r"\b[A-Z]{2,}\b", text))
        # Technical patterns: word_word
        terms.extend(re.findall(r"\b[a-z]+_[a-z_]+\b", text))
        return list(set(terms))

    def _extract_action_verbs(self, text: str) -> list[str]:
        """Extract docstring action verbs (Returns, Raises, etc.).

        Args:
            text: Text to extract from

        Returns:
            List of action verbs
        """
        pattern = r"\b(returns?|raises?|creates?|generates?|initializes?|loads?|saves?|parses?|validates?|processes?|handles?|builds?|performs?|executes?|extracts?|computes?|calculates?)\b"
        verbs = re.findall(pattern, text.lower())
        return list(set(verbs))
