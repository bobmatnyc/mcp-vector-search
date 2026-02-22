"""Query expansion with code-specific synonym dictionary."""

import re
from pathlib import Path
from typing import Any

import orjson
from loguru import logger

# Built-in code synonym dictionary
CODE_SYNONYMS: dict[str, list[str]] = {
    "auth": ["authentication", "authorize", "login", "session", "credential", "oauth"],
    "delete": ["remove", "destroy", "drop", "purge", "clean", "erase"],
    "create": ["new", "init", "initialize", "build", "make", "generate", "construct"],
    "error": ["exception", "fault", "failure", "crash", "bug", "issue"],
    "config": ["configuration", "settings", "options", "preferences", "env"],
    "db": ["database", "storage", "persistence", "datastore", "repo", "repository"],
    "api": ["endpoint", "route", "handler", "controller", "resource"],
    "test": ["spec", "assertion", "verify", "validate", "check", "expect"],
    "async": ["concurrent", "parallel", "await", "coroutine", "future", "promise"],
    "cache": ["memoize", "store", "buffer", "preload"],
    "parse": ["extract", "tokenize", "analyze", "process", "decode"],
    "render": ["display", "draw", "paint", "show", "present", "view"],
    "fetch": ["get", "retrieve", "load", "download", "pull", "request"],
    "send": ["post", "push", "emit", "dispatch", "publish", "transmit"],
    "log": ["trace", "debug", "print", "output", "record", "audit"],
    "user": ["account", "profile", "member", "identity", "principal"],
    "file": ["document", "path", "stream", "blob", "resource"],
    "search": ["find", "query", "lookup", "filter", "match", "grep"],
    "update": ["modify", "patch", "change", "edit", "mutate", "alter"],
    "serialize": ["encode", "marshal", "dump", "stringify", "format"],
    "deserialize": ["decode", "unmarshal", "load", "parse"],
    "validate": ["check", "verify", "sanitize", "assert", "ensure"],
    "transform": ["convert", "map", "translate", "adapt", "morph"],
    "middleware": ["interceptor", "filter", "hook", "plugin", "handler"],
    "deploy": ["release", "publish", "ship", "rollout", "launch"],
}


class QueryExpander:
    """Query expander with code-specific synonym dictionary.

    Expands search queries by substituting code-specific synonyms to improve
    semantic search recall. Supports both built-in synonyms and custom user-defined
    synonyms loaded from .mcp-vector-search/synonyms.json.

    Features:
    - One substitution per variant (generates multiple query variants)
    - Bidirectional mapping (query contains synonym → expand to key)
    - Custom synonyms merged with built-in dictionary
    - Token-based expansion (respects word boundaries)
    """

    def __init__(self, custom_synonyms_path: Path | None = None) -> None:
        """Initialize query expander with optional custom synonyms.

        Args:
            custom_synonyms_path: Optional path to custom synonyms JSON file
                                 (default: .mcp-vector-search/synonyms.json)
        """
        # Start with built-in synonyms
        self.synonyms = dict(CODE_SYNONYMS)

        # Build reverse mapping (synonym → key)
        # This enables bidirectional expansion: "authentication" → "auth"
        self.reverse_synonyms: dict[str, str] = {}
        for key, synonyms in CODE_SYNONYMS.items():
            for synonym in synonyms:
                # Store the key as the target for expansion
                # If synonym appears in query, expand to the key
                self.reverse_synonyms[synonym] = key

        # Load custom synonyms if provided
        if custom_synonyms_path and custom_synonyms_path.exists():
            self._load_custom_synonyms(custom_synonyms_path)
            logger.debug(
                f"Loaded custom synonyms from {custom_synonyms_path} "
                f"({len(self.synonyms)} total synonym groups)"
            )
        else:
            logger.trace(f"Using built-in synonyms only ({len(self.synonyms)} groups)")

    def _load_custom_synonyms(self, path: Path) -> None:
        """Load custom synonyms from JSON file and merge with built-in.

        Args:
            path: Path to custom synonyms JSON file

        Format:
            {
                "synonym_key": ["synonym1", "synonym2", ...],
                ...
            }
        """
        try:
            with path.open("rb") as f:
                custom: dict[str, list[str]] = orjson.loads(f.read())

            # Merge custom synonyms with built-in
            for key, synonyms in custom.items():
                if key in self.synonyms:
                    # Extend existing synonym group
                    self.synonyms[key].extend(synonyms)
                else:
                    # Add new synonym group
                    self.synonyms[key] = synonyms

                # Update reverse mapping
                for synonym in synonyms:
                    self.reverse_synonyms[synonym] = key

            logger.debug(f"Loaded {len(custom)} custom synonym groups from {path}")

        except Exception as e:
            logger.warning(f"Failed to load custom synonyms from {path}: {e}")

    def expand(self, query: str) -> list[str]:
        """Expand query into multiple variants using synonym substitution.

        Strategy:
        1. Tokenize query (split on whitespace, preserve punctuation)
        2. Find tokens that match synonym keys or reverse synonyms
        3. Generate variants by substituting ONE synonym per variant
        4. Return original query + all variants (deduplicated)

        Args:
            query: Original search query

        Returns:
            List of query variants including original query

        Example:
            >>> expander.expand("auth middleware")
            [
                "auth middleware",
                "authentication middleware",
                "authorize middleware",
                "login middleware",
                ...
                "auth interceptor",
                "auth filter",
                "auth hook",
                ...
            ]
        """
        if not query.strip():
            return [query]

        # Start with original query
        variants = [query]

        # Tokenize query (split on word boundaries, lowercase for matching)
        tokens = self._tokenize(query)

        # Generate variants by substituting one synonym at a time
        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Check if token matches a synonym key (e.g., "auth")
            if token_lower in self.synonyms:
                # Expand to each synonym in the group
                for synonym in self.synonyms[token_lower]:
                    # Create variant by replacing token with synonym
                    variant_tokens = tokens.copy()
                    variant_tokens[i] = synonym
                    variant = " ".join(variant_tokens)
                    if variant not in variants:
                        variants.append(variant)

            # Check if token matches a synonym value (reverse lookup)
            # e.g., "authentication" → "auth"
            elif token_lower in self.reverse_synonyms:
                # Expand to the key
                key = self.reverse_synonyms[token_lower]
                variant_tokens = tokens.copy()
                variant_tokens[i] = key
                variant = " ".join(variant_tokens)
                if variant not in variants:
                    variants.append(variant)

        logger.debug(f"Query expansion: '{query}' → {len(variants)} variants")
        if len(variants) > 1:
            logger.trace(f"Expanded variants: {variants}")

        return variants

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Tokenize query into words (split on whitespace, strip punctuation).

        Args:
            query: Search query

        Returns:
            List of tokens

        Example:
            >>> _tokenize("auth middleware, test")
            ["auth", "middleware", "test"]
        """
        # Split on whitespace
        tokens = query.split()

        # Strip punctuation from each token (but preserve hyphenated words)
        cleaned_tokens = []
        for token in tokens:
            # Remove leading/trailing punctuation, preserve internal hyphens
            cleaned = re.sub(r"^[^\w-]+|[^\w-]+$", "", token, flags=re.UNICODE)
            if cleaned:
                cleaned_tokens.append(cleaned)

        return cleaned_tokens

    def get_synonyms(self, key: str) -> list[str]:
        """Get synonyms for a given key.

        Args:
            key: Synonym key

        Returns:
            List of synonyms for the key (empty if key not found)
        """
        return self.synonyms.get(key, [])

    def get_all_keys(self) -> list[str]:
        """Get all synonym keys.

        Returns:
            List of all synonym keys (sorted)
        """
        return sorted(self.synonyms.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get expander statistics.

        Returns:
            Dictionary with stats: total keys, total synonyms, avg synonyms per key
        """
        total_keys = len(self.synonyms)
        total_synonyms = sum(len(syns) for syns in self.synonyms.values())
        avg_synonyms = total_synonyms / total_keys if total_keys > 0 else 0

        return {
            "total_synonym_groups": total_keys,
            "total_synonyms": total_synonyms,
            "avg_synonyms_per_group": round(avg_synonyms, 2),
            "built_in_groups": len(CODE_SYNONYMS),
            "custom_groups": total_keys - len(CODE_SYNONYMS),
        }
