"""Parser registry for MCP Vector Search."""

from pathlib import Path

from loguru import logger

from .base import BaseParser, FallbackParser
from .csharp import CSharpParser
from .dart import DartParser
from .go import GoParser
from .html import HTMLParser
from .java import JavaParser
from .javascript import JavaScriptParser, TypeScriptParser
from .php import PHPParser
from .python import PythonParser
from .ruby import RubyParser
from .rust import RustParser
from .text import TextParser


class ParserRegistry:
    """Registry for managing language parsers."""

    def __init__(self) -> None:
        """Initialize parser registry with lazy loading."""
        self._parsers: dict[
            str, BaseParser
        ] = {}  # Actual parser instances (created on-demand)
        self._parser_classes: dict[
            str, type[BaseParser]
        ] = {}  # Parser classes for lazy instantiation
        self._extension_map: dict[str, str] = {}  # Extension to language mapping
        self._fallback_parser = FallbackParser()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure parsers are initialized (lazy initialization)."""
        if not self._initialized:
            self._register_default_parsers()
            self._initialized = True

    def _register_default_parsers(self) -> None:
        """Register default parsers for supported languages (lazy creation).

        This method builds the extension-to-language mapping WITHOUT
        instantiating parsers. Parsers are created on-demand in get_parser().
        Error isolation happens at instantiation time.
        """
        # Map extensions to parser classes (not instances)
        parser_map = {
            ".py": ("python", PythonParser),
            ".pyw": ("python", PythonParser),
            ".js": ("javascript", JavaScriptParser),
            ".jsx": ("javascript", JavaScriptParser),
            ".mjs": ("javascript", JavaScriptParser),
            ".ts": ("typescript", TypeScriptParser),
            ".tsx": ("typescript", TypeScriptParser),
            ".java": ("java", JavaParser),
            ".cs": ("c_sharp", CSharpParser),
            ".go": ("go", GoParser),
            ".rs": ("rust", RustParser),
            ".dart": ("dart", DartParser),
            ".php": ("php", PHPParser),
            ".rb": ("ruby", RubyParser),
            ".txt": ("text", TextParser),
            ".md": ("text", TextParser),
            ".markdown": ("text", TextParser),
            ".html": ("html", HTMLParser),
            ".htm": ("html", HTMLParser),
        }

        # Build extension map without creating parsers
        for ext, (lang, parser_class) in parser_map.items():
            self._extension_map[ext.lower()] = lang
            # Store parser class for lazy instantiation
            if lang not in self._parser_classes:
                self._parser_classes[lang] = parser_class

        logger.debug(
            f"Registered {len(self._parser_classes)} parser classes (lazy loading enabled)"
        )

    def register_parser(self, language: str, parser: BaseParser) -> None:
        """Register a parser for a specific language.

        Args:
            language: Language name
            parser: Parser instance
        """
        self._parsers[language] = parser

        # Map file extensions to language
        for ext in parser.get_supported_extensions():
            if ext != "*":  # Skip fallback marker
                self._extension_map[ext.lower()] = language

        logger.debug(f"Registered parser for {language}: {parser.__class__.__name__}")

    def get_parser(self, file_extension: str) -> BaseParser:
        """Get parser for a file extension (lazy instantiation).

        Args:
            file_extension: File extension (including dot)

        Returns:
            Parser instance (fallback parser if no specific parser found)
        """
        self._ensure_initialized()
        language = self._extension_map.get(file_extension.lower())

        # Lazy instantiation: create parser only if not already created
        if language:
            if language not in self._parsers:
                # Create parser instance on first use
                parser_class = self._parser_classes.get(language)
                if parser_class:
                    self._parsers[language] = parser_class()
                    logger.debug(f"Lazily instantiated parser for {language}")

            if language in self._parsers:
                return self._parsers[language]

        # Return fallback parser for unsupported extensions
        return self._fallback_parser

    def get_parser_for_file(self, file_path: Path) -> BaseParser:
        """Get parser for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Parser instance
        """
        return self.get_parser(file_path.suffix)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            List of language names
        """
        self._ensure_initialized()
        return list(self._parsers.keys())

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of file extensions
        """
        self._ensure_initialized()
        return list(self._extension_map.keys())

    def is_supported(self, file_extension: str) -> bool:
        """Check if a file extension is supported.

        Args:
            file_extension: File extension to check

        Returns:
            True if supported (always True due to fallback parser)
        """
        return True  # Always supported due to fallback parser

    def get_language_for_extension(self, file_extension: str) -> str:
        """Get language name for a file extension.

        Args:
            file_extension: File extension

        Returns:
            Language name (or "text" for unsupported extensions)
        """
        self._ensure_initialized()
        return self._extension_map.get(file_extension.lower(), "text")

    def get_parser_info(self) -> dict[str, dict[str, any]]:
        """Get information about registered parsers.

        Returns:
            Dictionary with parser information
        """
        self._ensure_initialized()
        info = {}

        for language, parser in self._parsers.items():
            info[language] = {
                "class": parser.__class__.__name__,
                "extensions": parser.get_supported_extensions(),
                "language": getattr(parser, "language", None) or language,
            }

        # Add fallback parser info
        fallback_lang = getattr(self._fallback_parser, "language", None) or "unknown"
        info["fallback"] = {
            "class": self._fallback_parser.__class__.__name__,
            "extensions": ["*"],
            "language": fallback_lang,
        }

        return info


# Global parser registry instance
_registry = ParserRegistry()


def get_parser_registry() -> ParserRegistry:
    """Get the global parser registry instance.

    Returns:
        Parser registry instance
    """
    return _registry


def register_parser(language: str, parser: BaseParser) -> None:
    """Register a parser in the global registry.

    Args:
        language: Language name
        parser: Parser instance
    """
    _registry.register_parser(language, parser)


def get_parser(file_extension: str) -> BaseParser:
    """Get parser for a file extension from the global registry.

    Args:
        file_extension: File extension

    Returns:
        Parser instance
    """
    return _registry.get_parser(file_extension)


def get_parser_for_file(file_path: Path) -> BaseParser:
    """Get parser for a file from the global registry.

    Args:
        file_path: File path

    Returns:
        Parser instance
    """
    return _registry.get_parser_for_file(file_path)
