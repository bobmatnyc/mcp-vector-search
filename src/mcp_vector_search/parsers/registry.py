"""Parser registry for MCP Vector Search."""

from pathlib import Path

from loguru import logger

from .base import BaseParser, FallbackParser
from .dart import DartParser
from .html import HTMLParser
from .javascript import JavaScriptParser, TypeScriptParser
from .php import PHPParser
from .python import PythonParser
from .ruby import RubyParser
from .text import TextParser


class ParserRegistry:
    """Registry for managing language parsers."""

    def __init__(self) -> None:
        """Initialize parser registry."""
        self._parsers: dict[str, BaseParser] = {}
        self._extension_map: dict[str, str] = {}
        self._fallback_parser = FallbackParser()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure parsers are initialized (lazy initialization)."""
        if not self._initialized:
            self._register_default_parsers()
            self._initialized = True

    def _register_default_parsers(self) -> None:
        """Register default parsers for supported languages."""
        # Register Python parser
        python_parser = PythonParser()
        self.register_parser("python", python_parser)

        # Register JavaScript parser
        javascript_parser = JavaScriptParser()
        self.register_parser("javascript", javascript_parser)

        # Register TypeScript parser
        typescript_parser = TypeScriptParser()
        self.register_parser("typescript", typescript_parser)

        # Register Dart parser
        dart_parser = DartParser()
        self.register_parser("dart", dart_parser)

        # Register PHP parser
        php_parser = PHPParser()
        self.register_parser("php", php_parser)

        # Register Ruby parser
        ruby_parser = RubyParser()
        self.register_parser("ruby", ruby_parser)

        # Register Text parser for .txt files
        text_parser = TextParser()
        self.register_parser("text", text_parser)

        # Register HTML parser for .html files
        html_parser = HTMLParser()
        self.register_parser("html", html_parser)

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
        """Get parser for a file extension.

        Args:
            file_extension: File extension (including dot)

        Returns:
            Parser instance (fallback parser if no specific parser found)
        """
        self._ensure_initialized()
        language = self._extension_map.get(file_extension.lower())
        if language and language in self._parsers:
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
