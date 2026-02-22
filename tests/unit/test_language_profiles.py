"""Unit tests for language profiles and multi-language PR review support."""

import pytest

from mcp_vector_search.analysis.review.language_profiles import (
    LANGUAGE_PROFILES,
    detect_languages,
    get_languages_in_pr,
    get_profile,
)
from mcp_vector_search.analysis.review.pr_models import PRFilePatch


class TestLanguageProfiles:
    """Test language profile definitions and detection."""

    def test_all_profiles_have_required_fields(self):
        """Verify all language profiles have required fields."""
        for lang_key, profile in LANGUAGE_PROFILES.items():
            assert profile.name, f"{lang_key} missing name"
            assert profile.extensions, f"{lang_key} missing extensions"
            assert isinstance(profile.extensions, list)
            assert isinstance(profile.idioms, list)
            assert isinstance(profile.anti_patterns, list)
            assert isinstance(profile.security_patterns, list)

    def test_python_profile_complete(self):
        """Verify Python profile has comprehensive standards."""
        python = LANGUAGE_PROFILES["python"]
        assert python.name == "Python"
        assert ".py" in python.extensions
        assert ".pyi" in python.extensions
        assert len(python.idioms) >= 5
        assert len(python.anti_patterns) >= 5
        assert len(python.security_patterns) >= 5

        # Check for specific Python idioms
        idiom_text = " ".join(python.idioms).lower()
        assert "type hint" in idiom_text or "pep 484" in idiom_text
        assert "pep 8" in idiom_text or "snake_case" in idiom_text

    def test_typescript_profile_complete(self):
        """Verify TypeScript profile has comprehensive standards."""
        ts = LANGUAGE_PROFILES["typescript"]
        assert ts.name == "TypeScript"
        assert ".ts" in ts.extensions
        assert ".tsx" in ts.extensions
        assert len(ts.idioms) >= 5

        # Check for specific TypeScript idioms
        idiom_text = " ".join(ts.idioms).lower()
        assert "strict" in idiom_text
        assert "any" in idiom_text  # Avoid any type

    def test_get_profile_by_extension(self):
        """Test getting profile by file extension."""
        # Python
        profile = get_profile("src/main.py")
        assert profile is not None
        assert profile.name == "Python"

        # TypeScript
        profile = get_profile("src/component.tsx")
        assert profile is not None
        assert profile.name == "TypeScript"

        # Unknown extension
        profile = get_profile("README.md")
        assert profile is None

    def test_detect_languages_single_language(self):
        """Test detecting languages from patches (single language)."""
        patches = [
            PRFilePatch(
                file_path="src/main.py",
                old_content="",
                new_content="def main(): pass",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=True,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
            PRFilePatch(
                file_path="src/utils.py",
                old_content="",
                new_content="def helper(): pass",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=True,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
        ]

        languages = detect_languages(patches)
        assert "python" in languages
        assert len(languages["python"]) == 2
        assert "src/main.py" in languages["python"]
        assert "src/utils.py" in languages["python"]

    def test_detect_languages_multi_language(self):
        """Test detecting languages from patches (multi-language PR)."""
        patches = [
            PRFilePatch(
                file_path="backend/api.py",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
            PRFilePatch(
                file_path="frontend/App.tsx",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
            PRFilePatch(
                file_path="services/worker.go",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
        ]

        languages = detect_languages(patches)
        assert len(languages) == 3
        assert "python" in languages
        assert "typescript" in languages
        assert "go" in languages

    def test_detect_languages_skips_deleted_files(self):
        """Test that deleted files are excluded from language detection."""
        patches = [
            PRFilePatch(
                file_path="src/main.py",
                old_content="def main(): pass",
                new_content=None,
                diff_text="",
                additions=0,
                deletions=1,
                is_new_file=False,
                is_deleted=True,
                is_renamed=False,
                old_path=None,
            ),
        ]

        languages = detect_languages(patches)
        assert len(languages) == 0

    def test_get_languages_in_pr(self):
        """Test getting unique language profiles from PR patches."""
        patches = [
            PRFilePatch(
                file_path="src/main.py",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
            PRFilePatch(
                file_path="src/utils.py",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
            PRFilePatch(
                file_path="frontend/App.tsx",
                old_content="",
                new_content="",
                diff_text="",
                additions=1,
                deletions=0,
                is_new_file=False,
                is_deleted=False,
                is_renamed=False,
                old_path=None,
            ),
        ]

        profiles = get_languages_in_pr(patches)
        assert len(profiles) == 2
        profile_names = [p.name for p in profiles]
        assert "Python" in profile_names
        assert "TypeScript" in profile_names

        # Verify profiles are sorted alphabetically
        assert profiles[0].name == "Python"
        assert profiles[1].name == "TypeScript"

    def test_get_languages_in_pr_empty(self):
        """Test getting languages from empty PR."""
        profiles = get_languages_in_pr([])
        assert len(profiles) == 0

    def test_all_supported_languages(self):
        """Test that all documented languages are present."""
        expected_languages = [
            "python",
            "typescript",
            "javascript",
            "java",
            "csharp",
            "ruby",
            "go",
            "rust",
            "php",
            "swift",
            "kotlin",
            "scala",
        ]

        for lang in expected_languages:
            assert lang in LANGUAGE_PROFILES, f"{lang} profile missing"

    def test_language_config_files_defined(self):
        """Test that config files are defined for language profiles."""
        # Python
        python = LANGUAGE_PROFILES["python"]
        assert "pyproject.toml" in python.config_files

        # TypeScript
        ts = LANGUAGE_PROFILES["typescript"]
        assert "tsconfig.json" in ts.config_files

        # Ruby
        ruby = LANGUAGE_PROFILES["ruby"]
        assert ".rubocop.yml" in ruby.config_files

        # Java
        java = LANGUAGE_PROFILES["java"]
        assert "checkstyle.xml" in java.config_files or "pom.xml" in java.config_files

    def test_security_patterns_cover_owasp(self):
        """Test that security patterns cover OWASP top vulnerabilities."""
        # Check Python for common vulnerabilities
        python = LANGUAGE_PROFILES["python"]
        security_text = " ".join(python.security_patterns).lower()
        assert "sql injection" in security_text
        assert "command injection" in security_text or "subprocess" in security_text
        assert "deserialization" in security_text or "pickle" in security_text

        # Check TypeScript for web vulnerabilities
        ts = LANGUAGE_PROFILES["typescript"]
        security_text = " ".join(ts.security_patterns).lower()
        assert "xss" in security_text
        assert "csrf" in security_text or "token" in security_text
