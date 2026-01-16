"""Unit tests for gitignore pattern matching logic.

Tests the GitignorePattern.matches() method to ensure proper behavior
for directory patterns with trailing slashes.
"""

from pathlib import Path

from mcp_vector_search.utils.gitignore import GitignoreParser, GitignorePattern


class TestGitignorePatternMatching:
    """Unit tests for gitignore pattern matching logic."""

    def test_directory_pattern_matches_files_inside(self):
        """Test that 'node_modules/' pattern matches files inside the directory.

        This is a regression test for the bug where files inside directories
        weren't being ignored when the pattern ended with '/'.

        Expected behavior:
        - node_modules/ → matches directory AND files inside
        - node_modules/package.json → should be matched
        - node_modules/foo/bar.js → should be matched
        """
        pattern = GitignorePattern("node_modules/", is_directory_only=True)

        # Directory itself should match
        assert pattern.matches("node_modules", is_directory=True)

        # Files inside should match (this was the bug!)
        assert pattern.matches("node_modules/package.json", is_directory=False)
        assert pattern.matches("node_modules/foo/bar.js", is_directory=False)
        assert pattern.matches("node_modules/deep/nested/file.txt", is_directory=False)

        # Subdirectories inside should also match
        assert pattern.matches("node_modules/foo", is_directory=True)
        assert pattern.matches("node_modules/deep/nested", is_directory=True)

    def test_directory_pattern_does_not_match_unrelated_paths(self):
        """Test that 'node_modules/' doesn't match unrelated paths."""
        pattern = GitignorePattern("node_modules/", is_directory_only=True)

        # Should NOT match files/dirs that don't start with node_modules
        assert not pattern.matches("src/node_modules.txt", is_directory=False)
        assert not pattern.matches("node_modules.backup", is_directory=False)
        assert not pattern.matches("my_node_modules/file.js", is_directory=False)

    def test_build_directory_pattern(self):
        """Test that 'build/' pattern correctly matches build directory and contents."""
        pattern = GitignorePattern("build/", is_directory_only=True)

        # Directory matches
        assert pattern.matches("build", is_directory=True)

        # Files inside match
        assert pattern.matches("build/index.html", is_directory=False)
        assert pattern.matches("build/assets/app.js", is_directory=False)

        # Unrelated paths don't match
        assert not pattern.matches("src/build.py", is_directory=False)

    def test_pycache_directory_pattern(self):
        """Test that '__pycache__/' pattern matches Python cache directories.

        Note: In gitignore, patterns without leading slash match at any level.
        So '__pycache__/' matches both 'pkg/__pycache__' directory AND files
        inside it, but the pattern matching is done relative to each level.
        """
        pattern = GitignorePattern("__pycache__/", is_directory_only=True)

        # Matches __pycache__ directories at root level
        assert pattern.matches("__pycache__", is_directory=True)

        # Matches files inside root-level __pycache__
        assert pattern.matches("__pycache__/module.pyc", is_directory=False)
        assert pattern.matches("__pycache__/deep/nested.pyc", is_directory=False)

        # For nested __pycache__, gitignore parsers typically check each component
        # The pattern '__pycache__/' will match via the parent-matching logic
        # when the path includes __pycache__ as a parent directory

    def test_gitignore_parser_integration(self, tmp_path: Path):
        """Integration test: GitignoreParser correctly ignores files in directories."""
        # Create .gitignore with directory pattern
        gitignore_file = tmp_path / ".gitignore"
        gitignore_file.write_text("node_modules/\n")

        # Create test structure
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text("{}")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.js").write_text("console.log('hello')")

        # Parse gitignore
        parser = GitignoreParser(tmp_path)

        # node_modules directory should be ignored
        assert parser.is_ignored(node_modules, is_directory=True)

        # Files inside node_modules should be ignored (this was the bug!)
        assert parser.is_ignored(node_modules / "package.json", is_directory=False)

        # src/app.js should NOT be ignored
        assert not parser.is_ignored(src_dir / "app.js", is_directory=False)

    def test_multiple_directory_patterns(self, tmp_path: Path):
        """Test multiple directory patterns work correctly."""
        gitignore_file = tmp_path / ".gitignore"
        gitignore_file.write_text("build/\ndist/\n*.pyc\n")

        # Create test structure
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "index.html").write_text("<html></html>")
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "bundle.js").write_text("// code")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')")
        (tmp_path / "src" / "cache.pyc").write_text("bytecode")

        parser = GitignoreParser(tmp_path)

        # build/ files should be ignored
        assert parser.is_ignored(tmp_path / "build" / "index.html", is_directory=False)

        # dist/ files should be ignored
        assert parser.is_ignored(tmp_path / "dist" / "bundle.js", is_directory=False)

        # .pyc files should be ignored
        assert parser.is_ignored(tmp_path / "src" / "cache.pyc", is_directory=False)

        # Regular files should NOT be ignored
        assert not parser.is_ignored(tmp_path / "src" / "app.py", is_directory=False)

    def test_pattern_without_trailing_slash(self):
        """Test that pattern without trailing slash still works (matches files or dirs)."""
        pattern = GitignorePattern("*.pyc", is_directory_only=False)

        # Should match .pyc files
        assert pattern.matches("app.pyc", is_directory=False)
        assert pattern.matches("src/module.pyc", is_directory=False)

        # Should NOT match other files
        assert not pattern.matches("app.py", is_directory=False)

    def test_nested_directory_pattern(self):
        """Test pattern like 'src/build/' matches nested directory."""
        pattern = GitignorePattern("src/build/", is_directory_only=True)

        # Should match src/build directory
        assert pattern.matches("src/build", is_directory=True)

        # Should match files inside src/build
        assert pattern.matches("src/build/index.html", is_directory=False)
        assert pattern.matches("src/build/assets/app.js", is_directory=False)

        # Should NOT match build in other locations
        assert not pattern.matches("build", is_directory=True)
        assert not pattern.matches("dist/build", is_directory=True)
