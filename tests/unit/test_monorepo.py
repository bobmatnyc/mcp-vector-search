"""Tests for monorepo detection."""

from pathlib import Path

from src.mcp_vector_search.utils.monorepo import (
    EXCLUDED_SUBPROJECT_DIRS,
    MonorepoDetector,
)


class TestMonorepoDetector:
    """Test MonorepoDetector functionality."""

    def test_excluded_directories_defined(self):
        """Verify that excluded directories are defined."""
        assert len(EXCLUDED_SUBPROJECT_DIRS) > 0
        assert "tests" in EXCLUDED_SUBPROJECT_DIRS
        assert "examples" in EXCLUDED_SUBPROJECT_DIRS
        assert "docs" in EXCLUDED_SUBPROJECT_DIRS

    def test_is_excluded_path_tests_directory(self, tmp_path: Path):
        """Test that tests directory is excluded."""
        detector = MonorepoDetector(tmp_path)

        test_path = tmp_path / "tests" / "manual"
        assert detector._is_excluded_path(test_path) is True

    def test_is_excluded_path_examples_directory(self, tmp_path: Path):
        """Test that examples directory is excluded."""
        detector = MonorepoDetector(tmp_path)

        example_path = tmp_path / "examples" / "demo"
        assert detector._is_excluded_path(example_path) is True

    def test_is_excluded_path_docs_directory(self, tmp_path: Path):
        """Test that docs directory is excluded."""
        detector = MonorepoDetector(tmp_path)

        docs_path = tmp_path / "docs" / "guide"
        assert detector._is_excluded_path(docs_path) is True

    def test_is_excluded_path_src_directory(self, tmp_path: Path):
        """Test that src directory is NOT excluded."""
        detector = MonorepoDetector(tmp_path)

        src_path = tmp_path / "src" / "package"
        assert detector._is_excluded_path(src_path) is False

    def test_is_excluded_path_apps_directory(self, tmp_path: Path):
        """Test that apps directory is NOT excluded."""
        detector = MonorepoDetector(tmp_path)

        apps_path = tmp_path / "apps" / "frontend"
        assert detector._is_excluded_path(apps_path) is False

    def test_detect_by_package_json_excludes_tests(self, tmp_path: Path):
        """Test that package.json in tests directory is not detected as subproject."""
        # Create tests/manual/package.json
        tests_dir = tmp_path / "tests" / "manual"
        tests_dir.mkdir(parents=True)
        package_json = tests_dir / "package.json"
        package_json.write_text('{"name": "test-project"}')

        # Detect subprojects
        detector = MonorepoDetector(tmp_path)
        subprojects = detector.detect_subprojects()

        # Should not detect tests/manual as a subproject
        assert len(subprojects) == 0

    def test_detect_by_package_json_includes_apps(self, tmp_path: Path):
        """Test that package.json in apps directory IS detected as subproject."""
        # Create apps/frontend/package.json
        apps_dir = tmp_path / "apps" / "frontend"
        apps_dir.mkdir(parents=True)
        package_json = apps_dir / "package.json"
        package_json.write_text('{"name": "frontend-app"}')

        # Detect subprojects
        detector = MonorepoDetector(tmp_path)
        subprojects = detector.detect_subprojects()

        # Should detect apps/frontend as a subproject
        assert len(subprojects) == 1
        assert subprojects[0].name == "frontend-app"
        assert subprojects[0].relative_path == "apps/frontend"

    def test_detect_by_package_json_excludes_node_modules(self, tmp_path: Path):
        """Test that package.json in node_modules is not detected."""
        # Create node_modules/some-package/package.json
        node_modules = tmp_path / "node_modules" / "some-package"
        node_modules.mkdir(parents=True)
        package_json = node_modules / "package.json"
        package_json.write_text('{"name": "some-package"}')

        # Detect subprojects
        detector = MonorepoDetector(tmp_path)
        subprojects = detector.detect_subprojects()

        # Should not detect node_modules as subproject
        assert len(subprojects) == 0

    def test_detect_by_package_json_multiple_directories(self, tmp_path: Path):
        """Test detection with mix of excluded and included directories."""
        # Create various package.json files
        directories = [
            ("tests/manual", "test-manual", False),  # Should be excluded
            ("examples/demo", "demo-example", False),  # Should be excluded
            ("apps/frontend", "frontend-app", True),  # Should be included
            ("apps/backend", "backend-app", True),  # Should be included
            ("packages/shared", "shared-lib", True),  # Should be included
        ]

        for dir_path, name, _should_include in directories:
            dir_full = tmp_path / dir_path
            dir_full.mkdir(parents=True)
            package_json = dir_full / "package.json"
            package_json.write_text(f'{{"name": "{name}"}}')

        # Detect subprojects
        detector = MonorepoDetector(tmp_path)
        subprojects = detector.detect_subprojects()

        # Should only detect apps and packages, not tests or examples
        assert len(subprojects) == 3
        subproject_names = {sp.name for sp in subprojects}
        assert subproject_names == {"frontend-app", "backend-app", "shared-lib"}

        # Verify excluded directories are not present
        for sp in subprojects:
            assert not sp.relative_path.startswith("tests/")
            assert not sp.relative_path.startswith("examples/")
