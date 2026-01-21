"""Unit tests for entry point detection."""

import tempfile
from pathlib import Path

from src.mcp_vector_search.analysis.entry_points import (
    EntryPoint,
    EntryPointDetector,
    EntryPointType,
)


class TestEntryPointType:
    """Test EntryPointType enum."""

    def test_entry_point_types(self):
        """Test that all expected entry point types exist."""
        assert EntryPointType.MAIN.value == "MAIN"
        assert EntryPointType.CLI.value == "CLI"
        assert EntryPointType.ROUTE.value == "ROUTE"
        assert EntryPointType.TEST.value == "TEST"
        assert EntryPointType.EXPORT.value == "EXPORT"
        assert EntryPointType.PUBLIC.value == "PUBLIC"
        assert EntryPointType.CUSTOM.value == "CUSTOM"

    def test_string_conversion(self):
        """Test string conversion of entry point types."""
        assert str(EntryPointType.MAIN) == "MAIN"
        assert str(EntryPointType.CLI) == "CLI"


class TestEntryPoint:
    """Test EntryPoint dataclass."""

    def test_creation(self):
        """Test creating an EntryPoint instance."""
        ep = EntryPoint(
            name="main",
            file_path="main.py",
            line_number=10,
            type=EntryPointType.MAIN,
            confidence=1.0,
        )

        assert ep.name == "main"
        assert ep.file_path == "main.py"
        assert ep.line_number == 10
        assert ep.type == EntryPointType.MAIN
        assert ep.confidence == 1.0

    def test_string_representation(self):
        """Test string representation."""
        ep = EntryPoint(
            name="test_function",
            file_path="test.py",
            line_number=5,
            type=EntryPointType.TEST,
            confidence=1.0,
        )

        ep_str = str(ep)
        assert "TEST" in ep_str
        assert "test_function" in ep_str
        assert "test.py:5" in ep_str


class TestEntryPointDetector:
    """Test EntryPointDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = EntryPointDetector(include_public=False)
        assert detector.include_public is False

        detector = EntryPointDetector(include_public=True)
        assert detector.include_public is True

    def test_detect_main_block(self):
        """Test detection of if __name__ == '__main__' blocks."""
        code = """
def main():
    print("Hello")

if __name__ == "__main__":
    main()
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("main.py"), code)

        assert len(entry_points) == 1
        assert entry_points[0].name == "main"
        assert entry_points[0].type == EntryPointType.MAIN
        assert entry_points[0].confidence == 1.0

    def test_detect_main_block_multiple_calls(self):
        """Test detection of main block with multiple function calls."""
        code = """
def setup():
    pass

def main():
    pass

if __name__ == "__main__":
    setup()
    main()
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("main.py"), code)

        # Should detect both function calls in main block
        assert len(entry_points) == 2
        names = {ep.name for ep in entry_points}
        assert "setup" in names
        assert "main" in names

    def test_detect_cli_command_click(self):
        """Test detection of click CLI commands."""
        code = """
import click

@click.command()
def hello():
    print("Hello")
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("cli.py"), code)

        assert len(entry_points) == 1
        assert entry_points[0].name == "hello"
        assert entry_points[0].type == EntryPointType.CLI

    def test_detect_cli_command_typer(self):
        """Test detection of typer CLI commands."""
        code = """
import typer

app = typer.Typer()

@app.command()
def hello():
    print("Hello")
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("cli.py"), code)

        assert len(entry_points) == 1
        assert entry_points[0].name == "hello"
        assert entry_points[0].type == EntryPointType.CLI

    def test_detect_route_fastapi(self):
        """Test detection of FastAPI routes."""
        code = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
def get_users():
    return []

@app.post("/users")
def create_user():
    return {}
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("api.py"), code)

        assert len(entry_points) == 2
        names = {ep.name for ep in entry_points}
        assert "get_users" in names
        assert "create_user" in names
        assert all(ep.type == EntryPointType.ROUTE for ep in entry_points)

    def test_detect_route_flask(self):
        """Test detection of Flask routes."""
        code = """
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("app.py"), code)

        assert len(entry_points) == 1
        assert entry_points[0].name == "index"
        assert entry_points[0].type == EntryPointType.ROUTE

    def test_detect_test_functions(self):
        """Test detection of pytest test functions."""
        code = """
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 2 - 1 == 1

def helper_function():
    pass
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("test_math.py"), code)

        # Should only detect test_ functions, not helper
        assert len(entry_points) == 2
        names = {ep.name for ep in entry_points}
        assert "test_addition" in names
        assert "test_subtraction" in names
        assert all(ep.type == EntryPointType.TEST for ep in entry_points)

    def test_detect_pytest_fixtures(self):
        """Test detection of pytest fixtures."""
        code = """
import pytest

@pytest.fixture
def database():
    return connect_db()

@pytest.fixture
def client():
    return create_client()
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("conftest.py"), code)

        assert len(entry_points) == 2
        names = {ep.name for ep in entry_points}
        assert "database" in names
        assert "client" in names
        assert all(ep.type == EntryPointType.TEST for ep in entry_points)

    def test_detect_exports_in_init(self):
        """Test detection of __all__ exports in __init__.py."""
        code = """
from .models import User, Post
from .utils import validate

__all__ = ["User", "Post", "validate"]
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("package/__init__.py"), code)

        assert len(entry_points) == 3
        names = {ep.name for ep in entry_points}
        assert "User" in names
        assert "Post" in names
        assert "validate" in names
        assert all(ep.type == EntryPointType.EXPORT for ep in entry_points)

    def test_no_exports_in_regular_file(self):
        """Test that __all__ in non-__init__.py files is ignored."""
        code = """
__all__ = ["function"]

def function():
    pass
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("module.py"), code)

        # Should not detect exports in non-__init__.py
        assert len(entry_points) == 0

    def test_detect_public_functions(self):
        """Test detection of public module-level functions."""
        code = """
def public_function():
    pass

def _private_function():
    pass

def another_public():
    pass
"""
        detector = EntryPointDetector(include_public=True)
        entry_points = detector.detect_from_file(Path("module.py"), code)

        # Should only detect public functions (not starting with _)
        assert len(entry_points) == 2
        names = {ep.name for ep in entry_points}
        assert "public_function" in names
        assert "another_public" in names
        assert all(ep.type == EntryPointType.PUBLIC for ep in entry_points)

    def test_no_public_functions_when_disabled(self):
        """Test that public functions are not detected when disabled."""
        code = """
def public_function():
    pass
"""
        detector = EntryPointDetector(include_public=False)
        entry_points = detector.detect_from_file(Path("module.py"), code)

        # Should not detect public functions when disabled
        assert len(entry_points) == 0

    def test_public_function_confidence(self):
        """Test that public functions have lower confidence."""
        code = """
def public_function():
    pass
"""
        detector = EntryPointDetector(include_public=True)
        entry_points = detector.detect_from_file(Path("module.py"), code)

        assert len(entry_points) == 1
        # Public functions have lower confidence (0.7)
        assert entry_points[0].confidence == 0.7

    def test_multiple_entry_point_types(self):
        """Test detection of multiple entry point types in one file."""
        code = """
import click
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index():
    return "Hello"

@click.command()
def cli_command():
    print("CLI")

def test_something():
    assert True

if __name__ == "__main__":
    cli_command()
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("mixed.py"), code)

        # Should detect: route, CLI command, test, main
        assert (
            len(entry_points) >= 3
        )  # Might detect cli_command twice (decorator + main call)

        types = {ep.type for ep in entry_points}
        assert EntryPointType.ROUTE in types
        assert EntryPointType.CLI in types
        assert EntryPointType.TEST in types

    def test_syntax_error_handling(self):
        """Test that syntax errors are handled gracefully."""
        code = """
def broken_function(
    # Syntax error - unclosed parenthesis
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("broken.py"), code)

        # Should return empty list on syntax error, not crash
        assert entry_points == []

    def test_empty_file(self):
        """Test handling of empty files."""
        code = ""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("empty.py"), code)

        assert entry_points == []

    def test_detect_from_directory(self):
        """Test detection from directory with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "main.py").write_text(
                """
if __name__ == "__main__":
    main()
"""
            )
            (tmppath / "test_module.py").write_text(
                """
def test_function():
    assert True
"""
            )

            detector = EntryPointDetector()
            entry_points = detector.detect_from_directory(tmppath)

            # Should find entry points from both files
            assert len(entry_points) >= 2

            types = {ep.type for ep in entry_points}
            assert EntryPointType.MAIN in types
            assert EntryPointType.TEST in types

    def test_decorator_name_extraction(self):
        """Test extraction of decorator names in various formats."""
        code = """
@simple_decorator
def func1():
    pass

@module.decorator
def func2():
    pass

@decorator_with_call()
def func3():
    pass
"""
        detector = EntryPointDetector()

        # Test via internal method (white box testing)
        import ast

        tree = ast.parse(code)

        decorators = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    decorators.append(detector._extract_decorator_name(dec))

        assert "simple_decorator" in decorators
        assert "module.decorator" in decorators
        assert "decorator_with_call" in decorators

    def test_nested_functions_not_detected(self):
        """Test that nested functions are not detected as entry points."""
        code = """
def outer():
    def inner():
        pass
    return inner

if __name__ == "__main__":
    outer()
"""
        detector = EntryPointDetector()
        entry_points = detector.detect_from_file(Path("nested.py"), code)

        # Should only detect outer call in main, not inner function
        names = {ep.name for ep in entry_points}
        assert "outer" in names
        assert "inner" not in names
