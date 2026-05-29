"""Tests for MATLAB parser."""

import pytest

from mcp_vector_search.parsers.matlab import MatlabParser


@pytest.fixture
def matlab_parser():
    """Create MATLAB parser fixture."""
    return MatlabParser()


@pytest.fixture
def sample_function_code():
    """A MATLAB file with two top-level (local) functions."""
    return """function [a, b] = foo(x, y)
% FOO summary line
%  more detailed help
if x > 0
    a = bar(x);
else
    a = 0;
end
b = helper(y);
end

function z = helper(v)
z = v * 2;
end
"""


@pytest.fixture
def sample_class_code():
    """A MATLAB classdef with inheritance, properties, and methods."""
    return """classdef Shape < handle & matlab.mixin.Copyable
% SHAPE base class for shapes
    properties
        Name
        Area = 0
    end
    methods
        function obj = Shape(name)
            obj.Name = name;
        end
        function r = describe(obj, verbose)
            % describe the shape
            if verbose && obj.Area > 0
                r = sprintf('%s: %d', obj.Name, obj.Area);
            else
                r = obj.Name;
            end
        end
    end
end
"""


def test_matlab_parser_initialization(matlab_parser):
    """Test MATLAB parser initialization."""
    assert matlab_parser is not None
    assert matlab_parser.language == "matlab"
    assert ".m" in matlab_parser.get_supported_extensions()


@pytest.mark.asyncio
async def test_matlab_parser_functions(matlab_parser, sample_function_code, tmp_path):
    """Test MATLAB function extraction."""
    test_file = tmp_path / "foo.m"
    test_file.write_text(sample_function_code)

    chunks = await matlab_parser.parse_file(test_file)

    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(function_chunks) >= 2, "Should find both top-level functions"

    foo = next((c for c in function_chunks if c.function_name == "foo"), None)
    assert foo is not None, "Should find foo function"
    assert foo.language == "matlab"
    # MATLAB outputs are reported via return_type.
    assert foo.return_type == "[a, b]"
    assert [p["name"] for p in foo.parameters] == ["x", "y"]
    # if/else introduces one decision point on top of the base complexity.
    assert foo.complexity_score > 1.0
    # Help comment becomes the docstring (% markers stripped).
    assert foo.docstring is not None
    assert "FOO summary line" in foo.docstring
    # Calls to other functions are captured.
    assert "helper" in (foo.calls or [])

    helper = next((c for c in function_chunks if c.function_name == "helper"), None)
    assert helper is not None, "Should find helper function"
    assert helper.return_type == "z"


@pytest.mark.asyncio
async def test_matlab_parser_class(matlab_parser, sample_class_code, tmp_path):
    """Test MATLAB classdef extraction."""
    test_file = tmp_path / "Shape.m"
    test_file.write_text(sample_class_code)

    chunks = await matlab_parser.parse_file(test_file)

    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1, "Should find exactly one class"

    shape = class_chunks[0]
    assert shape.class_name == "Shape"
    assert shape.language == "matlab"
    # Both superclasses (including the dotted one) are captured.
    assert "handle" in shape.inherits_from
    assert "matlab.mixin.Copyable" in shape.inherits_from
    assert shape.docstring is not None
    assert "base class for shapes" in shape.docstring


@pytest.mark.asyncio
async def test_matlab_parser_methods(matlab_parser, sample_class_code, tmp_path):
    """Test MATLAB method extraction with class context."""
    test_file = tmp_path / "Shape.m"
    test_file.write_text(sample_class_code)

    chunks = await matlab_parser.parse_file(test_file)

    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) >= 2, "Should find both methods"
    # Methods carry their enclosing class name.
    assert all(c.class_name == "Shape" for c in method_chunks)

    describe = next((c for c in method_chunks if c.function_name == "describe"), None)
    assert describe is not None, "Should find describe method"
    assert describe.return_type == "r"
    assert [p["name"] for p in describe.parameters] == ["obj", "verbose"]
    # if + && short-circuit add two decision points above the base.
    assert describe.complexity_score >= 3.0


@pytest.mark.asyncio
async def test_matlab_parser_script(matlab_parser, tmp_path):
    """A script (no function/class) yields a single script chunk."""
    test_file = tmp_path / "script.m"
    test_file.write_text("x = 1;\ny = x + 2;\ndisp(y)\n")

    chunks = await matlab_parser.parse_file(test_file)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "script"
    assert chunks[0].language == "matlab"


@pytest.mark.asyncio
async def test_matlab_parser_empty(matlab_parser, tmp_path):
    """An empty file yields no chunks."""
    test_file = tmp_path / "empty.m"
    test_file.write_text("")

    chunks = await matlab_parser.parse_file(test_file)
    assert chunks == []


@pytest.mark.asyncio
async def test_matlab_parser_regex_fallback(matlab_parser, sample_class_code):
    """The regex fallback extracts definitions when tree-sitter is unavailable."""
    from pathlib import Path

    chunks = await matlab_parser._regex_parse(sample_class_code, Path("Shape.m"))

    # Fallback should at least find the class and the two methods by header.
    names = {c.class_name for c in chunks if c.chunk_type == "class"}
    assert "Shape" in names
    func_names = {c.function_name for c in chunks if c.chunk_type == "function"}
    assert {"Shape", "describe"} <= func_names


def test_matlab_parser_registry_integration():
    """Test that the MATLAB parser is registered in the parser registry."""
    from mcp_vector_search.parsers.registry import get_parser_registry

    registry = get_parser_registry()

    assert registry.get_language_for_extension(".m") == "matlab"

    parser = registry.get_parser(".m")
    assert parser.__class__.__name__ == "MatlabParser"
    assert parser.language == "matlab"

    assert "matlab" in registry.get_supported_languages()
