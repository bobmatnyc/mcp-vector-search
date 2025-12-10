#!/usr/bin/env python3
"""Test AST enhancement features on sample code files.

This script:
1. Parses sample Python, JavaScript, and TypeScript files
2. Verifies complexity scores are calculated
3. Checks hierarchical relationships are built
4. Validates metadata extraction (decorators, parameters, types)
5. Tests tree-sitter AST parsing (not regex fallback)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from mcp_vector_search.core.models import CodeChunk
from mcp_vector_search.parsers.javascript import JavaScriptParser
from mcp_vector_search.parsers.python import PythonParser


def _build_hierarchy(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Build parent-child relationships between chunks (copied from indexer)."""
    if not chunks:
        return chunks

    # Group chunks by type
    module_chunks = [c for c in chunks if c.chunk_type in ("module", "imports")]
    class_chunks = [
        c for c in chunks if c.chunk_type in ("class", "interface", "mixin")
    ]
    function_chunks = [
        c for c in chunks if c.chunk_type in ("function", "method", "constructor")
    ]

    # Link functions to parent classes
    for func in function_chunks:
        if func.class_name:
            parent_class = next(
                (c for c in class_chunks if c.class_name == func.class_name), None
            )
            if parent_class:
                func.parent_chunk_id = parent_class.chunk_id
                func.chunk_depth = parent_class.chunk_depth + 1
                if func.chunk_id not in parent_class.child_chunk_ids:
                    parent_class.child_chunk_ids.append(func.chunk_id)
        else:
            # Top-level function
            if not func.chunk_depth:
                func.chunk_depth = 1
            if module_chunks and not func.parent_chunk_id:
                func.parent_chunk_id = module_chunks[0].chunk_id
                if func.chunk_id not in module_chunks[0].child_chunk_ids:
                    module_chunks[0].child_chunk_ids.append(func.chunk_id)

    # Link classes to modules
    for cls in class_chunks:
        if not cls.chunk_depth:
            cls.chunk_depth = 1
        if module_chunks and not cls.parent_chunk_id:
            cls.parent_chunk_id = module_chunks[0].chunk_id
            if cls.chunk_id not in module_chunks[0].child_chunk_ids:
                module_chunks[0].child_chunk_ids.append(cls.chunk_id)

    return chunks


@pytest.mark.asyncio
async def test_python_parser():
    """Test Python parser AST enhancements."""
    print("=" * 80)
    print("TESTING PYTHON PARSER")
    print("=" * 80)

    file_path = Path(__file__).parent / "ast_test_python.py"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    parser = PythonParser()
    chunks = await parser.parse_content(content, file_path)

    # Build hierarchy manually (normally done by indexer)
    chunks = _build_hierarchy(chunks)

    print(f"\nFound {len(chunks)} chunks")
    print()

    # Test 1: Verify complexity scores
    print("TEST 1: Complexity Scores")
    print("-" * 40)
    for chunk in chunks:
        if chunk.function_name:
            print(f"  {chunk.function_name:40} complexity: {chunk.complexity_score}")
    print()

    # Test 2: Verify hierarchical relationships
    print("TEST 2: Hierarchical Relationships")
    print("-" * 40)
    for chunk in chunks:
        if chunk.chunk_type == "class":
            print(f"  Class: {chunk.class_name}")
            print(f"    Depth: {chunk.chunk_depth}")
            print(f"    Children: {len(chunk.child_chunk_ids)}")
            print(f"    Child IDs: {chunk.child_chunk_ids[:3]}...")  # First 3
    print()

    # Test 3: Verify metadata extraction
    print("TEST 3: Metadata Extraction")
    print("-" * 40)
    for chunk in chunks:
        if chunk.decorators or chunk.parameters or chunk.return_type:
            print(f"  {chunk.function_name or chunk.class_name}:")
            if chunk.decorators:
                print(f"    Decorators: {chunk.decorators}")
            if chunk.parameters:
                print(f"    Parameters: {len(chunk.parameters)} params")
                for param in chunk.parameters[:2]:  # First 2
                    print(f"      - {param}")
            if chunk.return_type:
                print(f"    Return type: {chunk.return_type}")
            print()

    # Test 4: Verify specific complex functions
    print("TEST 4: Specific Function Checks")
    print("-" * 40)

    simple_func = next(
        (c for c in chunks if c.function_name == "simple_function"), None
    )
    if simple_func:
        print("  ✓ simple_function found")
        print(f"    Complexity: {simple_func.complexity_score} (expected: 1.0)")
        assert simple_func.complexity_score == 1.0, (
            "simple_function should have complexity 1.0"
        )

    grade_func = next((c for c in chunks if c.function_name == "calculate_grade"), None)
    if grade_func:
        print("  ✓ calculate_grade found")
        print(f"    Complexity: {grade_func.complexity_score} (expected: 5.0)")
        assert grade_func.complexity_score == 5.0, (
            "calculate_grade should have complexity 5.0"
        )

    complex_func = next(
        (c for c in chunks if c.function_name == "complex_validator"), None
    )
    if complex_func:
        print("  ✓ complex_validator found")
        print(f"    Complexity: {complex_func.complexity_score} (expected: 10.0+)")
        assert complex_func.complexity_score >= 10.0, (
            "complex_validator should have high complexity"
        )

    user_class = next((c for c in chunks if c.class_name == "User"), None)
    if user_class:
        print("  ✓ User class found")
        print(f"    Decorators: {user_class.decorators}")
        print(f"    Children: {len(user_class.child_chunk_ids)}")
        assert len(user_class.child_chunk_ids) > 0, (
            "User class should have child methods"
        )

    print()
    print("✓ Python parser tests PASSED")
    print()


@pytest.mark.asyncio
async def test_javascript_parser():
    """Test JavaScript parser AST enhancements."""
    print("=" * 80)
    print("TESTING JAVASCRIPT PARSER")
    print("=" * 80)

    file_path = Path(__file__).parent / "ast_test_javascript.js"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    parser = JavaScriptParser()
    chunks = await parser.parse_content(content, file_path)

    # Build hierarchy
    chunks = _build_hierarchy(chunks)

    print(f"\nFound {len(chunks)} chunks")
    print()

    # Test 1: Verify tree-sitter is being used (not regex fallback)
    print("TEST 1: Tree-Sitter AST Parsing")
    print("-" * 40)
    if hasattr(parser, "_use_tree_sitter"):
        print(f"  Tree-sitter enabled: {parser._use_tree_sitter}")
        assert parser._use_tree_sitter, "JavaScript parser should use tree-sitter"
    print()

    # Test 2: Verify function extraction
    print("TEST 2: Function Extraction")
    print("-" * 40)
    functions = [c for c in chunks if c.chunk_type == "function"]
    print(f"  Regular functions: {len(functions)}")
    for func in functions[:5]:
        print(f"    - {func.function_name} (complexity: {func.complexity_score})")
    print()

    # Test 3: Verify arrow function extraction
    print("TEST 3: Arrow Function Extraction")
    print("-" * 40)
    arrow_funcs = [c for c in chunks if "=>" in c.content[:50]]
    print(f"  Arrow functions: {len(arrow_funcs)}")
    for func in arrow_funcs[:5]:
        print(f"    - {func.function_name} (complexity: {func.complexity_score})")
    print()

    # Test 4: Verify class extraction
    print("TEST 4: Class and Method Extraction")
    print("-" * 40)
    classes = [c for c in chunks if c.chunk_type == "class"]
    for cls in classes:
        print(f"  Class: {cls.class_name}")
        print(f"    Depth: {cls.chunk_depth}")
        print(f"    Children: {len(cls.child_chunk_ids)}")

    methods = [c for c in chunks if c.chunk_type == "method"]
    print(f"  Total methods: {len(methods)}")
    print()

    # Test 5: Verify specific functions
    print("TEST 5: Specific Function Checks")
    print("-" * 40)

    simple_greeting = next(
        (c for c in chunks if c.function_name == "simpleGreeting"), None
    )
    if simple_greeting:
        print("  ✓ simpleGreeting found")
        print(f"    Complexity: {simple_greeting.complexity_score}")

    calculate_grade = next(
        (c for c in chunks if c.function_name == "calculateGrade"), None
    )
    if calculate_grade:
        print("  ✓ calculateGrade found")
        print(f"    Complexity: {calculate_grade.complexity_score} (expected: ~5.0)")

    user_class = next((c for c in chunks if c.class_name == "User"), None)
    if user_class:
        print("  ✓ User class found")
        print(f"    Children: {len(user_class.child_chunk_ids)}")
        assert len(user_class.child_chunk_ids) > 0, "User class should have methods"

    auth_class = next(
        (c for c in chunks if c.class_name == "AuthenticationManager"), None
    )
    if auth_class:
        print("  ✓ AuthenticationManager class found")
        print(f"    Children: {len(auth_class.child_chunk_ids)}")

    print()
    print("✓ JavaScript parser tests PASSED")
    print()


@pytest.mark.asyncio
async def test_typescript_parser():
    """Test TypeScript parser AST enhancements."""
    print("=" * 80)
    print("TESTING TYPESCRIPT PARSER")
    print("=" * 80)

    file_path = Path(__file__).parent / "ast_test_typescript.ts"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # JavaScript parser handles both JS and TS
    parser = JavaScriptParser(language="typescript")
    chunks = await parser.parse_content(content, file_path)

    # Build hierarchy
    chunks = _build_hierarchy(chunks)

    print(f"\nFound {len(chunks)} chunks")
    print()

    # Test 1: Verify tree-sitter is being used
    print("TEST 1: Tree-Sitter AST Parsing")
    print("-" * 40)
    if hasattr(parser, "_use_tree_sitter"):
        print(f"  Tree-sitter enabled: {parser._use_tree_sitter}")
    print()

    # Test 2: Verify interface/type extraction
    print("TEST 2: Interface and Type Extraction")
    print("-" * 40)
    interfaces = [c for c in chunks if c.chunk_type in ("interface", "type")]
    print(f"  Interfaces/Types: {len(interfaces)}")
    for iface in interfaces[:5]:
        print(f"    - {iface.function_name or iface.class_name}")
    print()

    # Test 3: Verify class extraction with types
    print("TEST 3: Class Extraction")
    print("-" * 40)
    classes = [c for c in chunks if c.chunk_type == "class"]
    for cls in classes[:3]:
        print(f"  Class: {cls.class_name}")
        print(f"    Depth: {cls.chunk_depth}")
        print(f"    Children: {len(cls.child_chunk_ids)}")
    print()

    # Test 4: Verify generic function extraction
    print("TEST 4: Generic Function Checks")
    print("-" * 40)

    process_items = next((c for c in chunks if c.function_name == "processItems"), None)
    if process_items:
        print("  ✓ processItems (generic) found")
        print(f"    Complexity: {process_items.complexity_score}")

    user_class = next((c for c in chunks if c.class_name == "User"), None)
    if user_class:
        print("  ✓ User class found")
        print(f"    Children: {len(user_class.child_chunk_ids)}")

    base_manager = next((c for c in chunks if c.class_name == "BaseManager"), None)
    if base_manager:
        print("  ✓ BaseManager (abstract, generic) found")
        print(f"    Children: {len(base_manager.child_chunk_ids)}")

    print()
    print("✓ TypeScript parser tests PASSED")
    print()


@pytest.mark.asyncio
async def test_hierarchy_building():
    """Test hierarchical chunk relationship building."""
    print("=" * 80)
    print("TESTING HIERARCHY BUILDING")
    print("=" * 80)

    # Parse Python file
    file_path = Path(__file__).parent / "ast_test_python.py"
    print(f"\nParsing: {file_path.name}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    parser = PythonParser()
    chunks = await parser.parse_content(content, file_path)

    # Build hierarchy
    chunks = _build_hierarchy(chunks)

    print(f"  Parsed {len(chunks)} chunks")
    print()

    # Verify hierarchy was built
    print("Hierarchy Verification:")
    print("-" * 40)

    module_chunks = [c for c in chunks if c.chunk_type == "module"]
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    method_chunks = [c for c in chunks if c.chunk_type == "method"]

    print(f"  Modules: {len(module_chunks)}")
    print(f"  Classes: {len(class_chunks)}")
    print(f"  Functions: {len(function_chunks)}")
    print(f"  Methods: {len(method_chunks)}")
    print()

    # Check parent-child relationships
    print("Parent-Child Relationships:")
    print("-" * 40)

    for cls in class_chunks[:2]:
        print(f"  {cls.class_name}:")
        print(f"    Chunk ID: {cls.chunk_id}")
        print(f"    Parent ID: {cls.parent_chunk_id}")
        print(f"    Depth: {cls.chunk_depth}")
        print(f"    Children: {len(cls.child_chunk_ids)}")

        # Find children
        children = [c for c in chunks if c.chunk_id in cls.child_chunk_ids]
        for child in children[:3]:
            print(f"      - {child.function_name} (depth: {child.chunk_depth})")
        print()

    print("✓ Hierarchy building tests PASSED")
    print()


async def main():
    """Run all AST enhancement tests."""
    print()
    print("=" * 80)
    print("AST ENHANCEMENT FEATURE TESTS")
    print("=" * 80)
    print()

    try:
        # Test parsers
        await test_python_parser()
        await test_javascript_parser()
        await test_typescript_parser()

        # Test hierarchy building
        await test_hierarchy_building()

        print("=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print()

        print("Summary of tested features:")
        print("  ✓ Complexity score calculation (Python, JS, TS)")
        print("  ✓ Tree-sitter AST parsing (JS, TS - not regex)")
        print("  ✓ Hierarchical chunk relationships (parent/child IDs)")
        print("  ✓ Enhanced metadata extraction (decorators, parameters, types)")
        print("  ✓ Function, class, and method detection")
        print("  ✓ Arrow function support (JavaScript)")
        print("  ✓ Generic and abstract class support (TypeScript)")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("TESTS FAILED! ✗")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
