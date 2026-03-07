"""Tests that each non-Python/JS parser populates CodeChunk.calls.

Each test:
1. Builds a minimal source snippet containing at least one known call.
2. Parses it with the parser under test.
3. Asserts the known callee name appears in chunk.calls for at least one chunk.

If a tree-sitter grammar is not installed the test is skipped gracefully.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from synchronous test code."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _all_calls(chunks) -> list[str]:
    """Flatten all calls from all chunks into a single list."""
    result = []
    for chunk in chunks:
        result.extend(chunk.calls or [])
    return result


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def test_go_parser_calls():
    """GoParser extracts call names from call_expression nodes."""
    try:
        from mcp_vector_search.parsers.go import GoParser
    except ImportError:
        pytest.skip("GoParser not importable")

    parser = GoParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-go grammar not installed")

    source = """\
package main

import "fmt"

func greet(name string) {
    fmt.Println("Hello", name)
    doWork()
}

func doWork() {}
"""
    chunks = _run(parser.parse_content(source, Path("test.go")))
    calls = _all_calls(chunks)

    # fmt.Println is a selector_expression call -> stored as "fmt.Println"
    # doWork is a plain identifier call
    assert any("Println" in c for c in calls), (
        f"Expected 'Println' in calls, got: {calls}"
    )
    assert "doWork" in calls, f"Expected 'doWork' in calls, got: {calls}"


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def test_rust_parser_calls():
    """RustParser extracts call names from call_expression and method_call_expression."""
    try:
        from mcp_vector_search.parsers.rust import RustParser
    except ImportError:
        pytest.skip("RustParser not importable")

    parser = RustParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-rust grammar not installed")

    source = """\
fn process(data: &str) -> String {
    let result = helper(data);
    result.to_uppercase()
}

fn helper(s: &str) -> String {
    s.to_string()
}
"""
    chunks = _run(parser.parse_content(source, Path("test.rs")))
    calls = _all_calls(chunks)

    assert "helper" in calls, f"Expected 'helper' in calls, got: {calls}"
    # method calls: to_uppercase, to_string
    assert any(c in ("to_uppercase", "to_string") for c in calls), (
        f"Expected method call in calls, got: {calls}"
    )


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------


def test_java_parser_calls():
    """JavaParser extracts method invocation names from method_invocation nodes."""
    try:
        from mcp_vector_search.parsers.java import JavaParser
    except ImportError:
        pytest.skip("JavaParser not importable")

    parser = JavaParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-java grammar not installed")

    source = """\
public class Greeter {
    public void greet(String name) {
        System.out.println("Hello " + name);
        validate(name);
    }

    private void validate(String name) {}
}
"""
    chunks = _run(parser.parse_content(source, Path("Greeter.java")))
    calls = _all_calls(chunks)

    assert "println" in calls, f"Expected 'println' in calls, got: {calls}"
    assert "validate" in calls, f"Expected 'validate' in calls, got: {calls}"


# ---------------------------------------------------------------------------
# C#
# ---------------------------------------------------------------------------


def test_csharp_parser_calls():
    """CSharpParser extracts invocation names from invocation_expression nodes."""
    try:
        from mcp_vector_search.parsers.csharp import CSharpParser
    except ImportError:
        pytest.skip("CSharpParser not importable")

    parser = CSharpParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-csharp grammar not installed")

    source = """\
using System;

public class Greeter
{
    public void Greet(string name)
    {
        Console.WriteLine("Hello " + name);
        Validate(name);
    }

    private void Validate(string name) { }
}
"""
    chunks = _run(parser.parse_content(source, Path("Greeter.cs")))
    calls = _all_calls(chunks)

    assert "WriteLine" in calls, f"Expected 'WriteLine' in calls, got: {calls}"
    assert "Validate" in calls, f"Expected 'Validate' in calls, got: {calls}"


# ---------------------------------------------------------------------------
# Ruby
# ---------------------------------------------------------------------------


def test_ruby_parser_calls():
    """RubyParser extracts method call names from call/method_call nodes."""
    try:
        from mcp_vector_search.parsers.ruby import RubyParser
    except ImportError:
        pytest.skip("RubyParser not importable")

    parser = RubyParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-ruby grammar not installed")

    source = """\
class Greeter
  def greet(name)
    puts "Hello #{name}"
    validate(name)
  end

  def validate(name)
    raise ArgumentError if name.empty?
  end
end
"""
    chunks = _run(parser.parse_content(source, Path("greeter.rb")))
    calls = _all_calls(chunks)

    assert "puts" in calls, f"Expected 'puts' in calls, got: {calls}"
    assert "validate" in calls, f"Expected 'validate' in calls, got: {calls}"


# ---------------------------------------------------------------------------
# PHP
# ---------------------------------------------------------------------------


def test_php_parser_calls():
    """PHPParser extracts call names from function_call_expression / member_call_expression."""
    try:
        from mcp_vector_search.parsers.php import PHPParser
    except ImportError:
        pytest.skip("PHPParser not importable")

    parser = PHPParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-php grammar not installed")

    source = """\
<?php

class Greeter {
    public function greet(string $name): void {
        echo strtoupper($name);
        $this->validate($name);
    }

    private function validate(string $name): void {}
}
"""
    chunks = _run(parser.parse_content(source, Path("Greeter.php")))
    calls = _all_calls(chunks)

    assert "strtoupper" in calls, f"Expected 'strtoupper' in calls, got: {calls}"
    assert "validate" in calls, f"Expected 'validate' in calls, got: {calls}"


# ---------------------------------------------------------------------------
# Dart
# ---------------------------------------------------------------------------


def test_dart_parser_calls():
    """DartParser extracts call names from function_expression_invocation / method_invocation."""
    try:
        from mcp_vector_search.parsers.dart import DartParser
    except ImportError:
        pytest.skip("DartParser not importable")

    parser = DartParser()
    if not parser._use_tree_sitter:
        pytest.skip("tree-sitter-dart grammar not installed")

    source = """\
void greet(String name) {
  print('Hello \\$name');
  validate(name);
}

void validate(String name) {
  assert(name.isNotEmpty);
}
"""
    chunks = _run(parser.parse_content(source, Path("greeter.dart")))
    calls = _all_calls(chunks)

    assert "print" in calls, f"Expected 'print' in calls, got: {calls}"
    assert "validate" in calls, f"Expected 'validate' in calls, got: {calls}"
