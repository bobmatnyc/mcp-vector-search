#!/usr/bin/env python3
"""Verification script to check framework detection methods were added correctly."""

import ast
import sys
from pathlib import Path


def verify_framework_detection():
    """Verify all framework detection methods are present."""
    kg_builder_path = Path("src/mcp_vector_search/core/kg_builder.py")

    if not kg_builder_path.exists():
        print(f"❌ File not found: {kg_builder_path}")
        return False

    with open(kg_builder_path) as f:
        tree = ast.parse(f.read())

    # Find all method definitions
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            methods.append(node.name)

    # Check for framework detection methods
    expected_methods = [
        "_detect_python_frameworks",
        "_detect_javascript_frameworks",
        "_detect_rust_frameworks",
        "_detect_go_frameworks",
        "_detect_java_frameworks",
        "_detect_ruby_frameworks",
        "_detect_php_frameworks",
        "_detect_csharp_frameworks",
        "_detect_swift_frameworks",
        "_detect_kotlin_frameworks",
    ]

    print("Framework Detection Methods Verification")
    print("=" * 60)

    all_present = True
    for method in expected_methods:
        if method in methods:
            print(f"✓ {method}")
        else:
            print(f"❌ {method} - MISSING")
            all_present = False

    # Verify they're called in _extract_languages_and_frameworks
    print("\n" + "=" * 60)
    print("Checking method calls in _extract_languages_and_frameworks")
    print("=" * 60)

    with open(kg_builder_path) as f:
        content = f.read()

    expected_calls = [
        "python_frameworks = await self._detect_python_frameworks()",
        "js_frameworks = await self._detect_javascript_frameworks()",
        "rust_frameworks = await self._detect_rust_frameworks()",
        "go_frameworks = await self._detect_go_frameworks()",
        "java_frameworks = await self._detect_java_frameworks()",
        "ruby_frameworks = await self._detect_ruby_frameworks()",
        "php_frameworks = await self._detect_php_frameworks()",
        "csharp_frameworks = await self._detect_csharp_frameworks()",
        "swift_frameworks = await self._detect_swift_frameworks()",
        "kotlin_frameworks = await self._detect_kotlin_frameworks()",
    ]

    for call in expected_calls:
        if call in content:
            print(f"✓ {call.split('=')[0].strip()}")
        else:
            print(f"❌ {call} - MISSING")
            all_present = False

    print("\n" + "=" * 60)
    if all_present:
        print("✅ All framework detection methods are present and called!")
        return True
    else:
        print("❌ Some methods are missing or not called")
        return False


def verify_framework_patterns():
    """Verify framework patterns for each ecosystem."""
    kg_builder_path = Path("src/mcp_vector_search/core/kg_builder.py")

    with open(kg_builder_path) as f:
        content = f.read()

    print("\n" + "=" * 60)
    print("Framework Patterns Verification")
    print("=" * 60)

    ecosystems = {
        "Java": ["spring-boot", "hibernate", "junit", "log4j", "jackson"],
        "Ruby": ["rails", "rspec", "sinatra", "sidekiq", "activerecord"],
        "PHP": ["laravel", "symfony", "phpunit", "doctrine", "guzzle"],
        "C#/.NET": ["microsoft.aspnetcore", "entityframework", "xunit", "serilog"],
        "Swift": ["vapor", "swiftui", "alamofire", "combine"],
        "Kotlin": ["ktor", "spring", "exposed", "koin", "coroutines"],
    }

    for ecosystem, patterns in ecosystems.items():
        found_patterns = [p for p in patterns if p in content.lower()]
        print(f"\n{ecosystem}: {len(found_patterns)}/{len(patterns)} patterns found")
        for pattern in patterns:
            if pattern in content.lower():
                print(f"  ✓ {pattern}")
            else:
                print(f"  ❌ {pattern}")


def verify_config_file_parsing():
    """Verify correct config files are parsed."""
    kg_builder_path = Path("src/mcp_vector_search/core/kg_builder.py")

    with open(kg_builder_path) as f:
        content = f.read()

    print("\n" + "=" * 60)
    print("Configuration File Parsing Verification")
    print("=" * 60)

    config_files = {
        "Java": ["pom.xml", "build.gradle"],
        "Ruby": ["Gemfile"],
        "PHP": ["composer.json"],
        "C#/.NET": ["*.csproj"],
        "Swift": ["Package.swift"],
        "Kotlin": ["build.gradle.kts"],
    }

    for ecosystem, files in config_files.items():
        print(f"\n{ecosystem}:")
        for file in files:
            if file in content:
                print(f"  ✓ {file}")
            else:
                print(f"  ❌ {file}")


if __name__ == "__main__":
    success = verify_framework_detection()
    verify_framework_patterns()
    verify_config_file_parsing()

    sys.exit(0 if success else 1)
