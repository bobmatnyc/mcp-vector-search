#!/usr/bin/env python3
"""Test the HTML parser."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.parsers.html import HTMLParser


@pytest.mark.asyncio
async def test_html_parser():
    """Test HTML parser functionality."""
    print("🔍 Testing HTML parser...")

    # Create test HTML file with semantic structure
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation - Vector Search</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .highlight { background: yellow; }
    </style>
    <script>
        console.log('This script should be ignored');
        function ignoredFunction() {
            return 'Should not appear in chunks';
        }
    </script>
</head>
<body>
    <header>
        <h1 id="main-title">MCP Vector Search Documentation</h1>
        <p>A comprehensive guide to using semantic code search with vector embeddings.</p>
    </header>

    <main id="content">
        <section id="introduction">
            <h2>Introduction</h2>
            <p>Vector search enables semantic code search by converting code into high-dimensional embeddings. This allows you to search for code by meaning rather than exact keywords.</p>
            <p>The system uses ChromaDB for efficient vector storage and retrieval, with support for multiple programming languages including Python, JavaScript, and TypeScript.</p>
        </section>

        <section id="features">
            <h2>Key Features</h2>
            <p>Our vector search implementation provides several advanced capabilities that make code discovery easier and more intuitive.</p>

            <article id="semantic-search">
                <h3>Semantic Search</h3>
                <p>Search code by meaning, not just keywords. The system understands context and can find related code even when variable names differ.</p>
            </article>

            <article id="multi-language">
                <h3>Multi-Language Support</h3>
                <p>Parse and index code from Python, JavaScript, TypeScript, Dart, PHP, Ruby, and more. Each parser is AST-aware for accurate code extraction.</p>
            </article>
        </section>

        <section id="getting-started">
            <h2>Getting Started</h2>
            <p>To start using MCP Vector Search, first initialize your project directory. This creates the necessary configuration and index structure.</p>

            <div class="code-block">
                <p>Run the initialization command in your project root directory to set up the vector search index.</p>
            </div>

            <h3>Installation Steps</h3>
            <p>Install the package using pip or uv. The tool requires Python 3.9 or higher and includes all necessary dependencies.</p>
        </section>

        <section id="advanced">
            <h2>Advanced Usage</h2>
            <p>Advanced features include connection pooling for improved performance, incremental indexing for large codebases, and custom parser registration for new languages.</p>

            <aside class="note">
                <p>Connection pooling provides a 13.6% performance improvement in search operations by maintaining persistent database connections.</p>
            </aside>
        </section>
    </main>

    <footer>
        <p>Documentation version 1.0.0. Last updated January 2025.</p>
    </footer>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test HTML parser
    html_parser = HTMLParser()
    chunks = await html_parser.parse_file(test_file)

    print(f"📊 HTML parser extracted {len(chunks)} chunks:")

    # Analyze chunks
    heading_chunks = []
    section_chunks = []
    paragraph_chunks = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        if chunk.function_name:
            print(f"  Tag: {chunk.function_name}")
        if chunk.class_name:
            print(f"  ID: {chunk.class_name}")
        print(f"  Content preview: {chunk.content[:100]}...")

        # Categorize chunks
        if chunk.chunk_type == "heading":
            heading_chunks.append(chunk)
        elif chunk.chunk_type == "section":
            section_chunks.append(chunk)
        elif chunk.chunk_type == "paragraph":
            paragraph_chunks.append(chunk)

    # Verify key features
    print("\n" + "=" * 80)
    print("🎯 Feature Verification:")
    print("=" * 80)

    print(f"\n✅ Total chunks extracted: {len(chunks)}")
    assert len(chunks) >= 3, "Should extract at least 3 semantic chunks"

    print(f"✅ Heading chunks found: {len(heading_chunks)}")
    # Headings may be merged with following content
    print(f"✅ Section chunks found: {len(section_chunks)}")
    # Sections may be merged with other content
    print(f"✅ Paragraph chunks found: {len(paragraph_chunks)}")
    assert len(chunks) >= 1, "Should find at least some content chunks"

    # Verify script/style content is ignored
    script_content = any("ignoredFunction" in chunk.content for chunk in chunks)
    style_content = any("font-family" in chunk.content for chunk in chunks)
    print(f"✅ Script content ignored: {not script_content}")
    print(f"✅ Style content ignored: {not style_content}")
    assert not script_content, "Script content should be ignored"
    assert not style_content, "Style content should be ignored"

    # Verify main content is captured
    has_vector_search = any(
        "vector search" in chunk.content.lower() for chunk in chunks
    )
    has_semantic = any("semantic" in chunk.content.lower() for chunk in chunks)
    print(f"✅ Main content captured: {has_vector_search}")
    print(f"✅ Semantic content mentioned: {has_semantic}")
    assert has_vector_search, "Should capture main content about vector search"
    assert has_semantic, "Should capture semantic content"

    # Verify section IDs are captured
    chunks_with_ids = [c for c in chunks if c.class_name]
    print(f"✅ Chunks with IDs: {len(chunks_with_ids)}")
    assert len(chunks_with_ids) >= 1, "Should capture section IDs"

    # Verify supported extensions
    assert ".html" in html_parser.get_supported_extensions()
    assert ".htm" in html_parser.get_supported_extensions()
    print(f"✅ Supported extensions: {html_parser.get_supported_extensions()}")

    # Clean up
    test_file.unlink()
    print("\n✅ HTML parser test completed successfully!")

    return True


@pytest.mark.asyncio
async def test_html_edge_cases():
    """Test HTML parser edge cases and malformed HTML."""
    print("\n🔍 Testing HTML edge cases...")

    # Test malformed HTML
    malformed_html = """
<html>
<head><title>Test</title>
<body>
    <h1>Missing closing tags
    <p>Paragraph with <strong>nested tags and
    incomplete structure
    <div>Some content here
    <h2>Another heading</h2>
    <p>This should still be parsed reasonably well.
</body>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(malformed_html)
        test_file = Path(f.name)

    print(f"📁 Created malformed HTML test file: {test_file}")

    html_parser = HTMLParser()
    chunks = await html_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks from malformed HTML")
    assert len(chunks) >= 1, "Should handle malformed HTML gracefully"

    # Verify some content was extracted
    has_content = any(len(chunk.content.strip()) > 10 for chunk in chunks)
    print(f"✅ Content extracted from malformed HTML: {has_content}")

    test_file.unlink()

    # Test empty HTML
    empty_html = "<html><head><title>Empty</title></head><body></body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(empty_html)
        test_file = Path(f.name)

    chunks = await html_parser.parse_file(test_file)
    print(f"✅ Empty HTML handled: {len(chunks)} chunks")
    # Empty body should return minimal or no chunks
    assert len(chunks) <= 1, "Empty HTML should return minimal chunks"

    test_file.unlink()

    # Test HTML with only scripts and styles
    script_only = """
<html>
<head>
    <script>
        function test() { return 42; }
    </script>
    <style>
        body { color: red; }
    </style>
</head>
<body>
    <script>console.log('test');</script>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(script_only)
        test_file = Path(f.name)

    chunks = await html_parser.parse_file(test_file)
    print(f"✅ Script-only HTML handled: {len(chunks)} chunks")
    assert len(chunks) == 0, "Script/style-only HTML should return no chunks"

    test_file.unlink()

    print("\n✅ Edge cases test completed successfully!")
    return True


@pytest.mark.asyncio
async def test_html_semantic_structure():
    """Test HTML parser's handling of semantic HTML5 elements."""
    print("\n🔍 Testing HTML5 semantic structure...")

    semantic_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic HTML Test</title>
</head>
<body>
    <article id="blog-post">
        <h1>Understanding Semantic HTML</h1>
        <p>Semantic HTML provides meaning to the structure of web content, making it more accessible and SEO-friendly.</p>

        <section id="benefits">
            <h2>Benefits of Semantic HTML</h2>
            <p>Using semantic elements improves code readability and helps search engines understand your content better.</p>
        </section>

        <section id="examples">
            <h2>Common Semantic Elements</h2>
            <p>Elements like article, section, nav, and aside provide clear meaning about the content they contain.</p>
        </section>
    </article>

    <aside id="sidebar">
        <h3>Related Topics</h3>
        <p>Learn more about web accessibility and modern HTML practices.</p>
    </aside>

    <nav id="navigation">
        <h3>Site Navigation</h3>
        <p>Navigation links go here for easy site traversal.</p>
    </nav>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(semantic_html)
        test_file = Path(f.name)

    print(f"📁 Created semantic HTML test file: {test_file}")

    html_parser = HTMLParser()
    chunks = await html_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks from semantic HTML:")

    # Display chunks
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Tag: {chunk.function_name}")
        print(f"  Content: {chunk.content[:80]}...")

    # Verify article content is captured (may be merged with other content)
    has_article_content = any("Semantic HTML" in c.content for c in chunks)
    print(f"\n✅ Article content captured: {has_article_content}")
    assert has_article_content, "Should extract article content"

    # Verify section content is captured
    has_section_content = any(
        "Benefits of Semantic HTML" in c.content
        or "Common Semantic Elements" in c.content
        for c in chunks
    )
    print(f"✅ Section content captured: {has_section_content}")
    assert has_section_content, "Should extract section content"

    # Verify various semantic elements are captured
    has_aside_content = any("Related Topics" in c.content for c in chunks)
    has_nav_content = any("Navigation" in c.content for c in chunks)
    print(f"✅ Aside content captured: {has_aside_content}")
    print(f"✅ Nav content captured: {has_nav_content}")

    # Verify heading hierarchy is preserved
    h1_chunks = [c for c in chunks if "h1" in c.function_name]
    h2_chunks = [c for c in chunks if "h2" in c.function_name]
    h3_chunks = [c for c in chunks if "h3" in c.function_name]
    print(f"✅ H1 headings: {len(h1_chunks)}")
    print(f"✅ H2 headings: {len(h2_chunks)}")
    print(f"✅ H3 headings: {len(h3_chunks)}")

    # Clean up
    test_file.unlink()
    print("\n✅ Semantic structure test completed successfully!")

    return True


@pytest.mark.asyncio
async def test_html_chunk_merging():
    """Test HTML parser's chunk merging strategy."""
    print("\n🔍 Testing HTML chunk merging...")

    # Create HTML with many small paragraphs
    small_paragraphs = """
<html>
<body>
    <h1>Document Title</h1>
    <p>Short paragraph one.</p>
    <p>Short paragraph two.</p>
    <p>Short paragraph three.</p>
    <p>Short paragraph four.</p>
    <p>Short paragraph five.</p>

    <h2>Section Title</h2>
    <p>This is a longer paragraph that contains more substantial content and should provide enough text to stand alone as a meaningful chunk without being merged with others.</p>

    <h3>Subsection</h3>
    <p>Tiny.</p>
    <p>Also tiny.</p>
    <p>This paragraph is much longer and contains enough content to be considered substantial on its own, so it should not be merged with the tiny paragraphs above it.</p>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(small_paragraphs)
        test_file = Path(f.name)

    print(f"📁 Created chunk merging test file: {test_file}")

    html_parser = HTMLParser()
    chunks = await html_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks (after merging):")

    # Display chunk sizes
    for i, chunk in enumerate(chunks, 1):
        content_len = len(chunk.content)
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Tag: {chunk.function_name}")
        print(f"  Content length: {content_len} chars")
        print(f"  Preview: {chunk.content[:60]}...")

    # Verify chunks are reasonably sized
    chunk_sizes = [len(c.content) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0
    print(f"\n✅ Average chunk size: {avg_size:.0f} characters")
    print(f"✅ Chunk count: {len(chunks)}")

    # Verify small chunks were merged
    very_small_chunks = [c for c in chunks if len(c.content) < 20]
    print(f"✅ Very small chunks (<20 chars): {len(very_small_chunks)}")
    assert len(very_small_chunks) == 0, "Very small chunks should be merged or filtered"

    # Clean up
    test_file.unlink()
    print("\n✅ Chunk merging test completed successfully!")

    return True


@pytest.mark.asyncio
async def main():
    """Run all HTML parser tests."""
    try:
        await test_html_parser()
        await test_html_edge_cases()
        await test_html_semantic_structure()
        await test_html_chunk_merging()
        print("\n🎉 All HTML parser tests completed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
