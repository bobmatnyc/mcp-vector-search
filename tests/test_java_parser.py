"""Tests for Java parser."""

import pytest

from mcp_vector_search.parsers.java import JavaParser


@pytest.fixture
def java_parser():
    """Create Java parser fixture."""
    return JavaParser()


@pytest.fixture
def sample_java_code():
    """Sample Java code for testing."""
    return """
package com.example;

import java.util.List;

/**
 * Sample class for testing.
 */
@Service
public class UserService {

    /**
     * Finds a user by ID.
     */
    public User findById(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("ID cannot be null");
        }
        return repository.findById(id);
    }

    public void save(User user) {
        repository.save(user);
    }
}
"""


def test_java_parser_initialization(java_parser):
    """Test Java parser initialization."""
    assert java_parser is not None
    assert java_parser.language == "java"
    assert ".java" in java_parser.get_supported_extensions()


@pytest.mark.asyncio
async def test_java_parser_basic(java_parser, sample_java_code, tmp_path):
    """Test basic Java parsing."""
    test_file = tmp_path / "test.java"
    test_file.write_text(sample_java_code)

    chunks = await java_parser.parse_file(test_file)

    assert len(chunks) > 0, "Should extract at least one chunk"

    # Check that we got a class
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1, "Should find at least one class"

    class_chunk = class_chunks[0]
    assert class_chunk.language == "java"
    assert class_chunk.class_name == "UserService"


@pytest.mark.asyncio
async def test_java_parser_methods(java_parser, sample_java_code, tmp_path):
    """Test Java method extraction."""
    test_file = tmp_path / "test.java"
    test_file.write_text(sample_java_code)

    chunks = await java_parser.parse_file(test_file)

    # Find method chunks
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) >= 2, "Should find at least two methods"

    # Check method details
    find_method = next(
        (c for c in method_chunks if c.function_name == "findById"), None
    )
    assert find_method is not None, "Should find findById method"
    assert find_method.return_type == "User"
    assert find_method.complexity_score > 1.0, (
        "Should have complexity > 1 due to if statement"
    )


@pytest.mark.asyncio
async def test_java_parser_annotations(java_parser, sample_java_code, tmp_path):
    """Test Java annotation extraction."""
    test_file = tmp_path / "test.java"
    test_file.write_text(sample_java_code)

    chunks = await java_parser.parse_file(test_file)

    # Check class annotations
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1

    class_chunk = class_chunks[0]
    assert len(class_chunk.decorators) > 0, "Should extract annotations"


@pytest.mark.asyncio
async def test_java_parser_empty_file(java_parser, tmp_path):
    """Test parsing empty Java file."""
    test_file = tmp_path / "empty.java"
    test_file.write_text("")

    chunks = await java_parser.parse_file(test_file)
    assert len(chunks) == 0, "Empty file should produce no chunks"


@pytest.mark.asyncio
async def test_java_parser_complex_code(java_parser, tmp_path):
    """Test parsing more complex Java code."""
    complex_code = """
package com.example;

import java.util.List;
import java.util.Optional;

/**
 * Repository for users.
 */
public interface UserRepository {
    Optional<User> findById(Long id);
    List<User> findAll();
}

/**
 * User entity.
 */
@Entity
@Table(name = "users")
public class User {
    @Id
    private Long id;
    private String name;

    public User(Long id, String name) {
        this.id = id;
        this.name = name;
    }

    public Long getId() {
        return id;
    }
}

/**
 * User roles.
 */
public enum Role {
    ADMIN,
    USER
}
"""

    test_file = tmp_path / "complex.java"
    test_file.write_text(complex_code)

    chunks = await java_parser.parse_file(test_file)

    # Should find interface, class, enum
    types = {c.chunk_type for c in chunks}
    assert "interface" in types, "Should find interface"
    assert "class" in types, "Should find class"
    assert "enum" in types, "Should find enum"

    # Should find constructor
    constructors = [c for c in chunks if c.chunk_type == "constructor"]
    assert len(constructors) >= 1, "Should find constructor"


def test_java_parser_supported_extensions(java_parser):
    """Test supported extensions."""
    extensions = java_parser.get_supported_extensions()
    assert ".java" in extensions
    assert len(extensions) == 1  # Only .java


@pytest.mark.asyncio
async def test_java_parser_registry_integration():
    """Test that Java parser is registered in the parser registry."""
    from mcp_vector_search.parsers.registry import get_parser_registry

    registry = get_parser_registry()

    # Test extension mapping
    language = registry.get_language_for_extension(".java")
    assert language == "java", f"Expected 'java' but got '{language}'"

    # Test parser retrieval
    parser = registry.get_parser(".java")
    assert parser.__class__.__name__ == "JavaParser"
    assert parser.language == "java"

    # Test that Java is in supported languages
    supported_languages = registry.get_supported_languages()
    assert "java" in supported_languages
