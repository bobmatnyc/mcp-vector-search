"""Tests for Go parser."""

import pytest

from mcp_vector_search.parsers.go import GoParser


@pytest.fixture
def go_parser():
    """Create Go parser fixture."""
    return GoParser()


@pytest.fixture
def sample_go_code():
    """Sample Go code for testing."""
    return """
package main

import (
    "fmt"
    "errors"
)

// User represents a user entity.
type User struct {
    ID   int
    Name string
}

// UserRepository defines user repository interface.
type UserRepository interface {
    FindByID(id int) (*User, error)
    Save(user *User) error
}

// FindByID retrieves a user by ID.
func (u *User) FindByID(id int) (*User, error) {
    if id <= 0 {
        return nil, errors.New("invalid ID")
    }
    // Implementation here
    return u, nil
}

// NewUser creates a new user.
func NewUser(id int, name string) *User {
    return &User{
        ID:   id,
        Name: name,
    }
}
"""


def test_go_parser_initialization(go_parser):
    """Test Go parser initialization."""
    assert go_parser is not None
    assert go_parser.language == "go"
    assert ".go" in go_parser.get_supported_extensions()


@pytest.mark.asyncio
async def test_go_parser_basic(go_parser, sample_go_code, tmp_path):
    """Test basic Go parsing."""
    test_file = tmp_path / "test.go"
    test_file.write_text(sample_go_code)

    chunks = await go_parser.parse_file(test_file)

    assert len(chunks) > 0, "Should extract at least one chunk"

    # Check that we got types (struct, interface)
    type_chunks = [c for c in chunks if c.chunk_type in ["struct", "interface"]]
    assert len(type_chunks) >= 2, "Should find at least two type declarations"

    # Check struct
    struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
    assert len(struct_chunks) >= 1, "Should find at least one struct"
    struct_chunk = struct_chunks[0]
    assert struct_chunk.language == "go"
    assert struct_chunk.class_name == "User"


@pytest.mark.asyncio
async def test_go_parser_methods(go_parser, sample_go_code, tmp_path):
    """Test Go method extraction."""
    test_file = tmp_path / "test.go"
    test_file.write_text(sample_go_code)

    chunks = await go_parser.parse_file(test_file)

    # Find method chunks
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) >= 1, "Should find at least one method"

    # Check method details
    find_method = next(
        (c for c in method_chunks if c.function_name == "FindByID"), None
    )
    assert find_method is not None, "Should find FindByID method"
    assert find_method.class_name is not None, "Method should have a receiver type"
    assert find_method.complexity_score > 1.0, (
        "Should have complexity > 1 due to if statement"
    )


@pytest.mark.asyncio
async def test_go_parser_functions(go_parser, sample_go_code, tmp_path):
    """Test Go function extraction."""
    test_file = tmp_path / "test.go"
    test_file.write_text(sample_go_code)

    chunks = await go_parser.parse_file(test_file)

    # Find function chunks
    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(function_chunks) >= 1, "Should find at least one function"

    # Check function details
    new_user = next((c for c in function_chunks if c.function_name == "NewUser"), None)
    assert new_user is not None, "Should find NewUser function"
    assert len(new_user.parameters) >= 2, "Should extract parameters"


@pytest.mark.asyncio
async def test_go_parser_empty_file(go_parser, tmp_path):
    """Test parsing empty Go file."""
    test_file = tmp_path / "empty.go"
    test_file.write_text("")

    chunks = await go_parser.parse_file(test_file)
    assert len(chunks) == 0, "Empty file should produce no chunks"


@pytest.mark.asyncio
async def test_go_parser_complex_code(go_parser, tmp_path):
    """Test parsing more complex Go code."""
    complex_code = """
package service

import "context"

// Service provides business logic.
type Service struct {
    repo Repository
}

// NewService creates a new service.
func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

// Process processes a request.
func (s *Service) Process(ctx context.Context, id int) error {
    if id <= 0 {
        return errors.New("invalid ID")
    }

    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return err
    }

    // Process user
    return nil
}
"""

    test_file = tmp_path / "complex.go"
    test_file.write_text(complex_code)

    chunks = await go_parser.parse_file(test_file)

    # Should find struct, function, method
    types = {c.chunk_type for c in chunks}
    assert "struct" in types, "Should find struct"
    assert "function" in types, "Should find function"
    assert "method" in types, "Should find method"


def test_go_parser_supported_extensions(go_parser):
    """Test supported extensions."""
    extensions = go_parser.get_supported_extensions()
    assert ".go" in extensions
    assert len(extensions) == 1  # Only .go


@pytest.mark.asyncio
async def test_go_parser_registry_integration():
    """Test that Go parser is registered in the parser registry."""
    from mcp_vector_search.parsers.registry import get_parser_registry

    registry = get_parser_registry()

    # Test extension mapping
    language = registry.get_language_for_extension(".go")
    assert language == "go", f"Expected 'go' but got '{language}'"

    # Test parser retrieval
    parser = registry.get_parser(".go")
    assert parser.__class__.__name__ == "GoParser"
    assert parser.language == "go"

    # Test that Go is in supported languages
    supported_languages = registry.get_supported_languages()
    assert "go" in supported_languages
