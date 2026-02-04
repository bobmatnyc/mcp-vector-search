"""Tests for Rust parser."""

import pytest

from mcp_vector_search.parsers.rust import RustParser


@pytest.fixture
def rust_parser():
    """Create Rust parser fixture."""
    return RustParser()


@pytest.fixture
def sample_rust_code():
    """Sample Rust code for testing."""
    return """
use std::error::Error;

/// User represents a user entity.
#[derive(Debug, Clone)]
pub struct User {
    pub id: u32,
    pub name: String,
}

/// User repository trait.
pub trait UserRepository {
    fn find_by_id(&self, id: u32) -> Result<User, Box<dyn Error>>;
    fn save(&mut self, user: User) -> Result<(), Box<dyn Error>>;
}

impl User {
    /// Creates a new user.
    pub fn new(id: u32, name: String) -> Self {
        User { id, name }
    }

    /// Validates the user.
    pub fn validate(&self) -> Result<(), String> {
        if self.id == 0 {
            return Err("ID cannot be zero".to_string());
        }
        if self.name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Creates a new user with validation.
pub fn create_user(id: u32, name: String) -> Result<User, String> {
    let user = User::new(id, name);
    user.validate()?;
    Ok(user)
}
"""


def test_rust_parser_initialization(rust_parser):
    """Test Rust parser initialization."""
    assert rust_parser is not None
    assert rust_parser.language == "rust"
    assert ".rs" in rust_parser.get_supported_extensions()


@pytest.mark.asyncio
async def test_rust_parser_basic(rust_parser, sample_rust_code, tmp_path):
    """Test basic Rust parsing."""
    test_file = tmp_path / "test.rs"
    test_file.write_text(sample_rust_code)

    chunks = await rust_parser.parse_file(test_file)

    assert len(chunks) > 0, "Should extract at least one chunk"

    # Check that we got types (struct, trait, impl)
    type_chunks = [c for c in chunks if c.chunk_type in ["struct", "trait", "impl"]]
    assert len(type_chunks) >= 3, "Should find at least three type declarations"

    # Check struct
    struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
    assert len(struct_chunks) >= 1, "Should find at least one struct"
    struct_chunk = struct_chunks[0]
    assert struct_chunk.language == "rust"
    assert struct_chunk.class_name == "User"


@pytest.mark.asyncio
async def test_rust_parser_impl_blocks(rust_parser, sample_rust_code, tmp_path):
    """Test Rust impl block extraction."""
    test_file = tmp_path / "test.rs"
    test_file.write_text(sample_rust_code)

    chunks = await rust_parser.parse_file(test_file)

    # Find impl chunks
    impl_chunks = [c for c in chunks if c.chunk_type == "impl"]
    assert len(impl_chunks) >= 1, "Should find at least one impl block"

    impl_chunk = impl_chunks[0]
    assert impl_chunk.class_name == "User", "Impl block should reference User"


@pytest.mark.asyncio
async def test_rust_parser_methods(rust_parser, sample_rust_code, tmp_path):
    """Test Rust method extraction."""
    test_file = tmp_path / "test.rs"
    test_file.write_text(sample_rust_code)

    chunks = await rust_parser.parse_file(test_file)

    # Find method chunks
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) >= 2, "Should find at least two methods"

    # Check method details
    validate_method = next(
        (c for c in method_chunks if c.function_name == "validate"), None
    )
    assert validate_method is not None, "Should find validate method"
    assert validate_method.complexity_score > 1.0, (
        "Should have complexity > 1 due to if statements"
    )


@pytest.mark.asyncio
async def test_rust_parser_functions(rust_parser, sample_rust_code, tmp_path):
    """Test Rust function extraction."""
    test_file = tmp_path / "test.rs"
    test_file.write_text(sample_rust_code)

    chunks = await rust_parser.parse_file(test_file)

    # Find function chunks
    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(function_chunks) >= 1, "Should find at least one function"

    # Check function details
    create_user = next(
        (c for c in function_chunks if c.function_name == "create_user"), None
    )
    assert create_user is not None, "Should find create_user function"


@pytest.mark.asyncio
async def test_rust_parser_attributes(rust_parser, sample_rust_code, tmp_path):
    """Test Rust attribute extraction."""
    test_file = tmp_path / "test.rs"
    test_file.write_text(sample_rust_code)

    chunks = await rust_parser.parse_file(test_file)

    # Check struct attributes
    struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
    assert len(struct_chunks) >= 1
    struct_chunk = struct_chunks[0]
    assert len(struct_chunk.decorators) > 0, "Should extract attributes"


@pytest.mark.asyncio
async def test_rust_parser_empty_file(rust_parser, tmp_path):
    """Test parsing empty Rust file."""
    test_file = tmp_path / "empty.rs"
    test_file.write_text("")

    chunks = await rust_parser.parse_file(test_file)
    assert len(chunks) == 0, "Empty file should produce no chunks"


@pytest.mark.asyncio
async def test_rust_parser_complex_code(rust_parser, tmp_path):
    """Test parsing more complex Rust code."""
    complex_code = """
use std::collections::HashMap;

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Error enum.
#[derive(Debug)]
pub enum Error {
    NotFound,
    InvalidInput(String),
}

/// Database trait.
pub trait Database {
    fn query(&self, sql: &str) -> Result<Vec<String>>;
}

/// In-memory database implementation.
pub struct InMemoryDB {
    data: HashMap<String, String>,
}

impl InMemoryDB {
    /// Creates a new in-memory database.
    pub fn new() -> Self {
        InMemoryDB {
            data: HashMap::new(),
        }
    }
}

impl Database for InMemoryDB {
    fn query(&self, sql: &str) -> Result<Vec<String>> {
        // Implementation
        Ok(vec![])
    }
}
"""

    test_file = tmp_path / "complex.rs"
    test_file.write_text(complex_code)

    chunks = await rust_parser.parse_file(test_file)

    # Should find enum, trait, struct, impl
    types = {c.chunk_type for c in chunks}
    assert "enum" in types, "Should find enum"
    assert "trait" in types, "Should find trait"
    assert "struct" in types, "Should find struct"
    assert "impl" in types, "Should find impl"


def test_rust_parser_supported_extensions(rust_parser):
    """Test supported extensions."""
    extensions = rust_parser.get_supported_extensions()
    assert ".rs" in extensions
    assert len(extensions) == 1  # Only .rs


@pytest.mark.asyncio
async def test_rust_parser_registry_integration():
    """Test that Rust parser is registered in the parser registry."""
    from mcp_vector_search.parsers.registry import get_parser_registry

    registry = get_parser_registry()

    # Test extension mapping
    language = registry.get_language_for_extension(".rs")
    assert language == "rust", f"Expected 'rust' but got '{language}'"

    # Test parser retrieval
    parser = registry.get_parser(".rs")
    assert parser.__class__.__name__ == "RustParser"
    assert parser.language == "rust"

    # Test that Rust is in supported languages
    supported_languages = registry.get_supported_languages()
    assert "rust" in supported_languages
