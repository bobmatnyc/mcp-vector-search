"""Temporal Knowledge Graph using Kuzu.

Tracks code relationships over time:
- Function calls (A calls B)
- Imports (module A imports module B)
- Inheritance (class A extends B)
- Contains (file F contains class C)

Enables architectural queries like:
- "What functions does X call?"
- "What inherits from class Y?"
- "Show me the call graph for function Z"
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kuzu
from loguru import logger


@dataclass
class CodeEntity:
    """A node in the knowledge graph."""

    id: str  # Unique identifier (chunk_id)
    name: str  # Display name (function/class name)
    entity_type: str  # file, class, function, module
    file_path: str  # Source file path
    commit_sha: str | None = None  # Git commit for temporal tracking


@dataclass
class DocSection:
    """A documentation section node in the knowledge graph."""

    id: str  # Unique identifier (chunk_id)
    name: str  # Section title (markdown header)
    file_path: str  # Source file path
    level: int  # Header level (1-6 for markdown)
    line_start: int  # Starting line number
    line_end: int  # Ending line number
    doc_type: str = "section"  # section, topic
    commit_sha: str | None = None  # Git commit for temporal tracking


@dataclass
class Document:
    """A file-level document node in the knowledge graph."""

    id: str  # Unique identifier (document:<file_path_hash>)
    file_path: str  # Relative file path
    title: str  # First H1 heading, or filename
    doc_category: str  # readme, guide, api_doc, config, changelog, design, spec, other
    word_count: int  # Total word count
    section_count: int  # Number of heading sections
    last_modified: str | None = None  # ISO timestamp
    commit_sha: str | None = None


@dataclass
class Tag:
    """A tag node for topic clustering in the knowledge graph."""

    id: str  # Unique identifier (tag:name)
    name: str  # Tag name


@dataclass
class Person:
    """A person node for authorship tracking in the knowledge graph."""

    id: str  # Unique identifier (person:<email_hash>)
    name: str  # Display name from git
    email_hash: str  # SHA256 of email for privacy
    commits_count: int = 0  # Total commits
    first_commit: str | None = None  # ISO timestamp of first commit
    last_commit: str | None = None  # ISO timestamp of last commit


@dataclass
class Project:
    """A project node representing the codebase in the knowledge graph."""

    id: str  # Unique identifier (project:<name>)
    name: str  # Project name
    description: str = ""  # Project description
    repo_url: str = ""  # Git repository URL


@dataclass
class Repository:
    """A repository node for version control tracking in the knowledge graph."""

    id: str  # Unique identifier (repo:<name>)
    name: str  # Repository name
    url: str = ""  # Git remote URL
    default_branch: str = "main"  # Default branch name
    created_at: str | None = None  # ISO timestamp of first commit


@dataclass
class Branch:
    """A branch node for version control tracking in the knowledge graph."""

    id: str  # Unique identifier (branch:<repo_name>:<branch_name>)
    name: str  # Branch name
    repository_id: str = ""  # Parent repository ID
    is_default: bool = False  # Whether this is the default branch
    created_at: str | None = None  # ISO timestamp of first commit on branch


@dataclass
class Commit:
    """A commit node for version control tracking in the knowledge graph."""

    id: str  # Unique identifier (commit:<sha>)
    sha: str  # Git commit SHA (full 40-char hash)
    message: str = ""  # Commit message
    author_name: str = ""  # Commit author name
    author_email: str = ""  # Commit author email
    timestamp: str | None = None  # ISO timestamp of commit


@dataclass
class ProgrammingLanguage:
    """A programming language node in the knowledge graph."""

    id: str  # Unique identifier (lang:<name>)
    name: str  # Language name (Python, JavaScript, TypeScript, etc.)
    version: str = ""  # Version if detected (e.g., "3.12", "ES2022")
    file_extensions: str = ""  # Comma-separated list (e.g., ".py,.pyi")


@dataclass
class ProgrammingFramework:
    """A programming framework node in the knowledge graph."""

    id: str  # Unique identifier (framework:<name>)
    name: str  # Framework name (FastAPI, React, pytest, etc.)
    version: str = ""  # Version from package manifest
    language_id: str = ""  # Reference to ProgrammingLanguage node
    category: str = ""  # web, testing, orm, cli, etc.


@dataclass
class CodeRelationship:
    """An edge in the knowledge graph."""

    source_id: str  # Source entity ID
    target_id: str  # Target entity ID
    relationship_type: str  # calls, imports, inherits, contains, references, documents, follows, has_tag, demonstrates, links_to
    commit_sha: str | None = None  # Git commit
    weight: float = 1.0  # Relationship strength


class KnowledgeGraph:
    """Kuzu-based knowledge graph for code relationships.

    This class manages a temporal knowledge graph that tracks:
    - Code entities (files, classes, functions, modules)
    - Relationships between entities (calls, imports, inheritance)
    - Temporal changes via git commits

    The graph enables powerful architectural queries like:
    - Finding all callers of a function
    - Traversing inheritance hierarchies
    - Analyzing import dependencies
    """

    def __init__(self, db_path: Path):
        """Initialize Kuzu database.

        Args:
            db_path: Path to store Kuzu database files
        """
        self.db_path = db_path
        self.db = None
        self.conn = None
        self._initialized = False

        # Thread lock for Kuzu operations safety
        # We run Kuzu operations synchronously on the main asyncio thread
        # but use a lock to prevent any potential concurrent access
        self._kuzu_lock = threading.Lock()

    def initialize_sync(self):
        """Initialize Kuzu database with schema (synchronous, thread-safe version)."""
        if self._initialized:
            return

        # Ensure parent directory exists, but let Kuzu create the db directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Kuzu database and schema synchronously
        # CRITICAL: This must run in a single-threaded environment
        # Kuzu's Rust bindings are not thread-safe
        db_dir = self.db_path / "code_kg"
        with self._kuzu_lock:
            self.db = kuzu.Database(str(db_dir))
            self.conn = kuzu.Connection(self.db)
            logger.debug(
                f"Kuzu database and connection created in thread {threading.current_thread().name}"
            )

            # Create schema directly
            self._create_schema()

        self._initialized = True
        logger.info(f"✓ Knowledge graph initialized at {db_dir}")

    async def initialize(self):
        """Initialize Kuzu database with schema."""
        if self._initialized:
            return

        # Ensure parent directory exists, but let Kuzu create the db directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Kuzu database and schema synchronously
        # Run directly on the main asyncio thread with lock for safety
        db_dir = self.db_path / "code_kg"
        with self._kuzu_lock:
            self.db = kuzu.Database(str(db_dir))
            self.conn = kuzu.Connection(self.db)
            logger.debug(
                f"Kuzu database and connection created in thread {threading.current_thread().name}"
            )

            # Create schema directly
            self._create_schema()

        self._initialized = True
        logger.info(f"✓ Knowledge graph initialized at {db_dir}")

    def _execute_query(self, query: str, params: dict | None = None):
        """Execute Kuzu query synchronously with thread lock.

        Runs Kuzu operations directly on the calling thread (main asyncio thread)
        with a lock to prevent concurrent access. This eliminates the multi-threading
        complexity that may cause segfaults.

        Args:
            query: Cypher query to execute
            params: Optional query parameters

        Returns:
            Query result from Kuzu
        """
        with self._kuzu_lock:
            return self.conn.execute(query, params or {})

    def _create_schema(self):
        """Create Kuzu schema for code entities, doc sections, and relationships."""
        # Create CodeEntity node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS CodeEntity (
                    id STRING PRIMARY KEY,
                    name STRING,
                    entity_type STRING,
                    file_path STRING,
                    commit_sha STRING,
                    created_at TIMESTAMP DEFAULT current_timestamp()
                )
            """
            )
            logger.debug("Created CodeEntity node table")
        except Exception as e:
            # Table likely exists
            logger.debug(f"CodeEntity table creation: {e}")

        # Create DocSection node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS DocSection (
                    id STRING PRIMARY KEY,
                    name STRING,
                    doc_type STRING,
                    file_path STRING,
                    level INT64,
                    line_start INT64,
                    line_end INT64,
                    commit_sha STRING,
                    created_at TIMESTAMP DEFAULT current_timestamp()
                )
            """
            )
            logger.debug("Created DocSection node table")
        except Exception as e:
            # Table likely exists
            logger.debug(f"DocSection table creation: {e}")

        # Create Tag node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Tag (
                    id STRING PRIMARY KEY,
                    name STRING
                )
            """
            )
            logger.debug("Created Tag node table")
        except Exception as e:
            # Table likely exists
            logger.debug(f"Tag table creation: {e}")

        # Create Document node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Document (
                    id STRING PRIMARY KEY,
                    file_path STRING,
                    title STRING,
                    doc_category STRING,
                    word_count INT64,
                    section_count INT64,
                    last_modified STRING,
                    commit_sha STRING,
                    created_at TIMESTAMP DEFAULT current_timestamp()
                )
            """
            )
            logger.debug("Created Document node table")
        except Exception as e:
            logger.debug(f"Document table creation: {e}")

        # Create Topic node table for hierarchical taxonomy
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Topic (
                    id STRING PRIMARY KEY,
                    name STRING,
                    parent_id STRING
                )
            """
            )
            logger.debug("Created Topic node table")
        except Exception as e:
            logger.debug(f"Topic table creation: {e}")

        # Create relationship tables for code-to-code relationships
        code_relationship_types = ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]
        for rel_type in code_relationship_types:
            try:
                self.conn.execute(
                    f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_type} (
                        FROM CodeEntity TO CodeEntity,
                        weight DOUBLE DEFAULT 1.0,
                        commit_sha STRING,
                        MANY_MANY
                    )
                """
                )
                logger.debug(f"Created {rel_type} relationship table")
            except Exception as e:
                # Table likely exists
                logger.debug(f"{rel_type} table creation: {e}")

        # Create relationship tables for doc-to-doc relationships
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS FOLLOWS (
                    FROM DocSection TO DocSection,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created FOLLOWS relationship table")
        except Exception as e:
            logger.debug(f"FOLLOWS table creation: {e}")

        # Create relationship tables for doc-to-code relationships
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS REFERENCES (
                    FROM DocSection TO CodeEntity,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created REFERENCES relationship table")
        except Exception as e:
            logger.debug(f"REFERENCES table creation: {e}")

        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS DOCUMENTS (
                    FROM DocSection TO CodeEntity,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created DOCUMENTS relationship table")
        except Exception as e:
            logger.debug(f"DOCUMENTS table creation: {e}")

        # Create HAS_TAG relationship table (DocSection to Tag)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS HAS_TAG (
                    FROM DocSection TO Tag,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created HAS_TAG relationship table")
        except Exception as e:
            logger.debug(f"HAS_TAG table creation: {e}")

        # Create DEMONSTRATES relationship table (DocSection to Tag for languages)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS DEMONSTRATES (
                    FROM DocSection TO Tag,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created DEMONSTRATES relationship table")
        except Exception as e:
            logger.debug(f"DEMONSTRATES table creation: {e}")

        # Create LINKS_TO relationship table (DocSection to DocSection)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS LINKS_TO (
                    FROM DocSection TO DocSection,
                    weight DOUBLE DEFAULT 1.0,
                    commit_sha STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created LINKS_TO relationship table")
        except Exception as e:
            logger.debug(f"LINKS_TO table creation: {e}")

        # Create CONTAINS_SECTION relationship table (Document → DocSection)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS CONTAINS_SECTION (
                    FROM Document TO DocSection,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created CONTAINS_SECTION relationship table")
        except Exception as e:
            logger.debug(f"CONTAINS_SECTION table creation: {e}")

        # Create RELATED_TO relationship table (Document → Document cross-references)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS RELATED_TO (
                    FROM Document TO Document,
                    link_text STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created RELATED_TO relationship table")
        except Exception as e:
            logger.debug(f"RELATED_TO table creation: {e}")

        # Create DESCRIBES relationship table (Document → CodeEntity)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS DESCRIBES (
                    FROM Document TO CodeEntity,
                    weight DOUBLE DEFAULT 1.0,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created DESCRIBES relationship table")
        except Exception as e:
            logger.debug(f"DESCRIBES table creation: {e}")

        # Create HAS_TOPIC relationship table (Document → Topic)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS HAS_TOPIC (
                    FROM Document TO Topic,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created HAS_TOPIC relationship table")
        except Exception as e:
            logger.debug(f"HAS_TOPIC table creation: {e}")

        # Create Person node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Person (
                    id STRING PRIMARY KEY,
                    name STRING,
                    email_hash STRING,
                    commits_count INT64 DEFAULT 0,
                    first_commit STRING,
                    last_commit STRING
                )
            """
            )
            logger.debug("Created Person node table")
        except Exception as e:
            logger.debug(f"Person table creation: {e}")

        # Create Project node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Project (
                    id STRING PRIMARY KEY,
                    name STRING,
                    description STRING,
                    repo_url STRING
                )
            """
            )
            logger.debug("Created Project node table")
        except Exception as e:
            logger.debug(f"Project table creation: {e}")

        # Create AUTHORED relationship (Person -> CodeEntity)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS AUTHORED (
                    FROM Person TO CodeEntity,
                    timestamp STRING,
                    commit_sha STRING,
                    lines_authored INT64 DEFAULT 0,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created AUTHORED relationship table")
        except Exception as e:
            logger.debug(f"AUTHORED table creation: {e}")

        # Create MODIFIED relationship (Person -> CodeEntity)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS MODIFIED (
                    FROM Person TO CodeEntity,
                    timestamp STRING,
                    commit_sha STRING,
                    lines_changed INT64 DEFAULT 0,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created MODIFIED relationship table")
        except Exception as e:
            logger.debug(f"MODIFIED table creation: {e}")

        # Create PART_OF relationship (CodeEntity -> Project)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS PART_OF (
                    FROM CodeEntity TO Project,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created PART_OF relationship table")
        except Exception as e:
            logger.debug(f"PART_OF table creation: {e}")

        # Create Repository node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Repository (
                    id STRING PRIMARY KEY,
                    name STRING,
                    url STRING,
                    default_branch STRING,
                    created_at STRING
                )
            """
            )
            logger.debug("Created Repository node table")
        except Exception as e:
            logger.debug(f"Repository table creation: {e}")

        # Create Branch node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Branch (
                    id STRING PRIMARY KEY,
                    name STRING,
                    repository_id STRING,
                    is_default BOOLEAN,
                    created_at STRING
                )
            """
            )
            logger.debug("Created Branch node table")
        except Exception as e:
            logger.debug(f"Branch table creation: {e}")

        # Create Commit node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Commit (
                    id STRING PRIMARY KEY,
                    sha STRING,
                    message STRING,
                    author_name STRING,
                    author_email STRING,
                    timestamp STRING
                )
            """
            )
            logger.debug("Created Commit node table")
        except Exception as e:
            logger.debug(f"Commit table creation: {e}")

        # Create ProgrammingLanguage node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS ProgrammingLanguage (
                    id STRING PRIMARY KEY,
                    name STRING,
                    version STRING,
                    file_extensions STRING
                )
            """
            )
            logger.debug("Created ProgrammingLanguage node table")
        except Exception as e:
            logger.debug(f"ProgrammingLanguage table creation: {e}")

        # Create ProgrammingFramework node table
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS ProgrammingFramework (
                    id STRING PRIMARY KEY,
                    name STRING,
                    version STRING,
                    language_id STRING,
                    category STRING
                )
            """
            )
            logger.debug("Created ProgrammingFramework node table")
        except Exception as e:
            logger.debug(f"ProgrammingFramework table creation: {e}")

        # Create MODIFIES relationship (Commit -> CodeEntity)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS MODIFIES (
                    FROM Commit TO CodeEntity,
                    lines_added INT64 DEFAULT 0,
                    lines_deleted INT64 DEFAULT 0,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created MODIFIES relationship table")
        except Exception as e:
            logger.debug(f"MODIFIES table creation: {e}")

        # Create BRANCHED_FROM relationship (Branch -> Branch)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS BRANCHED_FROM (
                    FROM Branch TO Branch,
                    timestamp STRING,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created BRANCHED_FROM relationship table")
        except Exception as e:
            logger.debug(f"BRANCHED_FROM table creation: {e}")

        # Create COMMITTED_TO relationship (Commit -> Branch)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS COMMITTED_TO (
                    FROM Commit TO Branch,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created COMMITTED_TO relationship table")
        except Exception as e:
            logger.debug(f"COMMITTED_TO table creation: {e}")

        # Create BELONGS_TO relationship (Branch -> Repository)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS BELONGS_TO (
                    FROM Branch TO Repository,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created BELONGS_TO relationship table")
        except Exception as e:
            logger.debug(f"BELONGS_TO table creation: {e}")

        # Create WRITTEN_IN relationship (CodeEntity -> ProgrammingLanguage)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS WRITTEN_IN (
                    FROM CodeEntity TO ProgrammingLanguage,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created WRITTEN_IN relationship table")
        except Exception as e:
            logger.debug(f"WRITTEN_IN table creation: {e}")

        # Create USES_FRAMEWORK relationship (Project -> ProgrammingFramework)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS USES_FRAMEWORK (
                    FROM Project TO ProgrammingFramework,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created USES_FRAMEWORK relationship table")
        except Exception as e:
            logger.debug(f"USES_FRAMEWORK table creation: {e}")

        # Create FRAMEWORK_FOR relationship (ProgrammingFramework -> ProgrammingLanguage)
        try:
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS FRAMEWORK_FOR (
                    FROM ProgrammingFramework TO ProgrammingLanguage,
                    MANY_MANY
                )
            """
            )
            logger.debug("Created FRAMEWORK_FOR relationship table")
        except Exception as e:
            logger.debug(f"FRAMEWORK_FOR table creation: {e}")

    async def add_entity(self, entity: CodeEntity):
        """Add or update a code entity.

        Args:
            entity: CodeEntity to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (e:CodeEntity {id: $id})
                ON MATCH SET e.name = $name,
                             e.entity_type = $entity_type,
                             e.file_path = $file_path,
                             e.commit_sha = $commit_sha
                ON CREATE SET e.name = $name,
                              e.entity_type = $entity_type,
                              e.file_path = $file_path,
                              e.commit_sha = $commit_sha
            """,
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "file_path": entity.file_path,
                    "commit_sha": entity.commit_sha or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add entity {entity.id}: {e}")
            raise

    async def add_doc_section(self, doc: DocSection):
        """Add or update a documentation section.

        Args:
            doc: DocSection to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (d:DocSection {id: $id})
                ON MATCH SET d.name = $name,
                             d.doc_type = $doc_type,
                             d.file_path = $file_path,
                             d.level = $level,
                             d.line_start = $line_start,
                             d.line_end = $line_end,
                             d.commit_sha = $commit_sha
                ON CREATE SET d.name = $name,
                              d.doc_type = $doc_type,
                              d.file_path = $file_path,
                              d.level = $level,
                              d.line_start = $line_start,
                              d.line_end = $line_end,
                              d.commit_sha = $commit_sha
            """,
                {
                    "id": doc.id,
                    "name": doc.name,
                    "doc_type": doc.doc_type,
                    "file_path": doc.file_path,
                    "level": doc.level,
                    "line_start": doc.line_start,
                    "line_end": doc.line_end,
                    "commit_sha": doc.commit_sha or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add doc section {doc.id}: {e}")
            raise

    async def add_tag(self, tag_name: str):
        """Add or update a tag.

        Args:
            tag_name: Tag name to add/update
        """
        if not self._initialized:
            await self.initialize()

        tag_id = f"tag:{tag_name}"

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (t:Tag {id: $id})
                ON MATCH SET t.name = $name
                ON CREATE SET t.name = $name
            """,
                {
                    "id": tag_id,
                    "name": tag_name,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add tag {tag_name}: {e}")
            raise

    async def add_person(self, person: Person):
        """Add or update a person.

        Args:
            person: Person to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (p:Person {id: $id})
                ON MATCH SET p.name = $name,
                             p.email_hash = $email_hash,
                             p.commits_count = $commits_count,
                             p.first_commit = $first_commit,
                             p.last_commit = $last_commit
                ON CREATE SET p.name = $name,
                              p.email_hash = $email_hash,
                              p.commits_count = $commits_count,
                              p.first_commit = $first_commit,
                              p.last_commit = $last_commit
            """,
                {
                    "id": person.id,
                    "name": person.name,
                    "email_hash": person.email_hash,
                    "commits_count": person.commits_count,
                    "first_commit": person.first_commit or "",
                    "last_commit": person.last_commit or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add person {person.id}: {e}")
            raise

    async def add_project(self, project: Project):
        """Add or update a project.

        Args:
            project: Project to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (p:Project {id: $id})
                ON MATCH SET p.name = $name,
                             p.description = $description,
                             p.repo_url = $repo_url
                ON CREATE SET p.name = $name,
                              p.description = $description,
                              p.repo_url = $repo_url
            """,
                {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "repo_url": project.repo_url,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add project {project.id}: {e}")
            raise

    async def add_authored_relationship(
        self,
        person_id: str,
        entity_id: str,
        timestamp: str,
        commit_sha: str,
        lines: int,
    ):
        """Add an AUTHORED relationship between a person and code entity.

        Args:
            person_id: Person ID
            entity_id: Code entity ID
            timestamp: ISO timestamp
            commit_sha: Git commit SHA
            lines: Number of lines authored
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            person_exists = self.conn.execute(
                "MATCH (p:Person {id: $id}) RETURN p", {"id": person_id}
            )
            entity_exists = self.conn.execute(
                "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": entity_id}
            )

            if not person_exists.has_next() or not entity_exists.has_next():
                logger.warning(
                    f"Skipping AUTHORED: person or entity does not exist "
                    f"(person={person_id}, entity={entity_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
                MERGE (p)-[r:AUTHORED]->(e)
                ON MATCH SET r.timestamp = $timestamp,
                             r.commit_sha = $commit_sha,
                             r.lines_authored = $lines
                ON CREATE SET r.timestamp = $timestamp,
                              r.commit_sha = $commit_sha,
                              r.lines_authored = $lines
            """,
                {
                    "person_id": person_id,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "commit_sha": commit_sha,
                    "lines": lines,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add AUTHORED relationship: {e}")
            raise

    async def add_modified_relationship(
        self,
        person_id: str,
        entity_id: str,
        timestamp: str,
        commit_sha: str,
        lines: int,
    ):
        """Add a MODIFIED relationship between a person and code entity.

        Args:
            person_id: Person ID
            entity_id: Code entity ID
            timestamp: ISO timestamp
            commit_sha: Git commit SHA
            lines: Number of lines changed
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            person_exists = self.conn.execute(
                "MATCH (p:Person {id: $id}) RETURN p", {"id": person_id}
            )
            entity_exists = self.conn.execute(
                "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": entity_id}
            )

            if not person_exists.has_next() or not entity_exists.has_next():
                logger.warning(
                    f"Skipping MODIFIED: person or entity does not exist "
                    f"(person={person_id}, entity={entity_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
                MERGE (p)-[r:MODIFIED]->(e)
                ON MATCH SET r.timestamp = $timestamp,
                             r.commit_sha = $commit_sha,
                             r.lines_changed = $lines
                ON CREATE SET r.timestamp = $timestamp,
                              r.commit_sha = $commit_sha,
                              r.lines_changed = $lines
            """,
                {
                    "person_id": person_id,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "commit_sha": commit_sha,
                    "lines": lines,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add MODIFIED relationship: {e}")
            raise

    async def add_part_of_relationship(self, entity_id: str, project_id: str):
        """Add a PART_OF relationship between a code entity and project.

        Args:
            entity_id: Code entity ID
            project_id: Project ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            entity_exists = self.conn.execute(
                "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": entity_id}
            )
            project_exists = self.conn.execute(
                "MATCH (p:Project {id: $id}) RETURN p", {"id": project_id}
            )

            if not entity_exists.has_next() or not project_exists.has_next():
                logger.warning(
                    f"Skipping PART_OF: entity or project does not exist "
                    f"(entity={entity_id}, project={project_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (e:CodeEntity {id: $entity_id}), (p:Project {id: $project_id})
                MERGE (e)-[r:PART_OF]->(p)
            """,
                {
                    "entity_id": entity_id,
                    "project_id": project_id,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add PART_OF relationship: {e}")
            raise

    async def add_repository(self, repository: "Repository"):
        """Add or update a repository.

        Args:
            repository: Repository to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (r:Repository {id: $id})
                ON MATCH SET r.name = $name,
                             r.url = $url,
                             r.default_branch = $default_branch,
                             r.created_at = $created_at
                ON CREATE SET r.name = $name,
                              r.url = $url,
                              r.default_branch = $default_branch,
                              r.created_at = $created_at
            """,
                {
                    "id": repository.id,
                    "name": repository.name,
                    "url": repository.url,
                    "default_branch": repository.default_branch,
                    "created_at": repository.created_at or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add repository {repository.id}: {e}")
            raise

    async def add_branch(self, branch: "Branch"):
        """Add or update a branch.

        Args:
            branch: Branch to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (b:Branch {id: $id})
                ON MATCH SET b.name = $name,
                             b.repository_id = $repository_id,
                             b.is_default = $is_default,
                             b.created_at = $created_at
                ON CREATE SET b.name = $name,
                              b.repository_id = $repository_id,
                              b.is_default = $is_default,
                              b.created_at = $created_at
            """,
                {
                    "id": branch.id,
                    "name": branch.name,
                    "repository_id": branch.repository_id,
                    "is_default": branch.is_default,
                    "created_at": branch.created_at or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add branch {branch.id}: {e}")
            raise

    async def add_commit(self, commit: "Commit"):
        """Add or update a commit.

        Args:
            commit: Commit to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (c:Commit {id: $id})
                ON MATCH SET c.sha = $sha,
                             c.message = $message,
                             c.author_name = $author_name,
                             c.author_email = $author_email,
                             c.timestamp = $timestamp
                ON CREATE SET c.sha = $sha,
                              c.message = $message,
                              c.author_name = $author_name,
                              c.author_email = $author_email,
                              c.timestamp = $timestamp
            """,
                {
                    "id": commit.id,
                    "sha": commit.sha,
                    "message": commit.message,
                    "author_name": commit.author_name,
                    "author_email": commit.author_email,
                    "timestamp": commit.timestamp or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add commit {commit.id}: {e}")
            raise

    async def add_programming_language(self, language: "ProgrammingLanguage"):
        """Add or update a programming language.

        Args:
            language: ProgrammingLanguage to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (l:ProgrammingLanguage {id: $id})
                ON MATCH SET l.name = $name,
                             l.version = $version,
                             l.file_extensions = $file_extensions
                ON CREATE SET l.name = $name,
                              l.version = $version,
                              l.file_extensions = $file_extensions
            """,
                {
                    "id": language.id,
                    "name": language.name,
                    "version": language.version,
                    "file_extensions": language.file_extensions,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add programming language {language.id}: {e}")
            raise

    async def add_programming_framework(self, framework: "ProgrammingFramework"):
        """Add or update a programming framework.

        Args:
            framework: ProgrammingFramework to add/update
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use MERGE to avoid duplicates
            self.conn.execute(
                """
                MERGE (f:ProgrammingFramework {id: $id})
                ON MATCH SET f.name = $name,
                             f.version = $version,
                             f.language_id = $language_id,
                             f.category = $category
                ON CREATE SET f.name = $name,
                              f.version = $version,
                              f.language_id = $language_id,
                              f.category = $category
            """,
                {
                    "id": framework.id,
                    "name": framework.name,
                    "version": framework.version,
                    "language_id": framework.language_id,
                    "category": framework.category,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add programming framework {framework.id}: {e}")
            raise

    async def add_written_in_relationship(self, entity_id: str, language_id: str):
        """Add a WRITTEN_IN relationship between a code entity and programming language.

        Args:
            entity_id: Code entity ID
            language_id: Programming language ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            entity_exists = self.conn.execute(
                "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": entity_id}
            )
            language_exists = self.conn.execute(
                "MATCH (l:ProgrammingLanguage {id: $id}) RETURN l", {"id": language_id}
            )

            if not entity_exists.has_next() or not language_exists.has_next():
                logger.debug(
                    f"Skipping WRITTEN_IN: entity or language does not exist "
                    f"(entity={entity_id}, language={language_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (e:CodeEntity {id: $entity_id})
                MATCH (l:ProgrammingLanguage {id: $language_id})
                MERGE (e)-[:WRITTEN_IN]->(l)
            """,
                {"entity_id": entity_id, "language_id": language_id},
            )
        except Exception as e:
            logger.debug(f"Failed to add WRITTEN_IN relationship: {e}")

    async def add_uses_framework_relationship(self, project_id: str, framework_id: str):
        """Add a USES_FRAMEWORK relationship between a project and framework.

        Args:
            project_id: Project ID
            framework_id: Framework ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            project_exists = self.conn.execute(
                "MATCH (p:Project {id: $id}) RETURN p", {"id": project_id}
            )
            framework_exists = self.conn.execute(
                "MATCH (f:ProgrammingFramework {id: $id}) RETURN f",
                {"id": framework_id},
            )

            if not project_exists.has_next() or not framework_exists.has_next():
                logger.debug(
                    f"Skipping USES_FRAMEWORK: project or framework does not exist "
                    f"(project={project_id}, framework={framework_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (p:Project {id: $project_id})
                MATCH (f:ProgrammingFramework {id: $framework_id})
                MERGE (p)-[:USES_FRAMEWORK]->(f)
            """,
                {"project_id": project_id, "framework_id": framework_id},
            )
        except Exception as e:
            logger.debug(f"Failed to add USES_FRAMEWORK relationship: {e}")

    async def add_framework_for_relationship(self, framework_id: str, language_id: str):
        """Add a FRAMEWORK_FOR relationship between a framework and language.

        Args:
            framework_id: Framework ID
            language_id: Language ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            framework_exists = self.conn.execute(
                "MATCH (f:ProgrammingFramework {id: $id}) RETURN f",
                {"id": framework_id},
            )
            language_exists = self.conn.execute(
                "MATCH (l:ProgrammingLanguage {id: $id}) RETURN l", {"id": language_id}
            )

            if not framework_exists.has_next() or not language_exists.has_next():
                logger.debug(
                    f"Skipping FRAMEWORK_FOR: framework or language does not exist "
                    f"(framework={framework_id}, language={language_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (f:ProgrammingFramework {id: $framework_id})
                MATCH (l:ProgrammingLanguage {id: $language_id})
                MERGE (f)-[:FRAMEWORK_FOR]->(l)
            """,
                {"framework_id": framework_id, "language_id": language_id},
            )
        except Exception as e:
            logger.debug(f"Failed to add FRAMEWORK_FOR relationship: {e}")

    async def add_modifies_relationship(
        self,
        commit_id: str,
        entity_id: str,
        lines_added: int = 0,
        lines_deleted: int = 0,
    ):
        """Add a MODIFIES relationship between a commit and code entity.

        Args:
            commit_id: Commit ID
            entity_id: Code entity ID
            lines_added: Number of lines added
            lines_deleted: Number of lines deleted
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            commit_exists = self.conn.execute(
                "MATCH (c:Commit {id: $id}) RETURN c", {"id": commit_id}
            )
            entity_exists = self.conn.execute(
                "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": entity_id}
            )

            if not commit_exists.has_next() or not entity_exists.has_next():
                logger.warning(
                    f"Skipping MODIFIES: commit or entity does not exist "
                    f"(commit={commit_id}, entity={entity_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (c:Commit {id: $commit_id}), (e:CodeEntity {id: $entity_id})
                MERGE (c)-[r:MODIFIES]->(e)
                ON MATCH SET r.lines_added = $lines_added,
                             r.lines_deleted = $lines_deleted
                ON CREATE SET r.lines_added = $lines_added,
                              r.lines_deleted = $lines_deleted
            """,
                {
                    "commit_id": commit_id,
                    "entity_id": entity_id,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add MODIFIES relationship: {e}")
            raise

    async def add_branched_from_relationship(
        self, source_branch_id: str, target_branch_id: str, timestamp: str
    ):
        """Add a BRANCHED_FROM relationship between branches.

        Args:
            source_branch_id: Source branch ID
            target_branch_id: Target branch ID (parent branch)
            timestamp: ISO timestamp when branch was created
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            source_exists = self.conn.execute(
                "MATCH (b:Branch {id: $id}) RETURN b", {"id": source_branch_id}
            )
            target_exists = self.conn.execute(
                "MATCH (b:Branch {id: $id}) RETURN b", {"id": target_branch_id}
            )

            if not source_exists.has_next() or not target_exists.has_next():
                logger.warning(
                    f"Skipping BRANCHED_FROM: branch does not exist "
                    f"(source={source_branch_id}, target={target_branch_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (s:Branch {id: $source_id}), (t:Branch {id: $target_id})
                MERGE (s)-[r:BRANCHED_FROM]->(t)
                ON MATCH SET r.timestamp = $timestamp
                ON CREATE SET r.timestamp = $timestamp
            """,
                {
                    "source_id": source_branch_id,
                    "target_id": target_branch_id,
                    "timestamp": timestamp,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add BRANCHED_FROM relationship: {e}")
            raise

    async def add_committed_to_relationship(self, commit_id: str, branch_id: str):
        """Add a COMMITTED_TO relationship between commit and branch.

        Args:
            commit_id: Commit ID
            branch_id: Branch ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            commit_exists = self.conn.execute(
                "MATCH (c:Commit {id: $id}) RETURN c", {"id": commit_id}
            )
            branch_exists = self.conn.execute(
                "MATCH (b:Branch {id: $id}) RETURN b", {"id": branch_id}
            )

            if not commit_exists.has_next() or not branch_exists.has_next():
                logger.warning(
                    f"Skipping COMMITTED_TO: commit or branch does not exist "
                    f"(commit={commit_id}, branch={branch_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (c:Commit {id: $commit_id}), (b:Branch {id: $branch_id})
                MERGE (c)-[r:COMMITTED_TO]->(b)
            """,
                {
                    "commit_id": commit_id,
                    "branch_id": branch_id,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add COMMITTED_TO relationship: {e}")
            raise

    async def add_belongs_to_relationship(self, branch_id: str, repository_id: str):
        """Add a BELONGS_TO relationship between branch and repository.

        Args:
            branch_id: Branch ID
            repository_id: Repository ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if both nodes exist
            branch_exists = self.conn.execute(
                "MATCH (b:Branch {id: $id}) RETURN b", {"id": branch_id}
            )
            repo_exists = self.conn.execute(
                "MATCH (r:Repository {id: $id}) RETURN r", {"id": repository_id}
            )

            if not branch_exists.has_next() or not repo_exists.has_next():
                logger.warning(
                    f"Skipping BELONGS_TO: branch or repository does not exist "
                    f"(branch={branch_id}, repository={repository_id})"
                )
                return

            # Create relationship
            self.conn.execute(
                """
                MATCH (b:Branch {id: $branch_id}), (r:Repository {id: $repository_id})
                MERGE (b)-[rel:BELONGS_TO]->(r)
            """,
                {
                    "branch_id": branch_id,
                    "repository_id": repository_id,
                },
            )
        except Exception as e:
            logger.error(f"Failed to add BELONGS_TO relationship: {e}")
            raise

    def add_entities_batch_sync(
        self, entities: list[CodeEntity], batch_size: int = 500
    ) -> int:
        """Batch insert code entities using UNWIND (synchronous version for Kuzu safety).

        This is a synchronous version that avoids asyncio event loop threads
        which can cause segfaults with Kuzu's Rust bindings.

        Args:
            entities: List of CodeEntity objects
            batch_size: Number of entities per batch (default 500)

        Returns:
            Number of entities inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            params = [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "file_path": e.file_path or "",
                    "commit_sha": e.commit_sha or "",
                }
                for e in batch
            ]

            try:
                self._execute_query(
                    """
                    UNWIND $batch AS e
                    MERGE (n:CodeEntity {id: e.id})
                    ON CREATE SET n.name = e.name,
                                  n.entity_type = e.entity_type,
                                  n.file_path = e.file_path,
                                  n.commit_sha = e.commit_sha
                    ON MATCH SET n.name = e.name,
                                 n.entity_type = e.entity_type,
                                 n.file_path = e.file_path,
                                 n.commit_sha = e.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert entity batch: {e}")
                raise

        return total

    def add_doc_sections_batch_sync(
        self, docs: list[DocSection], batch_size: int = 500
    ) -> int:
        """Batch insert doc sections using UNWIND (synchronous).

        Args:
            docs: List of DocSection objects
            batch_size: Number of doc sections per batch (default 500)

        Returns:
            Number of doc sections inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            params = [
                {
                    "id": d.id,
                    "name": d.name,
                    "doc_type": d.doc_type,
                    "file_path": d.file_path,
                    "level": d.level,
                    "line_start": d.line_start,
                    "line_end": d.line_end,
                    "commit_sha": d.commit_sha or "",
                }
                for d in batch
            ]

            try:
                self._execute_query(
                    """
                    UNWIND $batch AS d
                    MERGE (n:DocSection {id: d.id})
                    ON CREATE SET n.name = d.name,
                                  n.doc_type = d.doc_type,
                                  n.file_path = d.file_path,
                                  n.level = d.level,
                                  n.line_start = d.line_start,
                                  n.line_end = d.line_end,
                                  n.commit_sha = d.commit_sha
                    ON MATCH SET n.name = d.name,
                                 n.doc_type = d.doc_type,
                                 n.file_path = d.file_path,
                                 n.level = d.level,
                                 n.line_start = d.line_start,
                                 n.line_end = d.line_end,
                                 n.commit_sha = d.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert doc sections batch: {e}")
                raise

        return total

    def add_documents_batch_sync(
        self, documents: list["Document"], batch_size: int = 500
    ) -> int:
        """Batch insert document nodes using UNWIND (synchronous).

        Args:
            documents: List of Document objects
            batch_size: Number of documents per batch (default 500)

        Returns:
            Number of documents inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            params = [
                {
                    "id": d.id,
                    "file_path": d.file_path,
                    "title": d.title,
                    "doc_category": d.doc_category,
                    "word_count": d.word_count,
                    "section_count": d.section_count,
                    "last_modified": d.last_modified or "",
                    "commit_sha": d.commit_sha or "",
                }
                for d in batch
            ]

            try:
                self._execute_query(
                    """
                    UNWIND $batch AS d
                    MERGE (n:Document {id: d.id})
                    ON CREATE SET n.file_path = d.file_path,
                                  n.title = d.title,
                                  n.doc_category = d.doc_category,
                                  n.word_count = d.word_count,
                                  n.section_count = d.section_count,
                                  n.last_modified = d.last_modified,
                                  n.commit_sha = d.commit_sha
                    ON MATCH SET n.file_path = d.file_path,
                                 n.title = d.title,
                                 n.doc_category = d.doc_category,
                                 n.word_count = d.word_count,
                                 n.section_count = d.section_count,
                                 n.last_modified = d.last_modified,
                                 n.commit_sha = d.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert documents batch: {e}")
                raise

        return total

    async def add_documents_batch(
        self, documents: list["Document"], batch_size: int = 500
    ) -> int:
        """Batch insert document nodes using UNWIND (async).

        Args:
            documents: List of Document objects
            batch_size: Number of documents per batch (default 500)

        Returns:
            Number of documents inserted
        """
        if not self._initialized:
            await self.initialize()

        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            params = [
                {
                    "id": d.id,
                    "file_path": d.file_path,
                    "title": d.title,
                    "doc_category": d.doc_category,
                    "word_count": d.word_count,
                    "section_count": d.section_count,
                    "last_modified": d.last_modified or "",
                    "commit_sha": d.commit_sha or "",
                }
                for d in batch
            ]

            try:
                self.conn.execute(
                    """
                    UNWIND $batch AS d
                    MERGE (n:Document {id: d.id})
                    ON CREATE SET n.file_path = d.file_path,
                                  n.title = d.title,
                                  n.doc_category = d.doc_category,
                                  n.word_count = d.word_count,
                                  n.section_count = d.section_count,
                                  n.last_modified = d.last_modified,
                                  n.commit_sha = d.commit_sha
                    ON MATCH SET n.file_path = d.file_path,
                                 n.title = d.title,
                                 n.doc_category = d.doc_category,
                                 n.word_count = d.word_count,
                                 n.section_count = d.section_count,
                                 n.last_modified = d.last_modified,
                                 n.commit_sha = d.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert documents batch: {e}")
                # Fallback to individual inserts
                for doc in batch:
                    try:
                        self._execute_query(
                            """
                            MERGE (n:Document {id: $id})
                            ON CREATE SET n.file_path = $file_path,
                                          n.title = $title,
                                          n.doc_category = $doc_category,
                                          n.word_count = $word_count,
                                          n.section_count = $section_count,
                                          n.last_modified = $last_modified,
                                          n.commit_sha = $commit_sha
                            ON MATCH SET n.file_path = $file_path,
                                         n.title = $title,
                                         n.doc_category = $doc_category,
                                         n.word_count = $word_count,
                                         n.section_count = $section_count,
                                         n.last_modified = $last_modified,
                                         n.commit_sha = $commit_sha
                            """,
                            {
                                "id": doc.id,
                                "file_path": doc.file_path,
                                "title": doc.title,
                                "doc_category": doc.doc_category,
                                "word_count": doc.word_count,
                                "section_count": doc.section_count,
                                "last_modified": doc.last_modified or "",
                                "commit_sha": doc.commit_sha or "",
                            },
                        )
                        total += 1
                    except Exception:
                        pass

        return total

    def add_tags_batch_sync(self, tag_names: list[str], batch_size: int = 500) -> int:
        """Batch insert tags using UNWIND (synchronous).

        Args:
            tag_names: List of tag names
            batch_size: Number of tags per batch (default 500)

        Returns:
            Number of tags inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(tag_names), batch_size):
            batch = tag_names[i : i + batch_size]
            params = [{"id": f"tag:{name}", "name": name} for name in batch]

            try:
                self._execute_query(
                    """
                    UNWIND $batch AS t
                    MERGE (n:Tag {id: t.id})
                    ON CREATE SET n.name = t.name
                    ON MATCH SET n.name = t.name
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert tags batch: {e}")
                raise

        return total

    def add_relationships_batch_sync(
        self, relationships: list[CodeRelationship], batch_size: int = 500
    ) -> int:
        """Batch insert relationships using UNWIND (synchronous).

        Groups relationships by type and inserts each type in batches.

        Args:
            relationships: List of CodeRelationship objects
            batch_size: Number of relationships per batch (default 500)

        Returns:
            Number of relationships inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        # Group by relationship type
        by_type: dict[str, list[CodeRelationship]] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.upper()
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            total += self._add_relationships_batch_by_type_sync(
                rels, rel_type, batch_size
            )

        return total

    def _add_relationships_batch_by_type_sync(
        self, relationships: list[CodeRelationship], rel_type: str, batch_size: int
    ) -> int:
        """Batch insert relationships of a specific type (synchronous).

        Args:
            relationships: List of CodeRelationship objects
            rel_type: Relationship type (CALLS, IMPORTS, etc.)
            batch_size: Number of relationships per batch

        Returns:
            Number of relationships inserted
        """
        total = 0

        # Determine node types based on relationship type
        if rel_type in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]:
            source_label = "CodeEntity"
            target_label = "CodeEntity"
        elif rel_type == "FOLLOWS":
            source_label = "DocSection"
            target_label = "DocSection"
        elif rel_type in ["REFERENCES", "DOCUMENTS"]:
            source_label = "DocSection"
            target_label = "CodeEntity"
        elif rel_type in ["HAS_TAG", "DEMONSTRATES"]:
            source_label = "DocSection"
            target_label = "Tag"
        elif rel_type == "LINKS_TO":
            source_label = "DocSection"
            target_label = "DocSection"
        elif rel_type == "CONTAINS_SECTION":
            source_label = "Document"
            target_label = "DocSection"
        elif rel_type == "RELATED_TO":
            source_label = "Document"
            target_label = "Document"
        elif rel_type == "DESCRIBES":
            source_label = "Document"
            target_label = "CodeEntity"
        elif rel_type == "HAS_TOPIC":
            source_label = "Document"
            target_label = "Topic"
        else:
            logger.warning(f"Unknown relationship type: {rel_type}")
            return 0

        # Insert in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            params = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "weight": r.weight,
                    "commit_sha": r.commit_sha or "",
                }
                for r in batch
            ]

            try:
                # CRITICAL: Use CREATE instead of MERGE to avoid Kuzu CSR bug
                # Kuzu 0.11.3 has assertion failures in csr_node_group.cpp with MERGE
                # We accept duplicate relationships rather than crash
                query = f"""
                    UNWIND $batch AS r
                    MATCH (a:{source_label} {{id: r.source_id}}),
                          (b:{target_label} {{id: r.target_id}})
                    CREATE (a)-[rel:{rel_type}]->(b)
                    SET rel.weight = r.weight,
                        rel.commit_sha = r.commit_sha
                """
                self._execute_query(query, {"batch": params})
                total += len(batch)
            except Exception as e:
                error_msg = str(e)
                logger.debug(f"Batch insert failed for {rel_type}: {error_msg}")

                # CRITICAL: If Kuzu assertion failure detected, STOP immediately
                # Continuing after assertion failure causes database corruption and crash
                if "KU_UNREACHABLE" in error_msg or "Assertion failed" in error_msg:
                    logger.error(
                        f"FATAL: Kuzu assertion failure in {rel_type} insertion. "
                        "This is a Kuzu bug. Stopping to prevent corruption."
                    )
                    raise RuntimeError(
                        f"Kuzu assertion failure during {rel_type} insertion: {error_msg}"
                    )

                # For other exceptions, skip this batch
                pass

        return total

    def add_persons_batch_sync(
        self, persons: list["Person"], batch_size: int = 500
    ) -> int:
        """Batch insert persons using UNWIND (synchronous).

        Args:
            persons: List of Person objects
            batch_size: Number of persons per batch (default 500)

        Returns:
            Number of persons inserted
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(persons), batch_size):
            batch = persons[i : i + batch_size]
            params = [
                {
                    "id": p.id,
                    "name": p.name,
                    "email_hash": p.email_hash,
                    "commits_count": p.commits_count,
                    "first_commit": p.first_commit,
                    "last_commit": p.last_commit,
                }
                for p in batch
            ]

            try:
                self._execute_query(
                    """
                    UNWIND $batch AS p
                    MERGE (n:Person {id: p.id})
                    ON CREATE SET
                        n.name = p.name,
                        n.email_hash = p.email_hash,
                        n.commits_count = p.commits_count,
                        n.first_commit = p.first_commit,
                        n.last_commit = p.last_commit
                    ON MATCH SET
                        n.name = p.name,
                        n.commits_count = p.commits_count,
                        n.last_commit = p.last_commit
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert persons batch: {e}")
                raise

        return total

    def add_project_sync(self, project: "Project") -> None:
        """Add or update a project (synchronous).

        Args:
            project: Project object to add/update
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        try:
            self._execute_query(
                """
                MERGE (p:Project {id: $id})
                ON CREATE SET
                    p.name = $name,
                    p.description = $description,
                    p.repo_url = $repo_url
                ON MATCH SET
                    p.name = $name,
                    p.description = $description,
                    p.repo_url = $repo_url
                """,
                {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description or "",
                    "repo_url": project.repo_url or "",
                },
            )
        except Exception as e:
            logger.error(f"Failed to add project {project.id}: {e}")
            raise

    def add_authored_relationship_sync(
        self,
        person_id: str,
        entity_id: str,
        timestamp: str,
        commit_sha: str,
        lines: int,
    ) -> None:
        """Add an AUTHORED relationship (synchronous).

        Args:
            person_id: ID of the Person node
            entity_id: ID of the CodeEntity node
            timestamp: ISO timestamp of authorship
            commit_sha: Git commit SHA
            lines: Number of lines authored
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        try:
            self._execute_query(
                """
                MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
                CREATE (p)-[r:AUTHORED]->(e)
                SET r.timestamp = $timestamp,
                    r.commit_sha = $commit_sha,
                    r.lines_authored = $lines
                """,
                {
                    "person_id": person_id,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "commit_sha": commit_sha,
                    "lines": lines,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to add AUTHORED relationship: {e}")

    def add_modified_relationship_sync(
        self,
        person_id: str,
        entity_id: str,
        timestamp: str,
        commit_sha: str,
        lines: int,
    ) -> None:
        """Add a MODIFIED relationship (synchronous).

        Args:
            person_id: ID of the Person node
            entity_id: ID of the CodeEntity node
            timestamp: ISO timestamp of modification
            commit_sha: Git commit SHA
            lines: Number of lines changed
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        try:
            self._execute_query(
                """
                MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
                CREATE (p)-[r:MODIFIED]->(e)
                SET r.timestamp = $timestamp,
                    r.commit_sha = $commit_sha,
                    r.lines_changed = $lines
                """,
                {
                    "person_id": person_id,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "commit_sha": commit_sha,
                    "lines": lines,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to add MODIFIED relationship: {e}")

    def add_part_of_batch_sync(
        self, entity_ids: list[str], project_id: str, batch_size: int = 500
    ) -> int:
        """Batch create PART_OF relationships (synchronous).

        Args:
            entity_ids: List of CodeEntity IDs
            project_id: Project ID to link to
            batch_size: Number of relationships per batch (default 500)

        Returns:
            Number of relationships created
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        total = 0
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i : i + batch_size]

            try:
                self._execute_query(
                    """
                    UNWIND $entity_ids AS entity_id
                    MATCH (e:CodeEntity {id: entity_id}), (p:Project {id: $project_id})
                    CREATE (e)-[:PART_OF]->(p)
                    """,
                    {"entity_ids": batch, "project_id": project_id},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Failed to create PART_OF relationships batch: {e}")
                raise

        return total

    async def add_entities_batch(
        self, entities: list[CodeEntity], batch_size: int = 500
    ) -> int:
        """Batch insert code entities using UNWIND.

        This reduces N individual MERGE queries to N/batch_size queries,
        dramatically improving performance for large-scale indexing.

        Args:
            entities: List of CodeEntity objects
            batch_size: Number of entities per batch (default 500)

        Returns:
            Number of entities inserted
        """
        if not self._initialized:
            await self.initialize()

        total = 0
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            params = [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "file_path": e.file_path or "",
                    "commit_sha": e.commit_sha or "",
                }
                for e in batch
            ]

            try:
                self.conn.execute(
                    """
                    UNWIND $batch AS e
                    MERGE (n:CodeEntity {id: e.id})
                    ON CREATE SET n.name = e.name,
                                  n.entity_type = e.entity_type,
                                  n.file_path = e.file_path,
                                  n.commit_sha = e.commit_sha
                    ON MATCH SET n.name = e.name,
                                 n.entity_type = e.entity_type,
                                 n.file_path = e.file_path,
                                 n.commit_sha = e.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Batch insert failed for entities: {e}")
                # Fallback to individual inserts for this batch
                for entity in batch:
                    try:
                        await self.add_entity(entity)
                        total += 1
                    except Exception:
                        pass

        return total

    async def add_doc_sections_batch(
        self, docs: list[DocSection], batch_size: int = 500
    ) -> int:
        """Batch insert doc sections using UNWIND.

        Args:
            docs: List of DocSection objects
            batch_size: Number of doc sections per batch (default 500)

        Returns:
            Number of doc sections inserted
        """
        if not self._initialized:
            await self.initialize()

        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            params = [
                {
                    "id": d.id,
                    "name": d.name,
                    "doc_type": d.doc_type,
                    "file_path": d.file_path,
                    "level": d.level,
                    "line_start": d.line_start,
                    "line_end": d.line_end,
                    "commit_sha": d.commit_sha or "",
                }
                for d in batch
            ]

            try:
                self.conn.execute(
                    """
                    UNWIND $batch AS d
                    MERGE (n:DocSection {id: d.id})
                    ON CREATE SET n.name = d.name,
                                  n.doc_type = d.doc_type,
                                  n.file_path = d.file_path,
                                  n.level = d.level,
                                  n.line_start = d.line_start,
                                  n.line_end = d.line_end,
                                  n.commit_sha = d.commit_sha
                    ON MATCH SET n.name = d.name,
                                 n.doc_type = d.doc_type,
                                 n.file_path = d.file_path,
                                 n.level = d.level,
                                 n.line_start = d.line_start,
                                 n.line_end = d.line_end,
                                 n.commit_sha = d.commit_sha
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Batch insert failed for doc sections: {e}")
                # Fallback to individual inserts
                for doc in batch:
                    try:
                        await self.add_doc_section(doc)
                        total += 1
                    except Exception:
                        pass

        return total

    async def add_relationships_batch(
        self, relationships: list[CodeRelationship], batch_size: int = 500
    ) -> int:
        """Batch insert relationships using UNWIND.

        Groups relationships by type and inserts each type in batches.

        Args:
            relationships: List of CodeRelationship objects
            batch_size: Number of relationships per batch (default 500)

        Returns:
            Number of relationships inserted
        """
        if not self._initialized:
            await self.initialize()

        # Group by relationship type
        by_type: dict[str, list[CodeRelationship]] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.upper()
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            total += await self._add_relationships_batch_by_type(
                rels, rel_type, batch_size
            )

        return total

    async def _add_relationships_batch_by_type(
        self, relationships: list[CodeRelationship], rel_type: str, batch_size: int
    ) -> int:
        """Batch insert relationships of a specific type.

        Args:
            relationships: List of CodeRelationship objects
            rel_type: Relationship type (CALLS, IMPORTS, etc.)
            batch_size: Number of relationships per batch

        Returns:
            Number of relationships inserted
        """
        total = 0

        # Determine node types based on relationship type
        if rel_type in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]:
            # Code-to-code
            source_label = "CodeEntity"
            target_label = "CodeEntity"
        elif rel_type == "FOLLOWS":
            # Doc-to-doc
            source_label = "DocSection"
            target_label = "DocSection"
        elif rel_type in ["REFERENCES", "DOCUMENTS"]:
            # Doc-to-code
            source_label = "DocSection"
            target_label = "CodeEntity"
        elif rel_type in ["HAS_TAG", "DEMONSTRATES"]:
            # Doc-to-tag
            source_label = "DocSection"
            target_label = "Tag"
        elif rel_type == "LINKS_TO":
            # Doc-to-doc (explicit links)
            source_label = "DocSection"
            target_label = "DocSection"
        elif rel_type == "CONTAINS_SECTION":
            # Document-to-DocSection containment
            source_label = "Document"
            target_label = "DocSection"
        elif rel_type == "RELATED_TO":
            # Document-to-Document cross-references
            source_label = "Document"
            target_label = "Document"
        elif rel_type == "DESCRIBES":
            # Document-to-CodeEntity
            source_label = "Document"
            target_label = "CodeEntity"
        elif rel_type == "HAS_TOPIC":
            # Document-to-Topic
            source_label = "Document"
            target_label = "Topic"
        else:
            logger.warning(f"Unknown relationship type: {rel_type}")
            return 0

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            params = [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "weight": r.weight or 1.0,
                    "commit_sha": r.commit_sha or "",
                }
                for r in batch
            ]

            query = f"""
                UNWIND $batch AS r
                MATCH (s:{source_label} {{id: r.source}})
                MATCH (t:{target_label} {{id: r.target}})
                MERGE (s)-[rel:{rel_type}]->(t)
                ON CREATE SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
                ON MATCH SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
            """

            try:
                # Thread-safe Kuzu operation via wrapper
                self._execute_query(query, {"batch": params})
                total += len(batch)
            except Exception as e:
                logger.error(
                    f"Batch relationship insert failed for {rel_type} (batch size {len(batch)}): {e}"
                )
                # Fallback to individual inserts
                for rel in batch:
                    try:
                        await self.add_relationship(rel)
                        total += 1
                    except Exception:
                        pass

        return total

    async def add_tags_batch(self, tag_names: list[str], batch_size: int = 500) -> int:
        """Batch insert tags using UNWIND.

        Args:
            tag_names: List of tag names
            batch_size: Number of tags per batch (default 500)

        Returns:
            Number of tags inserted
        """
        if not self._initialized:
            await self.initialize()

        # Remove duplicates
        unique_tags = list(set(tag_names))

        total = 0
        for i in range(0, len(unique_tags), batch_size):
            batch = unique_tags[i : i + batch_size]
            params = [{"id": f"tag:{name}", "name": name} for name in batch]

            try:
                self.conn.execute(
                    """
                    UNWIND $batch AS t
                    MERGE (n:Tag {id: t.id})
                    ON CREATE SET n.name = t.name
                    ON MATCH SET n.name = t.name
                """,
                    {"batch": params},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Batch insert failed for tags: {e}")
                # Fallback to individual inserts
                for tag_name in batch:
                    try:
                        await self.add_tag(tag_name)
                        total += 1
                    except Exception:
                        pass

        return total

    async def add_part_of_batch(
        self, entity_ids: list[str], project_id: str, batch_size: int = 500
    ) -> int:
        """Batch create PART_OF relationships.

        Args:
            entity_ids: List of entity IDs
            project_id: Project ID
            batch_size: Number of relationships per batch (default 500)

        Returns:
            Number of relationships created
        """
        if not self._initialized:
            await self.initialize()

        total = 0
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i : i + batch_size]

            try:
                self.conn.execute(
                    """
                    UNWIND $batch AS eid
                    MATCH (e:CodeEntity {id: eid})
                    MATCH (p:Project {id: $project_id})
                    MERGE (e)-[:PART_OF]->(p)
                """,
                    {"batch": batch, "project_id": project_id},
                )
                total += len(batch)
            except Exception as e:
                logger.error(f"Batch PART_OF failed: {e}")
                # Fallback to individual inserts
                for entity_id in batch:
                    try:
                        await self.add_part_of_relationship(entity_id, project_id)
                        total += 1
                    except Exception:
                        pass

        return total

    async def add_relationship(self, rel: CodeRelationship):
        """Add a relationship between entities.

        Args:
            rel: CodeRelationship to add
        """
        if not self._initialized:
            await self.initialize()

        rel_table = rel.relationship_type.upper()

        try:
            # Determine node types based on relationship type
            if rel_table in ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]:
                # Code-to-code relationship
                source_exists = self.conn.execute(
                    "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": rel.source_id}
                )
                target_exists = self.conn.execute(
                    "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": rel.target_id}
                )

                if not source_exists.has_next() or not target_exists.has_next():
                    logger.warning(
                        f"Skipping relationship {rel_table}: "
                        f"source or target does not exist"
                    )
                    return

                # Create relationship
                self.conn.execute(
                    f"""
                    MATCH (a:CodeEntity {{id: $source}}), (b:CodeEntity {{id: $target}})
                    MERGE (a)-[r:{rel_table}]->(b)
                    ON MATCH SET r.weight = $weight, r.commit_sha = $commit_sha
                    ON CREATE SET r.weight = $weight, r.commit_sha = $commit_sha
                """,
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "weight": rel.weight,
                        "commit_sha": rel.commit_sha or "",
                    },
                )

            elif rel_table == "FOLLOWS":
                # Doc-to-doc relationship
                source_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.source_id}
                )
                target_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.target_id}
                )

                if not source_exists.has_next() or not target_exists.has_next():
                    logger.warning(
                        f"Skipping relationship {rel_table}: "
                        f"source or target does not exist"
                    )
                    return

                # Create relationship
                self.conn.execute(
                    f"""
                    MATCH (a:DocSection {{id: $source}}), (b:DocSection {{id: $target}})
                    MERGE (a)-[r:{rel_table}]->(b)
                    ON MATCH SET r.weight = $weight, r.commit_sha = $commit_sha
                    ON CREATE SET r.weight = $weight, r.commit_sha = $commit_sha
                """,
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "weight": rel.weight,
                        "commit_sha": rel.commit_sha or "",
                    },
                )

            elif rel_table in ["REFERENCES", "DOCUMENTS"]:
                # Doc-to-code relationship
                source_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.source_id}
                )
                target_exists = self.conn.execute(
                    "MATCH (e:CodeEntity {id: $id}) RETURN e", {"id": rel.target_id}
                )

                if not source_exists.has_next() or not target_exists.has_next():
                    logger.warning(
                        f"Skipping relationship {rel_table}: "
                        f"source or target does not exist"
                    )
                    return

                # Create relationship
                self.conn.execute(
                    f"""
                    MATCH (d:DocSection {{id: $source}}), (e:CodeEntity {{id: $target}})
                    MERGE (d)-[r:{rel_table}]->(e)
                    ON MATCH SET r.weight = $weight, r.commit_sha = $commit_sha
                    ON CREATE SET r.weight = $weight, r.commit_sha = $commit_sha
                """,
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "weight": rel.weight,
                        "commit_sha": rel.commit_sha or "",
                    },
                )

            elif rel_table in ["HAS_TAG", "DEMONSTRATES"]:
                # Doc-to-tag relationship
                source_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.source_id}
                )
                target_exists = self.conn.execute(
                    "MATCH (t:Tag {id: $id}) RETURN t", {"id": rel.target_id}
                )

                if not source_exists.has_next() or not target_exists.has_next():
                    logger.warning(
                        f"Skipping relationship {rel_table}: "
                        f"source or target does not exist"
                    )
                    return

                # Create relationship
                self.conn.execute(
                    f"""
                    MATCH (d:DocSection {{id: $source}}), (t:Tag {{id: $target}})
                    MERGE (d)-[r:{rel_table}]->(t)
                    ON MATCH SET r.weight = $weight, r.commit_sha = $commit_sha
                    ON CREATE SET r.weight = $weight, r.commit_sha = $commit_sha
                """,
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "weight": rel.weight,
                        "commit_sha": rel.commit_sha or "",
                    },
                )

            elif rel_table == "LINKS_TO":
                # Doc-to-doc relationship (explicit links)
                source_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.source_id}
                )
                target_exists = self.conn.execute(
                    "MATCH (d:DocSection {id: $id}) RETURN d", {"id": rel.target_id}
                )

                if not source_exists.has_next() or not target_exists.has_next():
                    logger.warning(
                        f"Skipping relationship {rel_table}: "
                        f"source or target does not exist"
                    )
                    return

                # Create relationship
                self.conn.execute(
                    f"""
                    MATCH (a:DocSection {{id: $source}}), (b:DocSection {{id: $target}})
                    MERGE (a)-[r:{rel_table}]->(b)
                    ON MATCH SET r.weight = $weight, r.commit_sha = $commit_sha
                    ON CREATE SET r.weight = $weight, r.commit_sha = $commit_sha
                """,
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "weight": rel.weight,
                        "commit_sha": rel.commit_sha or "",
                    },
                )

        except Exception as e:
            logger.error(f"Failed to add relationship {rel_table}: {e}")
            raise

    async def find_entity_by_name(self, name: str) -> str | None:
        """Find entity ID by name (case-insensitive partial match).

        Args:
            name: Entity name or partial name

        Returns:
            Entity ID if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        # Generic entity names that should require exact match
        # This prevents queries like "search" from matching 1,260+ results
        generic_names = {
            "search",
            "main",
            "test",
            "config",
            "get",
            "set",
            "run",
            "init",
            "data",
            "name",
            "value",
            "key",
            "add",
            "delete",
            "update",
            "create",
            "read",
            "write",
            "load",
            "save",
            "process",
            "handle",
        }

        try:
            # Try exact match first
            result = self.conn.execute(
                "MATCH (e:CodeEntity) WHERE e.name = $name RETURN e.id LIMIT 1",
                {"name": name},
            )
            if result.has_next():
                return result.get_next()[0]

            # For generic names, require exact match (no partial matching)
            if name.lower() in generic_names:
                logger.debug(
                    f"Generic name '{name}' requires exact match, no partial matches"
                )
                return None

            # Try case-insensitive contains match for non-generic names
            # Note: Kuzu doesn't have strlen/length for strings in ORDER BY,
            # so we just return first match
            result = self.conn.execute(
                """
                MATCH (e:CodeEntity)
                WHERE lower(e.name) CONTAINS lower($name)
                RETURN e.id
                LIMIT 1
            """,
                {"name": name},
            )
            if result.has_next():
                return result.get_next()[0]

            return None
        except Exception as e:
            logger.error(f"Failed to find entity by name {name}: {e}")
            return None

    async def find_related(
        self, entity_name_or_id: str, max_hops: int = 2
    ) -> list[dict[str, Any]]:
        """Find entities related within N hops.

        Args:
            entity_name_or_id: Starting entity name or ID
            max_hops: Maximum number of hops (default: 2)

        Returns:
            List of related entity dictionaries
        """
        if not self._initialized:
            await self.initialize()

        # Resolve name to ID if needed
        entity_id = entity_name_or_id
        if not entity_name_or_id.startswith(
            ("entity:", "module:", "class:", "function:")
        ):
            resolved = await self.find_entity_by_name(entity_name_or_id)
            if resolved:
                entity_id = resolved
                logger.debug(f"Resolved '{entity_name_or_id}' to '{entity_id}'")
            else:
                logger.warning(f"Could not find entity matching '{entity_name_or_id}'")
                return []

        try:
            result = self.conn.execute(
                f"""
                MATCH (start:CodeEntity {{id: $id}})-[*1..{max_hops}]-(related:CodeEntity)
                WHERE related.id <> start.id
                RETURN DISTINCT related.id AS id,
                                related.name AS name,
                                related.entity_type AS type,
                                related.file_path AS file_path
                LIMIT 50
            """,
                {"id": entity_id},
            )

            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "file_path": row[3],
                    }
                )

            return entities
        except Exception as e:
            logger.error(f"Failed to find related entities for {entity_id}: {e}")
            return []

    async def get_call_graph(self, function_name_or_id: str) -> list[dict[str, Any]]:
        """Get functions called by or calling a function.

        Args:
            function_name_or_id: Function name or entity ID

        Returns:
            List of call relationships
        """
        if not self._initialized:
            await self.initialize()

        # Resolve name to ID if needed
        function_id = function_name_or_id
        if not function_name_or_id.startswith(
            ("entity:", "module:", "class:", "function:")
        ):
            resolved = await self.find_entity_by_name(function_name_or_id)
            if resolved:
                function_id = resolved
                logger.debug(f"Resolved '{function_name_or_id}' to '{function_id}'")
            else:
                logger.warning(
                    f"Could not find function matching '{function_name_or_id}'"
                )
                return []

        try:
            result = self.conn.execute(
                """
                MATCH (f:CodeEntity {id: $id})-[:CALLS]->(called:CodeEntity)
                RETURN called.id AS id, called.name AS name, 'calls' AS direction
                UNION
                MATCH (caller:CodeEntity)-[:CALLS]->(f:CodeEntity {id: $id})
                RETURN caller.id AS id, caller.name AS name, 'called_by' AS direction
            """,
                {"id": function_id},
            )

            calls = []
            while result.has_next():
                row = result.get_next()
                calls.append({"id": row[0], "name": row[1], "direction": row[2]})

            return calls
        except Exception as e:
            logger.error(f"Failed to get call graph for {function_id}: {e}")
            return []

    async def get_inheritance_tree(self, class_name_or_id: str) -> list[dict[str, Any]]:
        """Get class hierarchy (parents and children).

        Args:
            class_name_or_id: Class name or entity ID

        Returns:
            List of inheritance relationships
        """
        if not self._initialized:
            await self.initialize()

        # Resolve name to ID if needed
        class_id = class_name_or_id
        if not class_name_or_id.startswith(
            ("entity:", "module:", "class:", "function:")
        ):
            resolved = await self.find_entity_by_name(class_name_or_id)
            if resolved:
                class_id = resolved
                logger.debug(f"Resolved '{class_name_or_id}' to '{class_id}'")
            else:
                logger.warning(f"Could not find class matching '{class_name_or_id}'")
                return []

        try:
            result = self.conn.execute(
                """
                MATCH (c:CodeEntity {id: $id})-[:INHERITS*]->(parent:CodeEntity)
                RETURN parent.id AS id, parent.name AS name, 'parent' AS relation
                UNION
                MATCH (child:CodeEntity)-[:INHERITS*]->(c:CodeEntity {id: $id})
                RETURN child.id AS id, child.name AS name, 'child' AS relation
            """,
                {"id": class_id},
            )

            hierarchy = []
            while result.has_next():
                row = result.get_next()
                hierarchy.append({"id": row[0], "name": row[1], "relation": row[2]})

            return hierarchy
        except Exception as e:
            logger.error(f"Failed to get inheritance tree for {class_id}: {e}")
            return []

    async def get_doc_references(
        self, entity_name: str, relationship: str = "references"
    ) -> list[dict[str, Any]]:
        """Get documentation that references or documents a code entity.

        Args:
            entity_name: Code entity name to search for
            relationship: Relationship type ('references' or 'documents')

        Returns:
            List of documentation sections
        """
        if not self._initialized:
            await self.initialize()

        rel_type = relationship.upper()
        if rel_type not in ["REFERENCES", "DOCUMENTS"]:
            logger.warning(f"Invalid relationship type: {relationship}")
            return []

        try:
            # Search by entity name (not ID)
            result = self.conn.execute(
                f"""
                MATCH (d:DocSection)-[:{rel_type}]->(e:CodeEntity)
                WHERE e.name = $name
                RETURN d.id AS id,
                       d.name AS title,
                       d.file_path AS file_path,
                       d.line_start AS line_start,
                       d.line_end AS line_end,
                       d.level AS level
                ORDER BY d.file_path, d.line_start
            """,
                {"name": entity_name},
            )

            docs = []
            while result.has_next():
                row = result.get_next()
                docs.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "file_path": row[2],
                        "line_start": row[3],
                        "line_end": row[4],
                        "level": row[5],
                    }
                )

            return docs
        except Exception as e:
            logger.error(f"Failed to get doc references for {entity_name}: {e}")
            return []

    async def get_initial_visualization_data(
        self, max_nodes: int = 100
    ) -> dict[str, Any]:
        """Get initial KG view with entry points and top entities only.

        Returns limited graph for fast initial render:
        - Project node
        - Person nodes
        - Top N most-connected CodeEntities
        - Aggregation placeholders for collapsed groups

        Args:
            max_nodes: Maximum number of initial nodes to return

        Returns:
            Dictionary with nodes, links, and metadata
        """
        if not self._initialized:
            await self.initialize()

        nodes = []
        links = []

        # 1. Project node
        project_result = self.conn.execute(
            "MATCH (p:Project) RETURN p.id, p.name LIMIT 1"
        )
        if project_result.has_next():
            row = project_result.get_next()
            nodes.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": "project",
                    "group": 8,
                    "expandable": False,
                }
            )

        # 2. Person nodes (all - usually small)
        person_result = self.conn.execute("MATCH (p:Person) RETURN p.id, p.name")
        while person_result.has_next():
            row = person_result.get_next()
            nodes.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": "person",
                    "group": 7,
                    "expandable": True,
                }
            )

        # 3. Top connected CodeEntities (by degree)
        # Count incoming + outgoing relationships
        top_entities_result = self.conn.execute("""
            MATCH (e:CodeEntity)
            WITH e,
                 size((e)-[]-()) as degree
            ORDER BY degree DESC
            LIMIT 50
            RETURN e.id, e.name, e.entity_type, e.file_path, degree
        """)
        type_to_group = {"file": 1, "module": 2, "class": 3, "function": 4, "method": 4}
        while top_entities_result.has_next():
            row = top_entities_result.get_next()
            nodes.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2] or "unknown",
                    "file_path": row[3] or "",
                    "group": type_to_group.get(row[2], 0),
                    "degree": row[4],
                    "expandable": row[4] > 0,
                }
            )

        # 4. Aggregation nodes for collapsed groups
        # Count entities by type that aren't in top 50
        type_counts_result = self.conn.execute("""
            MATCH (e:CodeEntity)
            WITH e.entity_type as type, count(e) as cnt
            RETURN type, cnt
        """)
        while type_counts_result.has_next():
            row = type_counts_result.get_next()
            entity_type, count = row[0], row[1]
            if count > 50:  # Only aggregate large groups
                nodes.append(
                    {
                        "id": f"aggregate:{entity_type}",
                        "name": f"{entity_type.title()}s ({count})",
                        "type": "aggregate",
                        "group": 9,
                        "expandable": True,
                        "aggregated_type": entity_type,
                        "aggregated_count": count,
                    }
                )

        # Count DocSections
        doc_count = await self._count_entities("DocSection")
        if doc_count > 0:
            nodes.append(
                {
                    "id": "aggregate:doc_section",
                    "name": f"Doc Sections ({doc_count})",
                    "type": "aggregate",
                    "group": 9,
                    "expandable": True,
                    "aggregated_type": "doc_section",
                    "aggregated_count": doc_count,
                }
            )

        # 5. Get links only between visible nodes
        node_ids = [n["id"] for n in nodes if not n["id"].startswith("aggregate:")]
        if node_ids:
            # Get links between visible nodes
            for rel_type in [
                "CALLS",
                "IMPORTS",
                "INHERITS",
                "CONTAINS",
                "AUTHORED",
                "PART_OF",
            ]:
                try:
                    # Query all relationships of this type and filter in Python
                    rel_result = self.conn.execute(f"""
                        MATCH (a)-[r:{rel_type}]->(b)
                        RETURN a.id, b.id
                    """)
                    node_ids_set = set(node_ids)
                    while rel_result.has_next():
                        row = rel_result.get_next()
                        source_id, target_id = row[0], row[1]
                        # Only include links where both nodes are visible
                        if source_id in node_ids_set and target_id in node_ids_set:
                            links.append(
                                {
                                    "source": source_id,
                                    "target": target_id,
                                    "type": rel_type.lower(),
                                }
                            )
                except Exception as e:
                    logger.debug(f"Skipping {rel_type} relationships: {e}")
                    continue

        total_code = await self._count_entities("CodeEntity")

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "initial_view": True,
                "total_code_entities": total_code,
                "total_doc_sections": doc_count,
                "max_nodes": max_nodes,
            },
        }

    async def _count_entities(self, label: str) -> int:
        """Count entities with given label."""
        try:
            result = self.conn.execute(f"MATCH (n:{label}) RETURN count(n)")
            return result.get_next()[0] if result.has_next() else 0
        except Exception:
            return 0

    async def get_node_neighbors(
        self, node_id: str, hops: int = 1, max_per_type: int = 30
    ) -> dict:
        """Get neighboring nodes and edges for expansion.

        Args:
            node_id: ID of node to expand
            hops: Number of relationship hops (default 1)
            max_per_type: Max neighbors per type before aggregating

        Returns:
            {"nodes": [...], "links": [...], "aggregations": [...]}
        """
        if not self._initialized:
            await self.initialize()

        nodes = []
        links = []
        aggregations = []

        # Handle aggregate node expansion
        if node_id.startswith("aggregate:"):
            entity_type = node_id.split(":", 1)[1]
            # Return sample of this type
            if entity_type == "doc_section":
                result = self.conn.execute(f"""
                    MATCH (d:DocSection)
                    RETURN d.id, d.name, d.file_path
                    LIMIT {max_per_type}
                """)
                while result.has_next():
                    row = result.get_next()
                    nodes.append(
                        {
                            "id": row[0],
                            "name": row[1] or "untitled",
                            "type": "doc_section",
                            "file_path": row[2] or "",
                            "group": 5,
                            "expandable": True,
                        }
                    )
            else:
                result = self.conn.execute(f"""
                    MATCH (e:CodeEntity)
                    WHERE e.entity_type = '{entity_type}'
                    RETURN e.id, e.name, e.entity_type, e.file_path
                    LIMIT {max_per_type}
                """)
                type_to_group = {
                    "file": 1,
                    "module": 2,
                    "class": 3,
                    "function": 4,
                    "method": 4,
                }
                while result.has_next():
                    row = result.get_next()
                    nodes.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "file_path": row[3] or "",
                            "group": type_to_group.get(row[2], 0),
                            "expandable": True,
                        }
                    )
            return {"nodes": nodes, "links": [], "aggregations": []}

        # Get neighbors by relationship type
        neighbor_counts = {}  # type -> count
        neighbor_samples = {}  # type -> [nodes]

        # Escape single quotes in node_id
        safe_node_id = node_id.replace("'", "\\'")

        for rel_type in [
            "CALLS",
            "IMPORTS",
            "INHERITS",
            "CONTAINS",
            "FOLLOWS",
            "REFERENCES",
            "AUTHORED",
            "PART_OF",
        ]:
            try:
                # Outgoing relationships
                result = self.conn.execute(f"""
                    MATCH (a {{id: '{safe_node_id}'}})-[r:{rel_type}]->(b)
                    RETURN b.id, b.name, labels(b)[0] as label
                    LIMIT 100
                """)
                while result.has_next():
                    row = result.get_next()
                    neighbor_type = row[2].lower() if row[2] else "unknown"
                    if neighbor_type not in neighbor_counts:
                        neighbor_counts[neighbor_type] = 0
                        neighbor_samples[neighbor_type] = []
                    neighbor_counts[neighbor_type] += 1
                    if len(neighbor_samples[neighbor_type]) < max_per_type:
                        type_to_group = {
                            "codeentity": 0,
                            "file": 1,
                            "module": 2,
                            "class": 3,
                            "function": 4,
                            "method": 4,
                            "docsection": 5,
                            "tag": 6,
                            "person": 7,
                            "project": 8,
                        }
                        neighbor_samples[neighbor_type].append(
                            {
                                "id": row[0],
                                "name": row[1] or row[0],
                                "type": neighbor_type,
                                "group": type_to_group.get(neighbor_type, 0),
                                "expandable": True,
                            }
                        )
                        links.append(
                            {
                                "source": node_id,
                                "target": row[0],
                                "type": rel_type.lower(),
                            }
                        )

                # Incoming relationships
                result = self.conn.execute(f"""
                    MATCH (a)-[r:{rel_type}]->(b {{id: '{safe_node_id}'}})
                    RETURN a.id, a.name, labels(a)[0] as label
                    LIMIT 100
                """)
                while result.has_next():
                    row = result.get_next()
                    neighbor_type = row[2].lower() if row[2] else "unknown"
                    if neighbor_type not in neighbor_counts:
                        neighbor_counts[neighbor_type] = 0
                        neighbor_samples[neighbor_type] = []
                    neighbor_counts[neighbor_type] += 1
                    if len(neighbor_samples[neighbor_type]) < max_per_type:
                        type_to_group = {
                            "codeentity": 0,
                            "file": 1,
                            "module": 2,
                            "class": 3,
                            "function": 4,
                            "method": 4,
                            "docsection": 5,
                            "tag": 6,
                            "person": 7,
                            "project": 8,
                        }
                        neighbor_samples[neighbor_type].append(
                            {
                                "id": row[0],
                                "name": row[1] or row[0],
                                "type": neighbor_type,
                                "group": type_to_group.get(neighbor_type, 0),
                                "expandable": True,
                            }
                        )
                        links.append(
                            {
                                "source": row[0],
                                "target": node_id,
                                "type": rel_type.lower(),
                            }
                        )
            except Exception as e:
                logger.debug(f"Skipping {rel_type} relationships for {node_id}: {e}")
                continue

        # Build nodes list, aggregate if needed
        for ntype, count in neighbor_counts.items():
            if count > max_per_type:
                # Add samples + aggregation info
                nodes.extend(neighbor_samples[ntype])
                aggregations.append(
                    {
                        "type": ntype,
                        "shown": len(neighbor_samples[ntype]),
                        "total": count,
                        "hidden": count - len(neighbor_samples[ntype]),
                    }
                )
            else:
                nodes.extend(neighbor_samples[ntype])

        return {"nodes": nodes, "links": links, "aggregations": aggregations}

    async def get_visualization_data(self) -> dict[str, Any]:
        """Export knowledge graph data for D3.js visualization with monorepo support.

        Returns:
            Dictionary with nodes and links for force-directed graph:
            {
                "nodes": [{"id": str, "name": str, "type": str, "file_path": str, "group": int}],
                "links": [{"source": str, "target": str, "type": str, "weight": float}],
                "metadata": {"is_monorepo": bool, "subprojects": list[str], ...}
            }
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get all code entities as nodes
            nodes = []
            entities = []
            entity_result = self.conn.execute(
                """
                MATCH (e:CodeEntity)
                RETURN e.id AS id,
                       e.name AS name,
                       e.entity_type AS type,
                       e.file_path AS file_path
                """
            )

            # Collect entities and their file paths for subproject detection
            while entity_result.has_next():
                row = entity_result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2] or "unknown",
                        "file_path": row[3] or "",
                    }
                )

            # Add DocSection nodes
            doc_result = self.conn.execute(
                """
                MATCH (d:DocSection)
                RETURN d.id AS id,
                       d.name AS name,
                       d.file_path AS file_path
                """
            )
            while doc_result.has_next():
                row = doc_result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1] or "untitled",
                        "type": "doc_section",
                        "file_path": row[2] or "",
                    }
                )

            # Add Tag nodes
            tag_result = self.conn.execute(
                "MATCH (t:Tag) RETURN t.id AS id, t.name AS name"
            )
            while tag_result.has_next():
                row = tag_result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": "tag",
                        "file_path": "",
                    }
                )

            # Add Person nodes
            person_result = self.conn.execute(
                "MATCH (p:Person) RETURN p.id AS id, p.name AS name"
            )
            while person_result.has_next():
                row = person_result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": "person",
                        "file_path": "",
                    }
                )

            # Add Project nodes
            project_result = self.conn.execute(
                "MATCH (p:Project) RETURN p.id AS id, p.name AS name"
            )
            while project_result.has_next():
                row = project_result.get_next()
                entities.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": "project",
                        "file_path": "",
                    }
                )

            # Detect subprojects from file paths (monorepo patterns)
            subprojects: dict[str, str] = {}  # path_prefix -> subproject_name
            monorepo_patterns = ["packages", "apps", "libs", "modules", "services"]

            for entity in entities:
                file_path = entity["file_path"]
                if not file_path:
                    continue

                # Handle both absolute and relative paths
                # Convert absolute paths to relative by finding monorepo pattern in path
                parts = file_path.split("/")

                # Find first occurrence of monorepo pattern in path
                for i, part in enumerate(parts):
                    if part in monorepo_patterns and i + 1 < len(parts):
                        # Found pattern like packages/* or apps/*
                        subproject_name = parts[i + 1]
                        # Use relative path for subproject (packages/foo, apps/bar, etc.)
                        subproject_path = f"{parts[i]}/{parts[i + 1]}"
                        if subproject_path not in subprojects:
                            subprojects[subproject_path] = subproject_name
                        break

            # Determine if this is a monorepo (more than 1 subproject)
            is_monorepo = len(subprojects) > 1

            # Color palette for subprojects (GitHub-style colors matching chunk graph)
            colors = ["#238636", "#1f6feb", "#8957e5", "#f85149", "#d29922", "#3fb950"]

            # Create subproject root nodes if monorepo detected
            if is_monorepo:
                logger.debug(f"Detected monorepo with {len(subprojects)} subprojects")
                for i, (path, name) in enumerate(subprojects.items()):
                    nodes.append(
                        {
                            "id": f"subproject:{name}",
                            "name": name,
                            "type": "subproject",
                            "file_path": path,
                            "color": colors[i % len(colors)],
                            "group": 0,  # Special group for subproject roots
                        }
                    )

            # Map entity types to color groups
            type_to_group = {
                "file": 1,
                "module": 2,
                "class": 3,
                "function": 4,
                "method": 4,
                "doc_section": 5,
                "tag": 6,
                "person": 7,
                "project": 8,
            }

            # Create entity nodes with subproject assignment
            for entity in entities:
                file_path = entity["file_path"]
                subproject = None

                # Assign entity to subproject if in monorepo
                if is_monorepo:
                    # Check if file path contains any subproject pattern
                    for path, name in subprojects.items():
                        # Match against relative path pattern (packages/foo, apps/bar, etc.)
                        if f"/{path}/" in file_path or file_path.startswith(path):
                            subproject = name
                            break

                node = {
                    "id": entity["id"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "file_path": file_path,
                    "group": type_to_group.get(entity["type"], 0),
                }

                if subproject:
                    node["subproject"] = subproject

                nodes.append(node)

                # Link to subproject root if in monorepo
                if subproject:
                    # Will be added to links below
                    pass

            # Get all relationships as links
            links = []

            # Code relationships (CodeEntity -> CodeEntity)
            code_relationship_types = ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]
            for rel_type in code_relationship_types:
                try:
                    rel_result = self.conn.execute(
                        f"""
                        MATCH (a:CodeEntity)-[r:{rel_type}]->(b:CodeEntity)
                        RETURN a.id AS source,
                               b.id AS target,
                               r.weight AS weight
                    """
                    )

                    while rel_result.has_next():
                        row = rel_result.get_next()
                        links.append(
                            {
                                "source": row[0],
                                "target": row[1],
                                "type": rel_type.lower(),
                                "weight": row[2] if row[2] is not None else 1.0,
                            }
                        )
                except Exception as e:
                    logger.debug(f"No {rel_type} relationships found: {e}")
                    continue

            # Documentation relationships (DocSection -> DocSection)
            try:
                follows_result = self.conn.execute(
                    """
                    MATCH (a:DocSection)-[r:FOLLOWS]->(b:DocSection)
                    RETURN a.id AS source, b.id AS target
                    """
                )
                while follows_result.has_next():
                    row = follows_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "follows",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No FOLLOWS relationships found: {e}")

            # Documentation -> Code references
            try:
                references_result = self.conn.execute(
                    """
                    MATCH (a:DocSection)-[r:REFERENCES]->(b:CodeEntity)
                    RETURN a.id AS source, b.id AS target
                    """
                )
                while references_result.has_next():
                    row = references_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "references",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No REFERENCES relationships found: {e}")

            # Documentation -> Code demonstrations
            try:
                demonstrates_result = self.conn.execute(
                    """
                    MATCH (a:DocSection)-[r:DEMONSTRATES]->(b:CodeEntity)
                    RETURN a.id AS source, b.id AS target
                    """
                )
                while demonstrates_result.has_next():
                    row = demonstrates_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "demonstrates",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No DEMONSTRATES relationships found: {e}")

            # Tag relationships (DocSection/CodeEntity -> Tag)
            try:
                has_tag_result = self.conn.execute(
                    """
                    MATCH (a)-[r:HAS_TAG]->(t:Tag)
                    WHERE a:DocSection OR a:CodeEntity
                    RETURN a.id AS source, t.id AS target
                    """
                )
                while has_tag_result.has_next():
                    row = has_tag_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "has_tag",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No HAS_TAG relationships found: {e}")

            # Person relationships (AUTHORED)
            try:
                authored_result = self.conn.execute(
                    """
                    MATCH (p:Person)-[r:AUTHORED]->(e)
                    WHERE e:CodeEntity OR e:DocSection
                    RETURN p.id AS source, e.id AS target
                    """
                )
                while authored_result.has_next():
                    row = authored_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "authored",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No AUTHORED relationships found: {e}")

            # Part of relationships (various -> Project)
            try:
                part_of_result = self.conn.execute(
                    """
                    MATCH (a)-[r:PART_OF]->(p:Project)
                    RETURN a.id AS source, p.id AS target
                    """
                )
                while part_of_result.has_next():
                    row = part_of_result.get_next()
                    links.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "type": "part_of",
                            "weight": 1.0,
                        }
                    )
            except Exception as e:
                logger.debug(f"No PART_OF relationships found: {e}")

            # Add links from subproject roots to entities in monorepo
            if is_monorepo:
                for node in nodes:
                    if node.get("subproject") and node["type"] != "subproject":
                        links.append(
                            {
                                "source": f"subproject:{node['subproject']}",
                                "target": node["id"],
                                "type": "contains",
                                "weight": 1.0,
                            }
                        )

            return {
                "nodes": nodes,
                "links": links,
                "metadata": {
                    "is_monorepo": is_monorepo,
                    "subprojects": list(subprojects.values()) if is_monorepo else [],
                    "total_nodes": len(nodes),
                    "total_links": len(links),
                },
            }

        except Exception as e:
            logger.error(f"Failed to export KG visualization data: {e}")
            return {"nodes": [], "links": [], "metadata": {}}

    async def has_relationships(self) -> bool:
        """Check if KG has code relationships built.

        Checks for code-structure relationships (calls, imports, inherits, contains)
        rather than documentation relationships (follows, demonstrates, etc.).

        Returns:
            True if any code relationship count > 0, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        stats = await self.get_stats()
        relationships = stats.get("relationships", {})

        # Check for code-structure relationships specifically
        code_relationships = ["calls", "imports", "inherits", "contains"]
        code_rel_count = sum(relationships.get(rel, 0) for rel in code_relationships)
        return code_rel_count > 0

    async def get_detailed_stats(self) -> dict[str, Any]:
        """Get detailed knowledge graph statistics with entity type breakdowns.

        Returns:
            Dictionary with detailed entity counts by type and relationship counts
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Count total code entities
            entity_result = self.conn.execute(
                "MATCH (e:CodeEntity) RETURN count(e) AS count"
            )
            total_entities = (
                entity_result.get_next()[0] if entity_result.has_next() else 0
            )

            # Get all distinct entity types and their counts
            entity_types_result = self.conn.execute(
                """
                MATCH (e:CodeEntity)
                RETURN e.entity_type AS type, count(e) AS count
                ORDER BY count DESC
                """
            )
            entity_types = {}
            while entity_types_result.has_next():
                row = entity_types_result.get_next()
                entity_types[row[0]] = row[1]

            # Count doc sections
            doc_result = self.conn.execute(
                "MATCH (d:DocSection) RETURN count(d) AS count"
            )
            doc_count = doc_result.get_next()[0] if doc_result.has_next() else 0

            # Count tags
            tag_result = self.conn.execute("MATCH (t:Tag) RETURN count(t) AS count")
            tag_count = tag_result.get_next()[0] if tag_result.has_next() else 0

            # Count persons
            person_result = self.conn.execute(
                "MATCH (p:Person) RETURN count(p) AS count"
            )
            person_count = (
                person_result.get_next()[0] if person_result.has_next() else 0
            )

            # Count projects
            project_result = self.conn.execute(
                "MATCH (p:Project) RETURN count(p) AS count"
            )
            project_count = (
                project_result.get_next()[0] if project_result.has_next() else 0
            )

            # Count repositories
            repo_result = self.conn.execute(
                "MATCH (r:Repository) RETURN count(r) AS count"
            )
            repo_count = repo_result.get_next()[0] if repo_result.has_next() else 0

            # Count branches
            branch_result = self.conn.execute(
                "MATCH (b:Branch) RETURN count(b) AS count"
            )
            branch_count = (
                branch_result.get_next()[0] if branch_result.has_next() else 0
            )

            # Count commits
            commit_result = self.conn.execute(
                "MATCH (c:Commit) RETURN count(c) AS count"
            )
            commit_count = (
                commit_result.get_next()[0] if commit_result.has_next() else 0
            )

            # Count languages
            lang_result = self.conn.execute(
                "MATCH (l:ProgrammingLanguage) RETURN count(l) AS count"
            )
            lang_count = lang_result.get_next()[0] if lang_result.has_next() else 0

            # Count frameworks
            framework_result = self.conn.execute(
                "MATCH (f:ProgrammingFramework) RETURN count(f) AS count"
            )
            framework_count = (
                framework_result.get_next()[0] if framework_result.has_next() else 0
            )

            # Count relationships by type
            rel_counts = {}
            for rel_type in [
                "CALLS",
                "IMPORTS",
                "INHERITS",
                "CONTAINS",
                "REFERENCES",
                "DOCUMENTS",
                "FOLLOWS",
                "HAS_TAG",
                "DEMONSTRATES",
                "LINKS_TO",
                "AUTHORED",
                "MODIFIED",
                "PART_OF",
                "MODIFIES",
                "BRANCHED_FROM",
                "COMMITTED_TO",
                "BELONGS_TO",
                "WRITTEN_IN",
                "USES_FRAMEWORK",
                "FRAMEWORK_FOR",
            ]:
                try:
                    rel_result = self.conn.execute(
                        f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                    )
                    rel_counts[rel_type.lower()] = (
                        rel_result.get_next()[0] if rel_result.has_next() else 0
                    )
                except Exception:
                    rel_counts[rel_type.lower()] = 0

            return {
                "total_entities": total_entities
                + doc_count
                + tag_count
                + person_count
                + project_count
                + repo_count
                + branch_count
                + commit_count
                + lang_count
                + framework_count,
                "code_entities": total_entities,
                "entity_types": entity_types,
                "doc_sections": doc_count,
                "tags": tag_count,
                "persons": person_count,
                "projects": project_count,
                "repositories": repo_count,
                "branches": branch_count,
                "commits": commit_count,
                "languages": lang_count,
                "frameworks": framework_count,
                "relationships": rel_counts,
                "database_path": str(self.db_path / "code_kg"),
            }
        except Exception as e:
            logger.error(f"Failed to get detailed KG stats: {e}")
            return {
                "total_entities": 0,
                "code_entities": 0,
                "entity_types": {},
                "doc_sections": 0,
                "tags": 0,
                "persons": 0,
                "projects": 0,
                "repositories": 0,
                "branches": 0,
                "commits": 0,
                "languages": 0,
                "frameworks": 0,
                "relationships": {},
                "error": str(e),
            }

    def get_stats_sync(self) -> dict[str, Any]:
        """Get knowledge graph statistics (synchronous).

        Returns:
            Dictionary with entity counts and relationship counts
        """
        if not self._initialized:
            raise RuntimeError(
                "KnowledgeGraph not initialized. Call initialize_sync() first."
            )

        try:
            # Count code entities
            entity_result = self._execute_query(
                "MATCH (e:CodeEntity) RETURN count(e) AS count"
            )
            entity_count = (
                entity_result.get_next()[0] if entity_result.has_next() else 0
            )

            # Count doc sections
            doc_result = self._execute_query(
                "MATCH (d:DocSection) RETURN count(d) AS count"
            )
            doc_count = doc_result.get_next()[0] if doc_result.has_next() else 0

            # Count tags
            tag_result = self._execute_query("MATCH (t:Tag) RETURN count(t) AS count")
            tag_count = tag_result.get_next()[0] if tag_result.has_next() else 0

            # Count document nodes
            document_result = self._execute_query(
                "MATCH (d:Document) RETURN count(d) AS count"
            )
            document_count = (
                document_result.get_next()[0] if document_result.has_next() else 0
            )

            # Total entities
            total_entities = entity_count + doc_count + tag_count + document_count

            # Get relationship counts
            relationships = {}
            for rel_type in [
                "CALLS",
                "IMPORTS",
                "INHERITS",
                "CONTAINS",
                "REFERENCES",
                "DOCUMENTS",
                "FOLLOWS",
                "HAS_TAG",
                "DEMONSTRATES",
                "LINKS_TO",
                "CONTAINS_SECTION",
                "RELATED_TO",
                "DESCRIBES",
            ]:
                try:
                    rel_result = self._execute_query(
                        f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                    )
                    count = rel_result.get_next()[0] if rel_result.has_next() else 0
                    relationships[rel_type.lower()] = count
                except Exception:
                    relationships[rel_type.lower()] = 0

            return {
                "total_entities": total_entities,
                "code_entities": entity_count,
                "doc_sections": doc_count,
                "documents": document_count,
                "tags": tag_count,
                "relationships": relationships,
                "database_path": str(self.db_path / "code_kg"),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_entities": 0,
                "code_entities": 0,
                "doc_sections": 0,
                "documents": 0,
                "tags": 0,
                "relationships": {},
                "database_path": str(self.db_path / "code_kg"),
            }

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary with entity counts and relationship counts
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Count code entities
            entity_result = self.conn.execute(
                "MATCH (e:CodeEntity) RETURN count(e) AS count"
            )
            entity_count = (
                entity_result.get_next()[0] if entity_result.has_next() else 0
            )

            # Count doc sections
            doc_result = self.conn.execute(
                "MATCH (d:DocSection) RETURN count(d) AS count"
            )
            doc_count = doc_result.get_next()[0] if doc_result.has_next() else 0

            # Count tags
            tag_result = self.conn.execute("MATCH (t:Tag) RETURN count(t) AS count")
            tag_count = tag_result.get_next()[0] if tag_result.has_next() else 0

            # Count document nodes
            document_result = self.conn.execute(
                "MATCH (d:Document) RETURN count(d) AS count"
            )
            document_count = (
                document_result.get_next()[0] if document_result.has_next() else 0
            )

            # Count persons
            person_result = self.conn.execute(
                "MATCH (p:Person) RETURN count(p) AS count"
            )
            person_count = (
                person_result.get_next()[0] if person_result.has_next() else 0
            )

            # Count projects
            project_result = self.conn.execute(
                "MATCH (p:Project) RETURN count(p) AS count"
            )
            project_count = (
                project_result.get_next()[0] if project_result.has_next() else 0
            )

            # Count repositories
            repo_result = self.conn.execute(
                "MATCH (r:Repository) RETURN count(r) AS count"
            )
            repo_count = repo_result.get_next()[0] if repo_result.has_next() else 0

            # Count branches
            branch_result = self.conn.execute(
                "MATCH (b:Branch) RETURN count(b) AS count"
            )
            branch_count = (
                branch_result.get_next()[0] if branch_result.has_next() else 0
            )

            # Count commits
            commit_result = self.conn.execute(
                "MATCH (c:Commit) RETURN count(c) AS count"
            )
            commit_count = (
                commit_result.get_next()[0] if commit_result.has_next() else 0
            )

            # Count relationships by type
            rel_counts = {}
            for rel_type in [
                "CALLS",
                "IMPORTS",
                "INHERITS",
                "CONTAINS",
                "REFERENCES",
                "DOCUMENTS",
                "FOLLOWS",
                "HAS_TAG",
                "DEMONSTRATES",
                "LINKS_TO",
                "AUTHORED",
                "MODIFIED",
                "PART_OF",
                "MODIFIES",
                "BRANCHED_FROM",
                "COMMITTED_TO",
                "BELONGS_TO",
                "CONTAINS_SECTION",
                "RELATED_TO",
                "DESCRIBES",
            ]:
                try:
                    rel_result = self.conn.execute(
                        f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                    )
                    rel_counts[rel_type.lower()] = (
                        rel_result.get_next()[0] if rel_result.has_next() else 0
                    )
                except Exception:
                    rel_counts[rel_type.lower()] = 0

            return {
                "total_entities": entity_count
                + doc_count
                + tag_count
                + document_count
                + person_count
                + project_count
                + repo_count
                + branch_count
                + commit_count,
                "code_entities": entity_count,
                "doc_sections": doc_count,
                "documents": document_count,
                "tags": tag_count,
                "persons": person_count,
                "projects": project_count,
                "repositories": repo_count,
                "branches": branch_count,
                "commits": commit_count,
                "relationships": rel_counts,
                "database_path": str(self.db_path / "code_kg"),
            }
        except Exception as e:
            logger.error(f"Failed to get KG stats: {e}")
            return {
                "total_entities": 0,
                "code_entities": 0,
                "doc_sections": 0,
                "documents": 0,
                "tags": 0,
                "persons": 0,
                "projects": 0,
                "repositories": 0,
                "branches": 0,
                "commits": 0,
                "relationships": {},
                "error": str(e),
            }

    async def get_document_ontology(
        self, category: str | None = None
    ) -> dict[str, Any]:
        """Get the document ontology tree grouped by category.

        Returns documents organized by doc_category, with their sections,
        cross-references (RELATED_TO), and tags (via DocSection HAS_TAG).

        Args:
            category: Optional category filter (readme, guide, api_doc, etc.)

        Returns:
            Dict with categories, documents, sections, cross-references
        """
        if not self._initialized:
            await self.initialize()

        result: dict[str, Any] = {
            "categories": {},
            "total_documents": 0,
            "total_sections": 0,
            "total_cross_references": 0,
        }

        # Get all documents grouped by category
        try:
            query = """
                MATCH (d:Document)
                RETURN d.id, d.file_path, d.title, d.doc_category,
                       d.word_count, d.section_count
                ORDER BY d.doc_category, d.title
            """
            docs_result = self._execute_query(query, {})
            rows = []
            while docs_result.has_next():
                rows.append(docs_result.get_next())

            for row in rows:
                doc_id, file_path, title, cat, word_count, section_count = row
                if category and cat != category:
                    continue
                if cat not in result["categories"]:
                    result["categories"][cat] = []

                doc_info: dict[str, Any] = {
                    "id": doc_id,
                    "file_path": file_path,
                    "title": title,
                    "word_count": word_count,
                    "section_count": section_count,
                    "sections": [],
                    "cross_references": [],
                    "tags": [],
                }

                result["categories"][cat].append(doc_info)
                result["total_documents"] += 1
                result["total_sections"] += section_count or 0

        except Exception as e:
            logger.warning(f"Failed to query documents: {e}")
            return result

        # Get sections for each document
        try:
            sections_result = self._execute_query(
                """
                MATCH (d:Document)-[:CONTAINS_SECTION]->(s:DocSection)
                RETURN d.id, s.name, s.level, s.line_start
                ORDER BY d.id, s.line_start
                """,
                {},
            )

            sections_by_doc: dict[str, list[dict[str, Any]]] = {}
            while sections_result.has_next():
                row = sections_result.get_next()
                doc_id = row[0]
                if doc_id not in sections_by_doc:
                    sections_by_doc[doc_id] = []
                sections_by_doc[doc_id].append(
                    {
                        "name": row[1],
                        "level": row[2],
                        "line": row[3],
                    }
                )

            for category_docs in result["categories"].values():
                for doc in category_docs:
                    doc["sections"] = sections_by_doc.get(doc["id"], [])

        except Exception as e:
            logger.debug(f"Failed to query sections: {e}")

        # Get cross-references (RELATED_TO between Documents)
        try:
            refs_result = self._execute_query(
                """
                MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document)
                RETURN d1.id, d2.file_path, d2.title
                """,
                {},
            )

            refs_by_doc: dict[str, list[dict[str, str]]] = {}
            while refs_result.has_next():
                row = refs_result.get_next()
                doc_id = row[0]
                if doc_id not in refs_by_doc:
                    refs_by_doc[doc_id] = []
                refs_by_doc[doc_id].append(
                    {
                        "file_path": row[1],
                        "title": row[2],
                    }
                )
                result["total_cross_references"] += 1

            for category_docs in result["categories"].values():
                for doc in category_docs:
                    doc["cross_references"] = refs_by_doc.get(doc["id"], [])

        except Exception as e:
            logger.debug(f"Failed to query cross-references: {e}")

        # Get tags per document (via DocSection HAS_TAG)
        try:
            tags_result = self._execute_query(
                """
                MATCH (d:Document)-[:CONTAINS_SECTION]->(s:DocSection)-[:HAS_TAG]->(t:Tag)
                RETURN d.id, COLLECT(DISTINCT t.name)
                """,
                {},
            )

            tags_by_doc: dict[str, list[str]] = {}
            while tags_result.has_next():
                row = tags_result.get_next()
                tags_by_doc[row[0]] = row[1]

            for category_docs in result["categories"].values():
                for doc in category_docs:
                    doc["tags"] = tags_by_doc.get(doc["id"], [])

        except Exception as e:
            logger.debug(f"Failed to query tags: {e}")

        return result

    async def get_cross_entity_samples(
        self, limit_per_type: int = 3
    ) -> dict[str, list[dict[str, str]]]:
        """Get sample cross-entity relationships to show connections between different node types.

        Args:
            limit_per_type: Number of samples per relationship pattern (default: 3)

        Returns:
            Dictionary mapping relationship patterns to sample relationships:
            {
                "DocSection -> DOCUMENTS -> CodeEntity": [
                    {"source": "API Guide", "rel": "DOCUMENTS", "target": "UserService"},
                    ...
                ],
                "DocSection -> DEMONSTRATES -> Tag": [...],
                "Person -> AUTHORED -> CodeEntity": [...],
                ...
            }
        """
        if not self._initialized:
            await self.initialize()

        samples: dict[str, list[dict[str, str]]] = {}

        # Define cross-entity relationship patterns to sample
        patterns = [
            # Documentation to Code
            (
                "DocSection",
                "DOCUMENTS",
                "CodeEntity",
                "DocSection → DOCUMENTS → CodeEntity",
            ),
            (
                "DocSection",
                "DEMONSTRATES",
                "Tag",
                "DocSection → DEMONSTRATES → Tag",
            ),
            (
                "DocSection",
                "REFERENCES",
                "CodeEntity",
                "DocSection → REFERENCES → CodeEntity",
            ),
            # People to Code
            ("Person", "AUTHORED", "CodeEntity", "Person → AUTHORED → CodeEntity"),
            ("Person", "MODIFIED", "CodeEntity", "Person → MODIFIED → CodeEntity"),
            # Tags (reverse direction - CodeEntity has tags via DocSection)
            ("DocSection", "HAS_TAG", "Tag", "DocSection → HAS_TAG → Tag"),
            # Project relationships
            ("CodeEntity", "PART_OF", "Project", "CodeEntity → PART_OF → Project"),
        ]

        for source_label, rel_type, target_label, pattern_name in patterns:
            try:
                result = self.conn.execute(
                    f"""
                    MATCH (s:{source_label})-[r:{rel_type}]->(t:{target_label})
                    RETURN s.name AS source, '{rel_type}' AS rel, t.name AS target
                    LIMIT {limit_per_type}
                """
                )

                pattern_samples = []
                while result.has_next():
                    row = result.get_next()
                    pattern_samples.append(
                        {"source": row[0], "rel": row[1], "target": row[2]}
                    )

                if pattern_samples:
                    samples[pattern_name] = pattern_samples

            except Exception as e:
                logger.debug(f"No samples found for {pattern_name}: {e}")
                continue

        return samples

    def close_sync(self):
        """Close database connection (synchronous)."""
        with self._kuzu_lock:
            if self.conn:
                self.conn = None
            if self.db:
                self.db = None
            self._initialized = False

        logger.debug("Knowledge graph connection closed")

    async def close(self):
        """Close database connection."""
        with self._kuzu_lock:
            if self.conn:
                self.conn = None
            if self.db:
                self.db = None
            self._initialized = False

        logger.debug("Knowledge graph connection closed")
