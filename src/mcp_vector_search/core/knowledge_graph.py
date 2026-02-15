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

    async def initialize(self):
        """Initialize Kuzu database with schema."""
        if self._initialized:
            return

        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Kuzu database
        db_dir = self.db_path / "code_kg"
        self.db = kuzu.Database(str(db_dir))
        self.conn = kuzu.Connection(self.db)

        # Create schema
        self._create_schema()
        self._initialized = True
        logger.info(f"✓ Knowledge graph initialized at {db_dir}")

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
                + person_count
                + project_count,
                "code_entities": entity_count,
                "doc_sections": doc_count,
                "tags": tag_count,
                "persons": person_count,
                "projects": project_count,
                "relationships": rel_counts,
                "database_path": str(self.db_path / "code_kg"),
            }
        except Exception as e:
            logger.error(f"Failed to get KG stats: {e}")
            return {
                "total_entities": 0,
                "code_entities": 0,
                "doc_sections": 0,
                "tags": 0,
                "persons": 0,
                "projects": 0,
                "relationships": {},
                "error": str(e),
            }

    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None
        self._initialized = False
        logger.debug("Knowledge graph connection closed")
