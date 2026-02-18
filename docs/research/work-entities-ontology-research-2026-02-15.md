# Work Entities Ontology Research: People & Projects in Code Knowledge Graphs

**Date:** 2026-02-15
**Researcher:** Claude (Research Agent)
**Purpose:** Design entity types and relationships for tracking people (authors, contributors) and projects in the mcp-vector-search Knowledge Graph
**Context:** Extends existing KG with code entities (Function, Class, Module) and documentation entities (DocSection, DocEntity)

---

## Executive Summary

This research defines ontologies for adding **work entities** (People, Projects, Organizations) to the mcp-vector-search Knowledge Graph. The goal is to enable queries like:
- "Who authored this function?"
- "What code is part of project X?"
- "What did developer Y work on recently?"
- "Which team owns this module?"

**Key Findings:**

1. **PROV-O** provides Agent/Person/Organization types with attribution relationships (wasAttributedTo, wasAssociatedWith)
2. **Schema.org** Person/Organization types offer rich properties for authorship, affiliation, and ownership
3. **DOAP** (Description of a Project) defines project-developer relationships (maintainer, developer, documenter, tester)
4. **Git history** provides rich temporal authorship data via `git log` and `git blame`
5. **GitHub API** offers contributor statistics, PR authorship, and organizational membership
6. **CODEOWNERS** files define explicit code ownership and review responsibilities
7. **Package metadata** (pyproject.toml, package.json) contains author/maintainer information

**Recommended Approach:**
- Start with git history (readily available, no API dependencies)
- Add Person and Project entities to existing KG schema
- Extract authorship from git blame (file-level and line-level attribution)
- Extract project structure from package metadata
- Phase 2: Enhance with GitHub API data (contributors, teams, organizations)

---

## Table of Contents

1. [Ontology Standards Review](#1-ontology-standards-review)
2. [Proposed Entity Types](#2-proposed-entity-types)
3. [Proposed Relationship Types](#3-proposed-relationship-types)
4. [Data Sources Analysis](#4-data-sources-analysis)
5. [Implementation Plan](#5-implementation-plan)
6. [Example Queries](#6-example-queries)
7. [Privacy Considerations](#7-privacy-considerations)
8. [Performance & Scalability](#8-performance--scalability)

---

## 1. Ontology Standards Review

### 1.1 PROV-O (Provenance Ontology)

**Source:** W3C Recommendation
**Relevance:** Models agents, attribution, and activities
**Maturity:** Production-ready, widely adopted

#### Core Entity Types

| Type | Description | Use in Code KG |
|------|-------------|----------------|
| **Agent** | Responsible entity | Base type for Person/Organization/Bot |
| **Person** | Human actor | Developer, contributor, author |
| **Organization** | Group of agents | Company, team, open source project |
| **SoftwareAgent** | Automated actor | CI/CD bot, code generator, automation script |

#### Key Relationships

| Relationship | Source | Target | Description |
|--------------|--------|--------|-------------|
| **wasAttributedTo** | Entity | Agent | "This code was written by Person X" |
| **wasAssociatedWith** | Activity | Agent | "This commit was made by Developer Y" |
| **actedOnBehalfOf** | Agent | Agent | "Bot acted on behalf of Developer" |
| **qualifiedAttribution** | Entity | Attribution | Detailed attribution with roles and timestamps |

#### Implementation Mapping

```cypher
// Person attribution
(CodeEntity {name: 'search_code', file: 'search.py'})
  -[WAS_ATTRIBUTED_TO {role: 'author', timestamp: '2026-02-15'}]->(Person {name: 'Bob Matsuoka'})

// Commit activity
(Activity {type: 'commit', sha: 'abc123', message: 'Add search feature'})
  -[WAS_ASSOCIATED_WITH {role: 'committer'}]->(Person {name: 'Bob Matsuoka'})

// Bot acting for human
(SoftwareAgent {name: 'dependabot'})
  -[ACTED_ON_BEHALF_OF]->(Organization {name: 'GitHub'})
```

**Complexity:** Medium - requires activity/agent modeling
**Priority:** HIGH - foundational for authorship tracking

---

### 1.2 Schema.org Person & Organization

**Source:** https://schema.org/Person, https://schema.org/Organization
**Relevance:** Rich metadata for people and organizations
**Maturity:** Production-ready, widely adopted

#### Person Properties

| Property | Description | KG Application |
|----------|-------------|----------------|
| **name** | Full name | Display name for developer |
| **email** | Contact email | Git author email (handle privacy!) |
| **givenName** | First name | Parse from git author name |
| **familyName** | Last name | Parse from git author name |
| **worksFor** | Employer | Link to Organization node |
| **affiliation** | Associated orgs | Multiple organization memberships |
| **knows** | Social connections | Team relationships |
| **alumniOf** | Education | Optional metadata |
| **hasOccupation** | Job role | "Backend Engineer", "Data Scientist" |

#### Organization Properties

| Property | Description | KG Application |
|----------|-------------|----------------|
| **name** | Org name | "Anthropic", "mcp-vector-search project" |
| **member** | Members | Developers in organization |
| **parentOrganization** | Parent org | Subsidiary relationships |
| **subOrganization** | Sub-orgs | Teams within company |
| **founder** | Founder | Project creator |
| **owns** | Ownership | Code owned by organization |
| **maintainer** | Maintenance | Who maintains the project |

#### Implementation Example

```python
@dataclass
class Person:
    """Person entity (developer, contributor, author)."""
    id: str                          # Unique identifier (email hash or GitHub ID)
    name: str                        # Full name from git config
    email: str | None = None         # Git email (anonymize if needed)
    github_username: str | None = None   # GitHub username (from API)
    affiliation: str | None = None   # Company/organization
    roles: list[str] = None          # ['author', 'maintainer', 'reviewer']

    # Contribution statistics
    commits_count: int = 0           # Total commits
    lines_added: int = 0             # Total lines added
    lines_removed: int = 0           # Total lines removed
    first_commit: str | None = None  # Timestamp of first contribution
    last_commit: str | None = None   # Timestamp of last contribution

@dataclass
class Organization:
    """Organization entity (company, team, project)."""
    id: str                          # Unique identifier (domain or GitHub org)
    name: str                        # Display name
    org_type: str                    # 'company', 'team', 'open_source'
    members: list[str] = None        # Person IDs
    parent_org: str | None = None    # Parent organization ID
```

**Complexity:** Simple - straightforward metadata properties
**Priority:** HIGH - essential for authorship queries

---

### 1.3 DOAP (Description of a Project)

**Source:** https://github.com/edumbill/doap
**Relevance:** Project-developer relationships
**Maturity:** Widely used in open source

#### Core Entity Types

| Type | Description | KG Application |
|------|-------------|----------------|
| **Project** | Software project | Repository, codebase, module |
| **Version** | Release version | v1.0.0, v2.2.21 |
| **Repository** | Code repository | Git repo, GitHub URL |
| **BugDatabase** | Issue tracker | GitHub Issues, Jira |

#### Project-Person Relationships

| Property | Description | Use Case |
|----------|-------------|----------|
| **maintainer** | Person maintaining project | Active maintainers |
| **developer** | Person developing code | Core developers |
| **documenter** | Person writing docs | Documentation authors |
| **tester** | Person testing code | QA contributors |
| **translator** | Person translating docs | i18n contributors |
| **helper** | General contributor | Community helpers |

#### Implementation Example

```cypher
// Project structure
(Project {
  name: 'mcp-vector-search',
  shortdesc: 'CLI-first semantic code search',
  homepage: 'https://github.com/...',
  created: '2024-08-01'
})
  -[HAS_MAINTAINER]->(Person {name: 'Robert Matsuoka', role: 'creator'})
  -[HAS_DEVELOPER]->(Person {name: 'Contributor A', role: 'contributor'})
  -[HAS_REPOSITORY]->(Repository {url: 'github.com/...'})
  -[HAS_VERSION]->(Version {number: '2.2.21', date: '2026-02-15'})

// Code belongs to project
(CodeEntity {name: 'search_code'})
  -[PART_OF]->(Project {name: 'mcp-vector-search'})
```

**Complexity:** Simple - project-level metadata
**Priority:** MEDIUM - useful for multi-project codebases

---

### 1.4 Comparison Table

| Ontology | Focus | Entity Types | Relationships | Best For |
|----------|-------|--------------|---------------|----------|
| **PROV-O** | Provenance | Agent, Person, Organization, SoftwareAgent | wasAttributedTo, wasAssociatedWith | Temporal tracking, commit history |
| **Schema.org** | Metadata | Person, Organization | worksFor, memberOf, knows, owns | Rich user profiles, organizational structure |
| **DOAP** | Projects | Project, Version, Repository | maintainer, developer, documenter | Project-developer relationships |

**Recommended Hybrid Approach:**
- Use **PROV-O** for attribution relationships (ATTRIBUTED_TO, ASSOCIATED_WITH)
- Use **Schema.org** for entity properties (Person name, email, Organization structure)
- Use **DOAP** for project roles (maintainer, developer, tester)

---

## 2. Proposed Entity Types

### 2.1 Person Entity

**Purpose:** Represent developers, authors, contributors

**Properties:**

```python
@dataclass
class Person:
    """Person entity in knowledge graph."""
    id: str                          # Primary key (email_hash or github_id)
    name: str                        # Full name (from git config)
    display_name: str | None = None  # Preferred display name
    email: str | None = None         # Git email (anonymize for privacy)
    email_hash: str | None = None    # SHA256 hash of email (for privacy)

    # External identifiers
    github_username: str | None = None   # GitHub username (from API)
    github_id: int | None = None         # GitHub user ID

    # Organizational affiliation
    affiliation: str | None = None   # Current employer/organization
    works_for: str | None = None     # Organization ID

    # Roles and permissions
    roles: list[str] = None          # ['author', 'maintainer', 'reviewer', 'tester']

    # Contribution statistics (from git history)
    commits_count: int = 0           # Total commits
    files_modified: int = 0          # Unique files touched
    lines_added: int = 0             # Total lines added
    lines_removed: int = 0           # Total lines removed
    first_commit: str | None = None  # ISO timestamp of first contribution
    last_commit: str | None = None   # ISO timestamp of last contribution

    # GitHub statistics (optional, from API)
    github_contributions: int | None = None   # GitHub contribution count
    github_followers: int | None = None       # GitHub followers

    # Timestamps
    created_at: str | None = None    # When entity was created in KG
    updated_at: str | None = None    # When entity was last updated
```

**Extraction Sources:**
- Git log: `git log --format="%an|%ae|%ad"`
- Git config: `.git/config` or `~/.gitconfig`
- pyproject.toml: `authors = [{name, email}]`
- package.json: `authors`, `contributors`
- GitHub API: `/users/{username}`, `/repos/{owner}/{repo}/contributors`

**Privacy Handling:**
- Store email as SHA256 hash by default
- Allow opt-in for full email storage
- Use display name instead of full name when available
- Respect GitHub privacy settings (noreply emails)

---

### 2.2 Organization Entity

**Purpose:** Represent companies, teams, open source projects

**Properties:**

```python
@dataclass
class Organization:
    """Organization entity in knowledge graph."""
    id: str                          # Primary key (domain_hash or github_org_id)
    name: str                        # Display name ("Anthropic", "mcp-vector-search")
    org_type: str                    # 'company', 'team', 'open_source', 'community'

    # External identifiers
    github_org: str | None = None    # GitHub organization name
    github_id: int | None = None     # GitHub org ID
    domain: str | None = None        # Company domain (anthropic.com)

    # Organizational structure
    parent_org_id: str | None = None     # Parent organization
    sub_orgs: list[str] = None           # Child organizations/teams

    # Members
    member_count: int = 0            # Number of members

    # Metadata
    description: str | None = None   # Organization description
    homepage: str | None = None      # Website URL

    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None
```

**Extraction Sources:**
- Email domains: Extract from `@company.com` in git emails
- GitHub API: `/orgs/{org}`, `/orgs/{org}/members`
- pyproject.toml: `authors` field (infer from email domains)
- CODEOWNERS: Team definitions like `@org/team-name`

---

### 2.3 Project Entity

**Purpose:** Represent software projects, repositories, modules

**Properties:**

```python
@dataclass
class Project:
    """Project entity in knowledge graph."""
    id: str                          # Primary key (repo_url_hash or project_name)
    name: str                        # Display name ("mcp-vector-search")
    short_desc: str | None = None    # Brief description

    # Repository information
    repo_url: str | None = None      # Git repository URL
    repo_type: str = 'git'           # 'git', 'github', 'gitlab'

    # Project metadata
    license: str | None = None       # License type (MIT, Apache-2.0)
    language: str | None = None      # Primary language (Python, TypeScript)
    homepage: str | None = None      # Project website

    # Ownership
    owner_id: str | None = None      # Person or Organization ID
    owner_type: str | None = None    # 'person' or 'organization'

    # Versions
    current_version: str | None = None   # Latest version (v2.2.21)

    # Statistics
    total_commits: int = 0           # Total commits in history
    total_contributors: int = 0      # Unique contributors
    lines_of_code: int = 0           # Total LOC

    # Timestamps
    created_at: str | None = None    # Project creation date
    last_updated: str | None = None  # Last commit date
```

**Extraction Sources:**
- pyproject.toml: `name`, `description`, `license`, `authors`
- package.json: `name`, `description`, `license`, `author`, `contributors`
- Git remote: `git remote get-url origin`
- GitHub API: `/repos/{owner}/{repo}`

---

### 2.4 Team Entity (Optional)

**Purpose:** Represent groups within organizations (e.g., CODEOWNERS teams)

**Properties:**

```python
@dataclass
class Team:
    """Team entity (subtype of Organization)."""
    id: str                          # Primary key (team_name_hash or github_team_id)
    name: str                        # Team name ("backend-team", "ml-engineers")
    org_id: str                      # Parent organization

    # GitHub integration
    github_team_id: int | None = None    # GitHub team ID
    github_slug: str | None = None       # GitHub team slug

    # Members
    member_ids: list[str] = None     # List of Person IDs

    # Responsibilities
    owned_paths: list[str] = None    # File paths owned by team (from CODEOWNERS)

    # Metadata
    description: str | None = None   # Team description
```

**Extraction Sources:**
- CODEOWNERS: `@org/team-name`
- GitHub API: `/orgs/{org}/teams`, `/teams/{team_id}/members`

---

## 3. Proposed Relationship Types

### 3.1 Person-to-Code Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **AUTHORED** | Person | CodeEntity | Person wrote/created code | HIGH |
| **MODIFIED** | Person | CodeEntity | Person changed code | HIGH |
| **REVIEWED** | Person | CodeEntity | Person reviewed changes | MEDIUM |
| **OWNS** | Person | CodeEntity | Person owns/maintains code | HIGH |
| **CONTRIBUTED_TO** | Person | Project | Person contributed to project | MEDIUM |

#### Implementation Example

```cypher
// Authorship (from git blame)
(Person {name: 'Bob Matsuoka', email_hash: 'abc123...'})
  -[AUTHORED {
    timestamp: '2026-02-15T05:35:22Z',
    commit_sha: 'abc123',
    lines_authored: 120
  }]->(CodeEntity {name: 'KnowledgeGraph', file: 'knowledge_graph.py'})

// File modification (from git log)
(Person {name: 'Bob Matsuoka'})
  -[MODIFIED {
    timestamp: '2026-02-15T12:07:59Z',
    commit_sha: 'def456',
    lines_added: 50,
    lines_removed: 10,
    commit_message: 'feat: add Phase 1 text relationships'
  }]->(CodeEntity {name: 'knowledge_graph.py', type: 'file'})

// Code ownership (from CODEOWNERS)
(Person {github_username: 'bobmatnyc'})
  -[OWNS {
    source: 'CODEOWNERS',
    responsibility: 'reviewer',
    pattern: 'src/mcp_vector_search/core/*.py'
  }]->(CodeEntity {file: 'src/mcp_vector_search/core/knowledge_graph.py'})

// Project contribution (aggregated)
(Person {name: 'Bob Matsuoka'})
  -[CONTRIBUTED_TO {
    commits: 250,
    lines_added: 15000,
    lines_removed: 3000,
    first_commit: '2024-08-01',
    last_commit: '2026-02-15'
  }]->(Project {name: 'mcp-vector-search'})
```

---

### 3.2 Person-to-Organization Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **MEMBER_OF** | Person | Organization | Person is member of org | HIGH |
| **WORKS_FOR** | Person | Organization | Person employed by org | MEDIUM |
| **AFFILIATED_WITH** | Person | Organization | Person associated with org | LOW |
| **MAINTAINS** | Person | Project | Person maintains project | HIGH |

#### Implementation Example

```cypher
// Organizational membership (from email domain)
(Person {email: 'bob@anthropic.com'})
  -[MEMBER_OF {
    inferred_from: 'email_domain',
    confidence: 0.8
  }]->(Organization {name: 'Anthropic', domain: 'anthropic.com'})

// GitHub org membership (from GitHub API)
(Person {github_username: 'bobmatnyc'})
  -[MEMBER_OF {
    source: 'github_api',
    role: 'member',
    joined_at: '2023-05-01'
  }]->(Organization {github_org: 'anthropic-ai'})

// Project maintainer (from pyproject.toml)
(Person {name: 'Robert Matsuoka', email: 'bob@matsuoka.com'})
  -[MAINTAINS {
    source: 'pyproject.toml',
    role: 'creator',
    since: '2024-08-01'
  }]->(Project {name: 'mcp-vector-search'})
```

---

### 3.3 Project-to-Code Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **CONTAINS** | Project | CodeEntity | Project contains code | HIGH |
| **PART_OF** | CodeEntity | Project | Code belongs to project | HIGH |
| **DEPENDS_ON** | Project | Project | Project dependency | MEDIUM |
| **OWNS** | Organization | Project | Organization owns project | MEDIUM |

#### Implementation Example

```cypher
// Code belongs to project
(CodeEntity {name: 'search_code', file: 'search.py'})
  -[PART_OF {
    module: 'core.search',
    added_in_commit: 'abc123',
    added_at: '2024-09-15'
  }]->(Project {name: 'mcp-vector-search'})

// Project contains code
(Project {name: 'mcp-vector-search'})
  -[CONTAINS {
    file_count: 150,
    lines_of_code: 25000,
    languages: ['Python', 'JavaScript']
  }]->(CodeEntity {type: 'module', name: 'core'})

// Project ownership
(Organization {name: 'Robert Matsuoka', type: 'individual'})
  -[OWNS {
    license: 'MIT',
    ownership_type: 'creator'
  }]->(Project {name: 'mcp-vector-search'})
```

---

### 3.4 Team-to-Code Relationships (Optional)

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **OWNS** | Team | CodeEntity | Team owns/maintains code | HIGH |
| **REVIEWS** | Team | CodeEntity | Team responsible for reviews | MEDIUM |

#### Implementation Example

```cypher
// CODEOWNERS team ownership
(Team {name: 'backend-team', github_slug: 'org/backend'})
  -[OWNS {
    source: 'CODEOWNERS',
    pattern: 'src/mcp_vector_search/core/*.py',
    responsibility: 'reviewer'
  }]->(CodeEntity {file: 'src/mcp_vector_search/core/search.py'})
```

---

## 4. Data Sources Analysis

### 4.1 Git History (Primary Source)

**Availability:** Always available in git repositories
**Cost:** Free, local processing
**Reliability:** High (authoritative source of truth)

#### 4.1.1 Git Log (Commit History)

**Command:** `git log --all --format="%H|%an|%ae|%at|%s" --numstat`

**Data Extracted:**
- Commit SHA
- Author name
- Author email
- Commit timestamp (Unix epoch)
- Commit message
- Files modified (with line counts)

**Example Output:**
```
abc123|Bob Matsuoka|bob@matsuoka.com|1771149038|feat: add knowledge graph
50      10      src/mcp_vector_search/core/knowledge_graph.py
20      5       tests/test_knowledge_graph.py
```

**Parsing Strategy:**
```python
async def extract_commits(repo_path: Path) -> list[Commit]:
    """Extract commit history with file modifications."""
    cmd = [
        "git", "log", "--all",
        "--format=%H|%an|%ae|%at|%s",
        "--numstat"
    ]

    output = subprocess.check_output(cmd, cwd=repo_path, text=True)

    commits = []
    for line in output.split('\n'):
        if '|' in line:  # Commit metadata line
            sha, author, email, timestamp, message = line.split('|')
            commit = Commit(
                sha=sha,
                author_name=author,
                author_email=email,
                timestamp=datetime.fromtimestamp(int(timestamp)),
                message=message,
                files=[]
            )
            commits.append(commit)
        elif '\t' in line:  # File stat line
            added, removed, filepath = line.split('\t')
            commit.files.append({
                'path': filepath,
                'lines_added': int(added) if added.isdigit() else 0,
                'lines_removed': int(removed) if removed.isdigit() else 0,
            })

    return commits
```

**Entities Created:**
- Person (from author name/email)
- MODIFIED relationship (Person → CodeEntity)
- CONTRIBUTED_TO relationship (Person → Project)

**Performance:**
- Typical repo (1000 commits): ~1-2 seconds
- Large repo (50,000 commits): ~10-30 seconds
- Can be cached and incrementally updated

---

#### 4.1.2 Git Blame (Line-Level Attribution)

**Command:** `git blame --line-porcelain <file>`

**Data Extracted:**
- Commit SHA for each line
- Author name for each line
- Author email for each line
- Author timestamp for each line
- Line content

**Example Output:**
```
039dc771c8519277d6ffb0261cb287057f71971b 1 1 20
author Bob Matsuoka
author-mail <bobmatnyc@users.noreply.github.com>
author-time 1771149038
author-tz -0500
committer Bob Matsuoka
committer-mail <bobmatnyc@users.noreply.github.com>
committer-time 1771149038
committer-tz -0500
summary feat: add temporal knowledge graph with Kuzu integration (#99)
filename src/mcp_vector_search/core/knowledge_graph.py
	"""Temporal Knowledge Graph using Kuzu.
```

**Parsing Strategy:**
```python
async def extract_blame(file_path: Path) -> dict[str, BlameInfo]:
    """Extract line-level authorship for a file."""
    cmd = ["git", "blame", "--line-porcelain", str(file_path)]
    output = subprocess.check_output(cmd, text=True)

    blame_info = {}
    current_sha = None
    current_author = None
    current_email = None
    current_time = None
    line_num = 0

    for line in output.split('\n'):
        if line.startswith('author '):
            current_author = line[7:]
        elif line.startswith('author-mail '):
            current_email = line[12:].strip('<>')
        elif line.startswith('author-time '):
            current_time = int(line[12:])
        elif line.startswith('\t'):  # Line content
            line_num += 1
            blame_info[line_num] = BlameInfo(
                author_name=current_author,
                author_email=current_email,
                timestamp=datetime.fromtimestamp(current_time),
                commit_sha=current_sha,
                line_content=line[1:]
            )

    return blame_info
```

**Use Cases:**
- Function-level authorship: "Who wrote this function?"
- Ownership heuristics: Author of majority of lines owns the file
- Temporal analysis: When was each part of the code written?

**Entities Created:**
- AUTHORED relationship (Person → CodeEntity) with line counts

**Performance Considerations:**
- File-level blame: ~50-200ms per file
- For large codebases (1000+ files): Run incrementally, cache results
- Only run blame for files modified in recent commits (delta approach)

---

### 4.2 Package Metadata (Secondary Source)

**Availability:** Most projects have metadata files
**Cost:** Free, local parsing
**Reliability:** Medium (author lists may be outdated)

#### 4.2.1 Python (pyproject.toml, setup.py)

**File:** `pyproject.toml`

**Data Extracted:**
```toml
[project]
name = "mcp-vector-search"
authors = [
    {name = "Robert Matsuoka", email = "bob@matsuoka.com"},
]
maintainers = [
    {name = "Robert Matsuoka", email = "bob@matsuoka.com"},
]
```

**Parsing Strategy:**
```python
import toml

def extract_python_authors(pyproject_path: Path) -> list[Person]:
    """Extract authors from pyproject.toml."""
    with open(pyproject_path) as f:
        data = toml.load(f)

    authors = []
    for author in data.get('project', {}).get('authors', []):
        person = Person(
            id=hash_email(author['email']),
            name=author['name'],
            email=author['email'],
            roles=['author']
        )
        authors.append(person)

    for maintainer in data.get('project', {}).get('maintainers', []):
        person = Person(
            id=hash_email(maintainer['email']),
            name=maintainer['name'],
            email=maintainer['email'],
            roles=['maintainer']
        )
        authors.append(person)

    return authors
```

**Entities Created:**
- Person (from authors/maintainers)
- MAINTAINS relationship (Person → Project)

---

#### 4.2.2 JavaScript/TypeScript (package.json)

**File:** `package.json`

**Data Extracted:**
```json
{
  "name": "example-project",
  "author": "John Doe <john@example.com>",
  "contributors": [
    "Jane Smith <jane@example.com>",
    "Bob Johnson <bob@example.com>"
  ]
}
```

**Parsing Strategy:**
```python
import json

def extract_npm_authors(package_json_path: Path) -> list[Person]:
    """Extract authors from package.json."""
    with open(package_json_path) as f:
        data = json.load(f)

    authors = []

    # Parse author (string or object)
    if 'author' in data:
        author = data['author']
        if isinstance(author, str):
            name, email = parse_author_string(author)  # "Name <email>"
        else:
            name, email = author['name'], author.get('email')

        authors.append(Person(
            id=hash_email(email),
            name=name,
            email=email,
            roles=['author']
        ))

    # Parse contributors
    for contributor in data.get('contributors', []):
        if isinstance(contributor, str):
            name, email = parse_author_string(contributor)
        else:
            name, email = contributor['name'], contributor.get('email')

        authors.append(Person(
            id=hash_email(email),
            name=name,
            email=email,
            roles=['contributor']
        ))

    return authors
```

---

### 4.3 CODEOWNERS File (Tertiary Source)

**Availability:** Optional, primarily used in GitHub/GitLab
**Cost:** Free, local parsing
**Reliability:** High (explicit ownership declarations)

**File:** `.github/CODEOWNERS` or `CODEOWNERS`

**Format:**
```
# Format: <file-pattern> @username @org/team

# Global rule
* @bobmatnyc

# Backend team owns core
src/mcp_vector_search/core/*.py @org/backend-team @bobmatnyc

# Frontend team owns UI
src/mcp_vector_search/ui/*.js @org/frontend-team

# Docs team owns documentation
docs/** @org/docs-team
```

**Parsing Strategy:**
```python
def parse_codeowners(codeowners_path: Path) -> list[OwnershipRule]:
    """Parse CODEOWNERS file to extract ownership rules."""
    rules = []

    with open(codeowners_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            pattern = parts[0]
            owners = parts[1:]  # @username or @org/team

            for owner in owners:
                if owner.startswith('@'):
                    owner_name = owner[1:]  # Remove '@'

                    if '/' in owner_name:
                        # Team ownership
                        org, team = owner_name.split('/')
                        rule = OwnershipRule(
                            pattern=pattern,
                            owner_type='team',
                            owner_id=f"{org}/{team}",
                            responsibility='reviewer'
                        )
                    else:
                        # Individual ownership
                        rule = OwnershipRule(
                            pattern=pattern,
                            owner_type='person',
                            owner_id=owner_name,  # GitHub username
                            responsibility='reviewer'
                        )

                    rules.append(rule)

    return rules
```

**Entities Created:**
- OWNS relationship (Person/Team → CodeEntity)
- Team entity (if not exists)

**Use Cases:**
- "Who should review changes to this file?"
- "Which team owns the authentication module?"
- "Who is responsible for the CI/CD scripts?"

---

### 4.4 GitHub API (Optional Enhancement)

**Availability:** Requires GitHub repo + API token
**Cost:** Free (60 req/hour unauthenticated, 5000 req/hour authenticated)
**Reliability:** High (official GitHub data)

#### 4.4.1 Contributors API

**Endpoint:** `GET /repos/{owner}/{repo}/contributors`

**Data Extracted:**
```json
[
  {
    "login": "bobmatnyc",
    "id": 1234567,
    "avatar_url": "https://avatars.githubusercontent.com/u/1234567",
    "contributions": 250,
    "type": "User"
  }
]
```

**Use Cases:**
- Enrich Person entities with GitHub usernames
- Get accurate contribution counts
- Link to GitHub profiles

---

#### 4.4.2 User API

**Endpoint:** `GET /users/{username}`

**Data Extracted:**
```json
{
  "login": "bobmatnyc",
  "id": 1234567,
  "name": "Bob Matsuoka",
  "email": "bob@matsuoka.com",
  "company": "Anthropic",
  "location": "San Francisco",
  "bio": "Software Engineer",
  "public_repos": 50,
  "followers": 100
}
```

**Use Cases:**
- Get full names and affiliations
- Infer organizational memberships from company field
- Enrich Person entities with profile data

---

#### 4.4.3 Organization Members API

**Endpoint:** `GET /orgs/{org}/members`

**Data Extracted:**
```json
[
  {
    "login": "bobmatnyc",
    "id": 1234567,
    "type": "User"
  }
]
```

**Use Cases:**
- Discover organizational memberships
- Create MEMBER_OF relationships
- Build organizational structure

---

### 4.5 Data Source Priority Matrix

| Source | Priority | Availability | Cost | Reliability | Use Case |
|--------|----------|--------------|------|-------------|----------|
| **Git Log** | HIGH | Always | Free | High | Commit history, author tracking |
| **Git Blame** | HIGH | Always | Free | High | Line-level authorship |
| **pyproject.toml** | MEDIUM | Often | Free | Medium | Project metadata, author list |
| **package.json** | MEDIUM | Often | Free | Medium | Project metadata, author list |
| **CODEOWNERS** | MEDIUM | Rare | Free | High | Explicit ownership declarations |
| **GitHub API** | LOW | Optional | Limited | High | Enhanced profiles, org structure |

**Recommended Implementation Order:**
1. **Phase 1:** Git log + git blame (foundational, always available)
2. **Phase 2:** Package metadata (quick wins for project/author info)
3. **Phase 3:** CODEOWNERS (if available, high-value ownership data)
4. **Phase 4:** GitHub API (optional enhancement, requires authentication)

---

## 5. Implementation Plan

### Phase 1: Git-Based Authorship (Week 1-2)

**Goal:** Extract Person entities and authorship relationships from git history

#### Step 1.1: Extend KG Schema

```python
# knowledge_graph.py

@dataclass
class Person:
    """Person entity (developer, author, contributor)."""
    id: str                          # email_hash (SHA256)
    name: str                        # Full name from git
    email_hash: str                  # SHA256 of email (privacy)
    display_name: str | None = None  # Optional display name

    # Statistics from git
    commits_count: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_modified: int = 0
    first_commit: str | None = None  # ISO timestamp
    last_commit: str | None = None   # ISO timestamp

    # Optional fields
    github_username: str | None = None
    affiliation: str | None = None

@dataclass
class AuthorshipRelationship:
    """Authorship edge in knowledge graph."""
    source_id: str                   # Person ID
    target_id: str                   # CodeEntity ID
    relationship_type: str           # 'authored', 'modified'

    # Metadata
    commit_sha: str                  # Git commit SHA
    timestamp: str                   # ISO timestamp
    lines_authored: int = 0          # Lines attributed to author
    commit_message: str | None = None
```

#### Step 1.2: Create Kuzu Schema

```python
def _create_schema(self):
    # Existing tables...

    # NEW: Person node table
    self.conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Person (
            id STRING PRIMARY KEY,
            name STRING,
            email_hash STRING,
            display_name STRING,
            commits_count INT64,
            lines_added INT64,
            lines_removed INT64,
            files_modified INT64,
            first_commit TIMESTAMP,
            last_commit TIMESTAMP,
            github_username STRING,
            affiliation STRING,
            created_at TIMESTAMP DEFAULT current_timestamp()
        )
    """)

    # NEW: AUTHORED relationship (Person -> CodeEntity)
    self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS AUTHORED (
            FROM Person TO CodeEntity,
            commit_sha STRING,
            timestamp TIMESTAMP,
            lines_authored INT64,
            commit_message STRING,
            MANY_MANY
        )
    """)

    # NEW: MODIFIED relationship (Person -> CodeEntity)
    self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS MODIFIED (
            FROM Person TO CodeEntity,
            commit_sha STRING,
            timestamp TIMESTAMP,
            lines_added INT64,
            lines_removed INT64,
            commit_message STRING,
            MANY_MANY
        )
    """)
```

#### Step 1.3: Extract Git Log Data

```python
# git_extractor.py

import subprocess
from datetime import datetime
from pathlib import Path

class GitExtractor:
    """Extract authorship data from git history."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    async def extract_commits(self) -> list[dict]:
        """Extract commit history with file modifications."""
        cmd = [
            "git", "log", "--all",
            "--format=%H|%an|%ae|%at|%s",
            "--numstat"
        ]

        output = subprocess.check_output(
            cmd,
            cwd=self.repo_path,
            text=True
        )

        commits = []
        current_commit = None

        for line in output.split('\n'):
            if '|' in line:  # Commit metadata
                if current_commit:
                    commits.append(current_commit)

                sha, author, email, timestamp, message = line.split('|', 4)
                current_commit = {
                    'sha': sha,
                    'author_name': author,
                    'author_email': email,
                    'timestamp': datetime.fromtimestamp(int(timestamp)),
                    'message': message,
                    'files': []
                }
            elif '\t' in line and current_commit:  # File stat
                parts = line.split('\t')
                if len(parts) >= 3:
                    added = parts[0].strip()
                    removed = parts[1].strip()
                    filepath = parts[2].strip()

                    current_commit['files'].append({
                        'path': filepath,
                        'lines_added': int(added) if added.isdigit() else 0,
                        'lines_removed': int(removed) if removed.isdigit() else 0,
                    })

        if current_commit:
            commits.append(current_commit)

        return commits

    async def extract_blame(self, file_path: Path) -> dict[int, dict]:
        """Extract line-level authorship for a file."""
        cmd = ["git", "blame", "--line-porcelain", str(file_path)]

        try:
            output = subprocess.check_output(
                cmd,
                cwd=self.repo_path,
                text=True,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return {}  # File doesn't exist or not tracked

        blame_data = {}
        current_sha = None
        current_author = None
        current_email = None
        current_time = None
        line_num = 0

        for line in output.split('\n'):
            if len(line) > 40 and line[40] == ' ':  # SHA line
                current_sha = line[:40]
            elif line.startswith('author '):
                current_author = line[7:]
            elif line.startswith('author-mail '):
                current_email = line[12:].strip('<>')
            elif line.startswith('author-time '):
                current_time = int(line[12:])
            elif line.startswith('\t'):  # Line content
                line_num += 1
                blame_data[line_num] = {
                    'author_name': current_author,
                    'author_email': current_email,
                    'timestamp': datetime.fromtimestamp(current_time),
                    'commit_sha': current_sha,
                }

        return blame_data
```

#### Step 1.4: Build Person Entities and Relationships

```python
# kg_builder.py enhancement

class KGBuilder:
    async def build_authorship_graph(self):
        """Build Person entities and authorship relationships."""
        logger.info("Building authorship graph from git history...")

        git_extractor = GitExtractor(self.project_path)

        # Step 1: Extract commits
        commits = await git_extractor.extract_commits()
        logger.info(f"Extracted {len(commits)} commits")

        # Step 2: Aggregate person statistics
        person_stats = self._aggregate_person_stats(commits)

        # Step 3: Create Person entities
        for person_id, stats in person_stats.items():
            person = Person(
                id=person_id,
                name=stats['name'],
                email_hash=person_id,  # Already hashed
                commits_count=stats['commits'],
                lines_added=stats['lines_added'],
                lines_removed=stats['lines_removed'],
                files_modified=len(stats['files']),
                first_commit=stats['first_commit'].isoformat(),
                last_commit=stats['last_commit'].isoformat(),
            )
            await self.kg.add_person(person)

        # Step 4: Create MODIFIED relationships
        for commit in commits:
            person_id = hash_email(commit['author_email'])

            for file_info in commit['files']:
                # Find CodeEntity for this file
                code_entity_id = await self._find_code_entity_by_path(
                    file_info['path']
                )

                if code_entity_id:
                    rel = AuthorshipRelationship(
                        source_id=person_id,
                        target_id=code_entity_id,
                        relationship_type='modified',
                        commit_sha=commit['sha'],
                        timestamp=commit['timestamp'].isoformat(),
                        lines_added=file_info['lines_added'],
                        lines_removed=file_info['lines_removed'],
                        commit_message=commit['message'],
                    )
                    await self.kg.add_authorship_relationship(rel)

        logger.info(f"Created {len(person_stats)} Person entities")

    def _aggregate_person_stats(self, commits: list[dict]) -> dict:
        """Aggregate statistics for each person."""
        stats = {}

        for commit in commits:
            email = commit['author_email']
            person_id = hash_email(email)

            if person_id not in stats:
                stats[person_id] = {
                    'name': commit['author_name'],
                    'email': email,
                    'commits': 0,
                    'lines_added': 0,
                    'lines_removed': 0,
                    'files': set(),
                    'first_commit': commit['timestamp'],
                    'last_commit': commit['timestamp'],
                }

            s = stats[person_id]
            s['commits'] += 1
            s['last_commit'] = max(s['last_commit'], commit['timestamp'])

            for file_info in commit['files']:
                s['lines_added'] += file_info['lines_added']
                s['lines_removed'] += file_info['lines_removed']
                s['files'].add(file_info['path'])

        return stats

def hash_email(email: str) -> str:
    """Hash email for privacy (SHA256)."""
    import hashlib
    return hashlib.sha256(email.encode()).hexdigest()[:16]
```

**Expected Results:**
- Person entities for all git authors
- MODIFIED relationships tracking all file changes
- Statistics: commits, lines added/removed, files modified

**Performance:**
- Small repo (1000 commits, 50 authors): ~2-5 seconds
- Medium repo (10,000 commits, 200 authors): ~10-30 seconds
- Large repo (100,000 commits, 1000 authors): ~2-5 minutes

---

### Phase 2: Project Metadata Extraction (Week 2)

**Goal:** Extract Project entities and maintainer relationships

#### Step 2.1: Parse Package Metadata

```python
# metadata_extractor.py

class MetadataExtractor:
    """Extract project metadata from package files."""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    async def extract_project_info(self) -> dict | None:
        """Extract project information from metadata files."""
        # Try pyproject.toml first
        pyproject = self.project_path / 'pyproject.toml'
        if pyproject.exists():
            return await self._parse_pyproject(pyproject)

        # Try package.json
        package_json = self.project_path / 'package.json'
        if package_json.exists():
            return await self._parse_package_json(package_json)

        # Fallback: infer from git remote
        return await self._infer_from_git()

    async def _parse_pyproject(self, path: Path) -> dict:
        """Parse pyproject.toml for project info."""
        import toml

        with open(path) as f:
            data = toml.load(f)

        project = data.get('project', {})

        return {
            'name': project.get('name'),
            'description': project.get('description'),
            'license': project.get('license', {}).get('file'),
            'authors': project.get('authors', []),
            'maintainers': project.get('maintainers', []),
            'homepage': project.get('urls', {}).get('Homepage'),
            'repository': project.get('urls', {}).get('Repository'),
        }
```

#### Step 2.2: Create Project Entity

```python
@dataclass
class Project:
    """Project entity in knowledge graph."""
    id: str
    name: str
    short_desc: str | None = None
    license: str | None = None
    homepage: str | None = None
    repo_url: str | None = None

    # Statistics (from git)
    total_commits: int = 0
    total_contributors: int = 0
    lines_of_code: int = 0

    created_at: str | None = None
    last_updated: str | None = None
```

#### Step 2.3: Create MAINTAINS Relationships

```python
async def build_project_graph(self):
    """Build Project entity and maintainer relationships."""
    metadata = await self.metadata_extractor.extract_project_info()

    if metadata:
        project = Project(
            id=metadata['name'],
            name=metadata['name'],
            short_desc=metadata['description'],
            license=metadata['license'],
            homepage=metadata['homepage'],
            repo_url=metadata['repository'],
        )
        await self.kg.add_project(project)

        # Create MAINTAINS relationships
        for author in metadata.get('authors', []):
            person_id = hash_email(author['email'])
            rel = ProjectRelationship(
                source_id=person_id,
                target_id=project.id,
                relationship_type='maintains',
                role='author',
            )
            await self.kg.add_project_relationship(rel)
```

---

### Phase 3: CODEOWNERS Integration (Week 3)

**Goal:** Extract explicit code ownership rules

```python
async def build_ownership_graph(self):
    """Build ownership relationships from CODEOWNERS."""
    codeowners_path = self.project_path / '.github' / 'CODEOWNERS'

    if not codeowners_path.exists():
        codeowners_path = self.project_path / 'CODEOWNERS'

    if not codeowners_path.exists():
        logger.info("No CODEOWNERS file found, skipping ownership extraction")
        return

    rules = self._parse_codeowners(codeowners_path)

    for rule in rules:
        # Match files against pattern
        matching_files = self._match_codeowners_pattern(rule.pattern)

        for file_path in matching_files:
            code_entity_id = await self._find_code_entity_by_path(file_path)

            if code_entity_id:
                if rule.owner_type == 'person':
                    # Person ownership
                    person_id = await self._find_person_by_github_username(
                        rule.owner_id
                    )
                    if person_id:
                        rel = OwnershipRelationship(
                            source_id=person_id,
                            target_id=code_entity_id,
                            relationship_type='owns',
                            source='CODEOWNERS',
                            responsibility='reviewer',
                        )
                        await self.kg.add_ownership_relationship(rel)
```

---

### Phase 4: GitHub API Enhancement (Week 4, Optional)

**Goal:** Enrich Person entities with GitHub data

```python
class GitHubEnhancer:
    """Enhance KG with GitHub API data."""

    def __init__(self, token: str):
        self.token = token
        self.client = httpx.AsyncClient()

    async def enrich_person(self, person: Person) -> Person:
        """Enrich Person entity with GitHub profile data."""
        if not person.github_username:
            return person

        url = f"https://api.github.com/users/{person.github_username}"
        headers = {"Authorization": f"token {self.token}"}

        resp = await self.client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()

            person.name = data.get('name', person.name)
            person.affiliation = data.get('company')
            person.github_id = data['id']

        return person
```

---

## 6. Example Queries

### 6.1 Query: "Who authored this function?"

```cypher
MATCH (p:Person)-[a:AUTHORED]->(c:CodeEntity {name: 'search_code'})
RETURN p.name AS author,
       p.email_hash AS email_hash,
       a.lines_authored AS lines,
       a.timestamp AS when
ORDER BY lines DESC
LIMIT 5
```

**Expected Output:**
```
author             | email_hash     | lines | when
-------------------|----------------|-------|--------------------
Bob Matsuoka       | abc123...      | 250   | 2024-09-15T10:30:00
```

---

### 6.2 Query: "What code is part of project X?"

```cypher
MATCH (c:CodeEntity)-[r:PART_OF]->(pr:Project {name: 'mcp-vector-search'})
RETURN c.name AS entity,
       c.entity_type AS type,
       c.file_path AS file
ORDER BY c.entity_type, c.name
LIMIT 50
```

**Expected Output:**
```
entity              | type     | file
--------------------|----------|----------------------------------
KnowledgeGraph      | class    | core/knowledge_graph.py
search_code         | function | core/search.py
index_project       | function | core/indexer.py
```

---

### 6.3 Query: "What did developer Y work on recently?"

```cypher
MATCH (p:Person {name: 'Bob Matsuoka'})-[m:MODIFIED]->(c:CodeEntity)
WHERE m.timestamp > '2026-02-01T00:00:00Z'
RETURN c.name AS entity,
       c.file_path AS file,
       m.timestamp AS when,
       m.commit_message AS what,
       m.lines_added + m.lines_removed AS changes
ORDER BY m.timestamp DESC
LIMIT 20
```

**Expected Output:**
```
entity              | file                    | when                | what                          | changes
--------------------|-------------------------|---------------------|-------------------------------|--------
knowledge_graph.py  | core/knowledge_graph.py | 2026-02-15 12:07:59 | feat: add Phase 1 text rel... | 150
DocSection          | core/knowledge_graph.py | 2026-02-15 05:35:22 | feat: add temporal KG         | 200
```

---

### 6.4 Query: "Who are the top contributors to this project?"

```cypher
MATCH (p:Person)-[c:CONTRIBUTED_TO]->(pr:Project {name: 'mcp-vector-search'})
RETURN p.name AS contributor,
       c.commits AS commits,
       c.lines_added AS added,
       c.lines_removed AS removed,
       c.first_commit AS since
ORDER BY c.commits DESC
LIMIT 10
```

**Expected Output:**
```
contributor       | commits | added  | removed | since
------------------|---------|--------|---------|------------
Bob Matsuoka      | 250     | 25000  | 3000    | 2024-08-01
```

---

### 6.5 Query: "Which team owns the authentication module?"

```cypher
MATCH (t:Team)-[o:OWNS]->(c:CodeEntity)
WHERE c.file_path CONTAINS 'auth'
RETURN t.name AS team,
       count(c) AS files_owned,
       collect(c.file_path)[..5] AS sample_files
```

**Expected Output:**
```
team          | files_owned | sample_files
--------------|-------------|----------------------------------
backend-team  | 5           | [core/auth.py, api/auth.py, ...]
```

---

### 6.6 Query: "Find co-authors (people who edited the same files)"

```cypher
MATCH (p1:Person)-[:MODIFIED]->(c:CodeEntity)<-[:MODIFIED]-(p2:Person)
WHERE p1.id < p2.id  // Avoid duplicates
WITH p1, p2, count(c) AS shared_files
WHERE shared_files >= 3
RETURN p1.name AS person1,
       p2.name AS person2,
       shared_files
ORDER BY shared_files DESC
LIMIT 10
```

**Expected Output:**
```
person1       | person2       | shared_files
--------------|---------------|-------------
Bob Matsuoka  | Alice Smith   | 15
Bob Matsuoka  | Charlie Brown | 8
```

---

## 7. Privacy Considerations

### 7.1 Email Address Handling

**Problem:** Git commits contain email addresses (PII)

**Solution:** Hash emails by default

```python
import hashlib

def hash_email(email: str) -> str:
    """Hash email using SHA256 (one-way, privacy-preserving)."""
    return hashlib.sha256(email.encode()).hexdigest()[:16]  # First 16 chars

# Example:
# "bob@example.com" -> "5d41402abc4b2a76"
```

**Benefits:**
- Prevents email exposure in KG queries
- Still allows entity deduplication (same email → same hash)
- One-way function (cannot reverse to get original email)

**Trade-offs:**
- Cannot display actual email addresses
- Cannot send emails to users directly

**Alternative:** Store full email with opt-in consent flag

```python
@dataclass
class Person:
    email_hash: str                  # Always present (privacy-safe)
    email: str | None = None         # Only if consent given
    consent_email_storage: bool = False  # User opt-in flag
```

---

### 7.2 Name Privacy

**Problem:** Full names may be sensitive in some contexts

**Solution:** Support display names

```python
@dataclass
class Person:
    name: str                        # Full name from git config
    display_name: str | None = None  # Preferred display name (optional)
```

**Usage:**
- Display `display_name` if available, otherwise `name`
- Allow users to set display names via configuration

---

### 7.3 GitHub Privacy

**Problem:** GitHub usernames may link to public profiles

**Solution:** Make GitHub data opt-in

```python
@dataclass
class Person:
    github_username: str | None = None   # Optional
    github_public_profile: bool = False  # Opt-in flag
```

---

### 7.4 Organizational Privacy

**Problem:** Company affiliations may be sensitive

**Solution:** Infer from email domains but allow override

```python
def infer_affiliation(email: str) -> str | None:
    """Infer company from email domain."""
    domain = email.split('@')[1]

    # Skip common email providers
    if domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
        return None

    # Map domains to company names
    company_map = {
        'anthropic.com': 'Anthropic',
        'google.com': 'Google',
        'microsoft.com': 'Microsoft',
    }

    return company_map.get(domain)
```

**Privacy Flag:**
```python
@dataclass
class Person:
    affiliation: str | None = None
    affiliation_public: bool = False  # Opt-in flag
```

---

### 7.5 Privacy Configuration

**Config File:** `~/.mcp-vector-search/privacy.toml`

```toml
[privacy]
# Email storage
hash_emails = true                   # Default: true (hash emails)
store_full_emails = false            # Default: false (only hash)

# Name display
use_display_names = true             # Default: true (use display_name if available)

# GitHub integration
fetch_github_profiles = false        # Default: false (opt-in)
public_github_profiles = false       # Default: false (opt-in)

# Organizational data
infer_company_from_email = true      # Default: true (infer from domain)
public_company_affiliation = false   # Default: false (opt-in)
```

---

## 8. Performance & Scalability

### 8.1 Indexing Performance

**Git Log Extraction:**
- Small repo (1,000 commits): ~1-2 seconds
- Medium repo (10,000 commits): ~10-30 seconds
- Large repo (100,000 commits): ~2-5 minutes

**Git Blame (per file):**
- Small file (100 lines): ~50ms
- Medium file (1,000 lines): ~150ms
- Large file (10,000 lines): ~500ms

**Optimization Strategies:**
- **Incremental updates:** Only process new commits since last index
- **Parallel blame:** Run git blame on multiple files concurrently
- **Caching:** Cache blame results, invalidate on file modification
- **Sampling:** For large repos, blame only critical files (main modules, frequently modified)

```python
async def incremental_git_update(last_indexed_commit: str):
    """Extract only new commits since last index."""
    cmd = [
        "git", "log",
        f"{last_indexed_commit}..HEAD",  # Only new commits
        "--format=%H|%an|%ae|%at|%s",
        "--numstat"
    ]
    # Process output...
```

---

### 8.2 Query Performance

**Simple queries (1-2 hops):**
- "Who authored this function?" → ~5-10ms
- "What did developer X modify?" → ~10-20ms

**Complex queries (3+ hops):**
- "Find co-authors (shared files)" → ~50-100ms
- "Traverse organizational hierarchy" → ~20-50ms

**Optimization:**
- Kuzu uses columnar storage (fast aggregations)
- Indexes on Person.id, CodeEntity.id (automatic primary key indexes)
- Consider adding index on Person.name for name-based queries

---

### 8.3 Storage Overhead

**Person entities:**
- 10 properties × ~50 bytes = 500 bytes per person
- 100 contributors: 50KB
- 1,000 contributors: 500KB

**AUTHORED relationships:**
- 5 properties × ~50 bytes = 250 bytes per relationship
- 10,000 relationships: 2.5MB
- 100,000 relationships: 25MB

**MODIFIED relationships:**
- 6 properties × ~50 bytes = 300 bytes per relationship
- 50,000 commits × 5 files/commit = 250,000 relationships
- Storage: ~75MB

**Total overhead:**
- Small repo (10 contributors, 1K commits): ~5MB
- Medium repo (100 contributors, 10K commits): ~50MB
- Large repo (1000 contributors, 100K commits): ~500MB

**Conclusion:** Storage overhead is minimal (tens to hundreds of MB)

---

### 8.4 Scalability Considerations

**For very large repositories (>100K commits):**

1. **Lazy loading:** Don't index all history upfront
   - Index recent commits (last 1 year)
   - Expand on-demand for older commits

2. **Sampling:** Blame only modified files, not entire codebase
   - Track which files changed in recent commits
   - Only re-run blame on those files

3. **Aggregation:** Store pre-computed statistics
   - Cache `Person.commits_count`, `lines_added`, etc.
   - Update incrementally instead of full recalculation

4. **Partitioning:** Separate KG databases by project
   - One KG per project/repository
   - Federated queries across projects if needed

---

## 9. Summary and Recommendations

### 9.1 Key Takeaways

1. **PROV-O + Schema.org + DOAP** provide comprehensive ontologies for people, projects, and authorship
2. **Git history** is the primary, always-available data source for authorship tracking
3. **Git blame** enables line-level authorship attribution (high value, low cost)
4. **Package metadata** provides project-level author/maintainer information
5. **CODEOWNERS** (if available) gives explicit ownership rules
6. **GitHub API** (optional) enhances with profiles and organizational data
7. **Privacy** is critical: hash emails by default, make sensitive data opt-in

---

### 9.2 Recommended Implementation Priority

**Phase 1 (Week 1-2): Git-Based Authorship [HIGH PRIORITY]**
- Entity types: Person
- Relationships: AUTHORED, MODIFIED
- Data sources: git log, git blame
- Value: Enables "who authored this?" queries
- Effort: Medium (2 weeks)

**Phase 2 (Week 2-3): Project Metadata [MEDIUM PRIORITY]**
- Entity types: Project
- Relationships: MAINTAINS, PART_OF
- Data sources: pyproject.toml, package.json
- Value: Enables project-level queries
- Effort: Low (1 week)

**Phase 3 (Week 3-4): CODEOWNERS Integration [MEDIUM PRIORITY]**
- Relationships: OWNS (Person/Team → CodeEntity)
- Data sources: CODEOWNERS file
- Value: Explicit ownership tracking
- Effort: Low (1 week, conditional on CODEOWNERS availability)

**Phase 4 (Week 5+): GitHub API Enhancement [LOW PRIORITY]**
- Entity enhancements: GitHub profiles, organizations
- Relationships: MEMBER_OF (Person → Organization)
- Data sources: GitHub API
- Value: Rich organizational structure
- Effort: Medium (2 weeks, requires API authentication)

---

### 9.3 Success Metrics

**Coverage:**
- >90% of code files have AUTHORED relationships
- >80% of git authors mapped to Person entities
- >50% of Person entities have contribution statistics

**Query Performance:**
- Simple queries (<50ms): "Who authored function X?"
- Medium queries (<200ms): "What did developer Y work on?"
- Complex queries (<500ms): "Find co-authors"

**Accuracy:**
- Git blame attribution accuracy: >95% (spot-check against actual git history)
- Email hash collision rate: <0.01% (SHA256 ensures uniqueness)

---

### 9.4 Future Enhancements

**Phase 5: Temporal Analysis (Future)**
- Track code ownership over time
- Show contributor trends (new vs. veteran developers)
- Identify code churn hotspots (high modification frequency)

**Phase 6: Team Structure (Future)**
- Extract teams from CODEOWNERS (`@org/team-name`)
- Build organizational hierarchy (Company → Teams → Developers)
- Enable team-based queries ("Which team owns authentication?")

**Phase 7: Contribution Patterns (Future)**
- Identify code review patterns (who reviews whose code?)
- Find knowledge silos (code only touched by one person)
- Suggest reviewers based on authorship history

---

## 10. References

- **PROV-O:** W3C Provenance Ontology - https://www.w3.org/TR/prov-o/
- **Schema.org Person:** https://schema.org/Person
- **Schema.org Organization:** https://schema.org/Organization
- **DOAP:** Description of a Project - https://github.com/edumbill/doap
- **Git Documentation:** https://git-scm.com/docs
- **GitHub API:** https://docs.github.com/en/rest
- **CODEOWNERS Syntax:** https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

---

**End of Research Document**
