# KG Implementation Map — IA Root Feature (Issue #109)

**Date**: 2026-02-27
**Purpose**: Pre-implementation research for Information Architecture root node feature
**Scope**: Schema, KGBuilder, KnowledgeGraph query class, MCP handlers, CLI

---

## 1. Kuzu Schema

**File**: `src/mcp_vector_search/core/knowledge_graph.py`
**Method**: `_create_schema()` — line 255

### Topic Node Table (lines 338–344)

```sql
CREATE NODE TABLE IF NOT EXISTS Topic (
    id STRING PRIMARY KEY,
    name STRING,
    parent_id STRING
)
```

**Key facts**:
- `id`: Primary key (string)
- `name`: Display name
- `parent_id`: Self-referential — enables hierarchical (tree) taxonomy
- Table is created at schema init time but **never populated** (zero rows in practice — no INSERT anywhere in codebase)

### HAS_TOPIC Relationship Table (lines 511–514)

```sql
CREATE REL TABLE IF NOT EXISTS HAS_TOPIC (
    FROM Document TO Topic,
    MANY_MANY
)
```

**Key facts**:
- Source: `Document` node
- Target: `Topic` node
- `MANY_MANY` cardinality — one Document can have multiple Topics
- No weight or metadata properties
- Created at schema init time but **never has rows inserted** (no populate path exists)

### Document Node Table (lines 317–327)

```sql
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
```

---

## 2. KGBuilder

**File**: `src/mcp_vector_search/core/kg_builder.py`

### `build_from_chunks_sync` (line 277)

Signature:
```python
def build_from_chunks_sync(
    self,
    chunks: list[CodeChunk],
    show_progress: bool = True,
    skip_documents: bool = False,
    progress_tracker: Optional["ProgressTracker"] = None,
) -> dict[str, int]:
```

**Four-phase flow**:

| Phase | Lines | Action |
|-------|-------|--------|
| Phase 1 | 363–454 | Scan chunks — collect `code_entities`, `doc_sections`, `tags`, `relationships` dict |
| Phase 1 cont. | 426–434 | `_build_document_nodes()` — creates `Document` nodes from `doc_sections` |
| Phase 2 | 456–498 | Insert into Kuzu: entities, doc_sections, documents, tags |
| Phase 3 | 504–685 | Validate IDs, insert relationships (CALLS, IMPORTS, etc.) |
| Phase 4 | 692–727 | Git metadata — persons, projects, AUTHORED, PART_OF |

**Document creation** happens at Phase 1 → Phase 2:
- Line 427: `documents, doc_relationships = self._build_document_nodes(doc_sections, text_chunks)`
- Line 487: `stats["doc_nodes"] = self.kg.add_documents_batch_sync(documents)`

**Topic insertion point**: Currently absent. Would fit after Phase 2 (after documents are inserted) or as a new sub-phase inside Phase 2. Topics must exist before HAS_TOPIC relationships can be created.

### `_classify_document` (line 1448)

Signature:
```python
def _classify_document(self, file_path: str) -> str:
```

Three-pass classification strategy: extension/config → exact filename stem → path patterns → filename keyword patterns.

**All 27 return values** (enumerated from passes 1–4):

| Category | Detection Rule | Line |
|----------|---------------|------|
| `configuration` | `.toml/.yaml/.yml/.json/.ini/.cfg/.env` suffix OR `claude.md` name OR `config`/`configuration` in stem | 1473 / 1479 / 1633 |
| `script` | `.sh/.bash/.zsh/.fish` suffix | 1475 |
| `readme` | filename starts with `readme` | 1484 |
| `changelog` | starts with `changelog` or `changes`, or `history`/`release-notes` in stem | 1486 / 1714 |
| `contributing` | starts with `contributing` | 1488 |
| `license` | starts with `license` | 1490 |
| `design` | exact `architecture.md`/`design.md`/`adr.md` OR `/design/`/`/adr/`/`/architecture/` path OR `architecture`/`design`/`summary`/`implementation`/`refactor`/`integration`/`visualization`/`iterator`/`streaming`/`generation`/`backend`/`manager`/`management`/`state`/`phase`/`protection`/`async`/`schema`/`pattern`/`cap`/`limit`/`structure`/`verification`/`nonblocking`/`non-blocking` in stem | 1492 / 1530 / 1691 / 1735 |
| `api_doc` | exact `api.md`/`api-reference.md`/`reference.md` OR `/api/`/`/reference/`/`/tools/` path | 1494 / 1506 / 1510 |
| `spec` | starts with `spec`/`rfc` OR `/spec/`/`/rfc/`/`/prd/` path | 1496 / 1534 |
| `roadmap` | starts with `todo`/`roadmap` | 1498 |
| `guide` | `index.md`/`index.rst`/`index.html` OR `/guides/`/`/guide/`/`/tutorial/`/`/tutorials/`/`/howto/`/`/getting-started/`/`/getting_started/`/`/skills/` path OR `quickstart`/`quick-start`/`quickref`/`quick-ref`/`quickguide`/`checklist`/`standard`/`guide`/`workflow`/`process`/`quality`/`organization`/`testing`/`patterns` in stem OR `/project-template/`/`/templates/`/`template` in stem/path | 1500 / 1522 / 1574 / 1666 / 1711 / 1744 / 1752 |
| `research` | `/research/` in path | 1538 |
| `performance` | `/performance/`/`/benchmarks/` path OR `performance`/`benchmark`/`optimization`/`optimiz` in stem | 1542 / 1643 |
| `deployment` | `/deployment/`/`/deploy/` path OR `deploy`/`deployment`/`versioning`/`ci-cd`/`cicd` in stem | 1546 / 1676 |
| `test_doc` | `/qa/`/`/tests/`/`/test/` path | 1550 |
| `internal` | `/internal/`/`/private/` path OR `sprint`/`kanban`/`backlog`/`codestory`/`code-story` in stem | 1554 / 1695 / 1703 |
| `example` | `/examples/`/`/example/`/`/demos/` path OR `example`/`demo`/`sample` in stem | 1558 / 1653 |
| `report` | `/reports/`/`/report/` path OR `report`/`analysis` in stem | 1562 / 1756 |
| `feature` | `/features/`/`/feature/` path OR starts with `feature-`/`feature_` | 1566 / 1707 |
| `project` | `/projects/`/`/project/` path | 1570 |
| `bugfix` | `bugfix`/`bug-fix`/`bug_fix` in stem OR `fix-`/`fix_` prefix/suffix | 1591 |
| `troubleshooting` | `troubleshoot`/`crash-`/`diagnostics`/`recovery`/`panic`/`defense` in stem | 1602 |
| `faq` | stem is `faq` or starts with `faq-`/`faq_` | 1610 |
| `migration` | `migration`/`migrate` in stem | 1614 |
| `release_notes` | `release` in stem OR starts with `releasing` | 1618 |
| `upgrade_guide` | `upgrade` in stem | 1622 |
| `setup` | `setup`/`install`/`installation` in stem | 1630 |
| `tutorial` | `tutorial` in stem | 1651 |
| `security` | `security`/`vulnerabilit` in stem | 1647 |
| `other` | fallthrough | 1758 |

**Total**: 27 distinct return values (docstring at line 1458–1462 lists 26 — `configuration`, `script`, `feature`, `project`, `tutorial` were added after the docstring was written).

---

## 3. KnowledgeGraph Query Class

**File**: `src/mcp_vector_search/core/knowledge_graph.py`

### `get_document_ontology` (line 4241)

Signature:
```python
async def get_document_ontology(
    self, category: str | None = None
) -> dict[str, Any]:
```

**Returns**:
```python
{
    "categories": {
        "<doc_category>": [
            {
                "id": str,
                "file_path": str,
                "title": str,
                "word_count": int,
                "section_count": int,
                "sections": [{"name": str, "level": int, "line": int}],
                "cross_references": [{"file_path": str, "title": str}],
                "tags": [str],
            }
        ]
    },
    "total_documents": int,
    "total_sections": int,
    "total_cross_references": int,
}
```

**Four sub-queries inside the method**:
1. `MATCH (d:Document) RETURN d.id, d.file_path, d.title, d.doc_category, d.word_count, d.section_count ORDER BY d.doc_category, d.title`
2. `MATCH (d:Document)-[:CONTAINS_SECTION]->(s:DocSection) RETURN d.id, s.name, s.level, s.line_start ORDER BY d.id, s.line_start`
3. `MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document) RETURN d1.id, d2.file_path, d2.title`
4. `MATCH (d:Document)-[:CONTAINS_SECTION]->(s:DocSection)-[:HAS_TAG]->(t:Tag) RETURN d.id, COLLECT(DISTINCT t.name)`

### `get_visualization_data` (line 3433)

Signature:
```python
async def get_visualization_data(self) -> dict[str, Any]:
```

**Node-query pattern** — each node type uses direct `self.conn.execute()`:
```python
# Code entities
entity_result = self.conn.execute("MATCH (e:CodeEntity) RETURN e.id, e.name, e.entity_type, e.file_path")
# DocSection nodes
doc_result = self.conn.execute("MATCH (d:DocSection) RETURN d.id, d.name, d.file_path")
# Tag nodes
tag_result = self.conn.execute("MATCH (t:Tag) RETURN t.id, t.name")
# Person nodes
person_result = self.conn.execute("MATCH (p:Person) RETURN p.id, p.name")
# Project nodes
project_result = self.conn.execute("MATCH (p:Project) RETURN p.id, p.name")
```

**Pattern for adding new node type**: add a `self.conn.execute("MATCH (x:NewType) RETURN ...")` block, iterate with `while result.has_next()`, append to `entities` list with `{"id": ..., "name": ..., "type": "new_type", "file_path": ""}`.

### Adding new query methods — pattern

New query methods follow this pattern:
1. Check `if not self._initialized: await self.initialize()`
2. Call `self._execute_query(cypher_str, params_dict)`
3. Iterate with `while result.has_next(): row = result.get_next()`
4. Return structured `dict[str, Any]`

### `get_stats_sync` (line 4005) and `get_stats` (line 4094)

Both count node tables individually (`CodeEntity`, `DocSection`, `Tag`, `Document`) — **Topic is not counted** in either method. The `total_entities` sum (line 4044) excludes Topic rows.

---

## 4. MCP Handlers

**File**: `src/mcp_vector_search/mcp/kg_handlers.py`

### `handle_kg_ontology` (line 197)

```python
async def handle_kg_ontology(self, args: dict[str, Any]) -> CallToolResult:
    category = args.get("category")
    kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
    kg = KnowledgeGraph(kg_path)
    await kg.initialize()
    ontology = await kg.get_document_ontology(category=category)
    await kg.close()
    result = {"status": "success", "ontology": ontology}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2))],
        isError=False,
    )
```

**Pattern for adding a new handler**:
1. Add `async def handle_kg_<name>(self, args: dict[str, Any]) -> CallToolResult:` in `KGHandlers`
2. Standard boilerplate: `kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"`, `kg = KnowledgeGraph(kg_path)`, `await kg.initialize()`, call kg method, `await kg.close()`
3. Return `CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))], isError=False)`

### MCP server dispatch (server.py lines 304–312)

```python
elif tool_name == "kg_build":
    return await self._kg_handlers.handle_kg_build(args)
elif tool_name == "kg_stats":
    return await self._kg_handlers.handle_kg_stats(args)
elif tool_name == "kg_query":
    return await self._kg_handlers.handle_kg_query(args)
elif tool_name == "kg_ontology":
    return await self._kg_handlers.handle_kg_ontology(args)
```

**File**: `src/mcp_vector_search/mcp/server.py` (lines 304–312)

To add a new handler, add a new `elif tool_name == "kg_<name>": return await self._kg_handlers.handle_kg_<name>(args)` before the `else` clause (line 328).

### Tool schema pattern (tool_schemas.py)

**File**: `src/mcp_vector_search/mcp/tool_schemas.py`

Pattern (example from `_get_kg_ontology_schema` at line 985):
```python
def _get_kg_ontology_schema() -> Tool:
    """Get kg_ontology tool schema."""
    return Tool(
        name="kg_ontology",
        description="...",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "...",
                    "enum": [...],
                },
            },
        },
    )
```

1. Add a private `_get_kg_<name>_schema()` function
2. Add the call to `get_tool_schemas()` list (lines 12–35)

---

## 5. CLI

**File**: `src/mcp_vector_search/cli/commands/kg.py`

### `kg ontology` command (line 841)

```python
@kg_app.command("ontology")
def kg_ontology(
    category: Annotated[str | None, typer.Option("--category", "-c", ...)] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", ...)] = False,
    project_root: Path = typer.Option(".", ...),
):
```

Runs `asyncio.run(_ontology())` — the inner `async def _ontology()` pattern is used for all KG commands that need async Kuzu access.

### `rich.tree.Tree` usage (lines 960–1018)

```python
from rich.tree import Tree

cat_tree = Tree(
    f"{icon} [bold green]{cat_name}[/bold green] [dim]({len(docs)} documents)[/dim]"
)
for doc in docs:
    doc_node = cat_tree.add(" ".join(doc_label_parts))   # child node
    doc_node.add(f"[dim]Tags:[/dim] ...")                 # grandchild node
    doc_node.add(f"[dim]→[/dim] [yellow]{ref_str}[/yellow]")
    # verbose: sections as grandchildren
console.print(cat_tree)
```

**Pattern**: `Tree(root_label)` → `.add(child_label)` returns a sub-Tree → `.add()` on that for grandchildren → `console.print(tree)`.

### Adding a new subcommand/flag

Pattern for all existing commands:
```python
@kg_app.command("new-command")
def kg_new_command(
    some_arg: str = typer.Argument(..., help="..."),
    some_option: bool = typer.Option(False, "--flag", "-f", help="..."),
    project_root: Path = typer.Option(".", help="Project root directory", exists=True, file_okay=False),
):
    """Docstring shown in --help."""
    project_root = project_root.resolve()
    async def _inner():
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()
        try:
            result = await kg.some_new_method(...)
        finally:
            await kg.close()
        console.print(...)  # rich output
    asyncio.run(_inner())
```

Existing commands for reference:
- `build` — line 546
- `index` — line 594
- `query` — line 642
- `stats` — line 700
- `ontology` — line 841
- `status` — line 1024
- `calls` — line 1311
- `inherits` — line 1372
- `visualize` — line 1433

---

## 6. Topic Table — Current State: Zero Rows

**Confirmed**: The `Topic` node table and `HAS_TOPIC` relationship table are defined in the schema (`_create_schema()` at lines 338–347 and 511–519) but **no code path ever inserts rows into them**.

Evidence:
- `kg_builder.py`: grep for `Topic` → zero matches (no `INSERT INTO Topic`, no `MERGE`, no `add_topics_batch` call)
- `knowledge_graph.py`: `Topic` appears only in:
  - Schema `CREATE TABLE` DDL (lines 338, 511)
  - Relationship type dispatch in `add_relationships_batch_sync` / async equivalent (lines 1992–1994, 2469–2472) — these are routing stubs that handle `HAS_TOPIC` if it were present, but never called
- `get_stats_sync` (line 4005): does **not** query `Topic` count; `total_entities` sum excludes it

**Impact**: Any `MATCH (t:Topic)` query returns zero rows today. HAS_TOPIC edges likewise have zero rows.

---

## 7. Key File Summary

| File | Role |
|------|------|
| `src/mcp_vector_search/core/knowledge_graph.py` | Schema DDL (`_create_schema` L255), batch insert methods (`add_documents_batch_sync` L1722, `add_tags_batch_sync` L1880), query methods (`get_document_ontology` L4241, `get_visualization_data` L3433, `get_stats_sync` L4005) |
| `src/mcp_vector_search/core/kg_builder.py` | Build orchestration (`build_from_chunks_sync` L277), document classification (`_classify_document` L1448), document node assembly (`_build_document_nodes` L1760) |
| `src/mcp_vector_search/mcp/kg_handlers.py` | MCP handler class (`handle_kg_ontology` L197, `handle_kg_build` L28, `handle_kg_stats` L149, `handle_kg_query` L244) |
| `src/mcp_vector_search/mcp/server.py` | Tool dispatch `elif` chain (kg_ tools: L304–312) |
| `src/mcp_vector_search/mcp/tool_schemas.py` | Tool JSON schemas (`_get_kg_ontology_schema` L985, `get_tool_schemas` list L12–35) |
| `src/mcp_vector_search/cli/commands/kg.py` | CLI commands (`kg_ontology` L841, `kg_app` Typer instance L24, `rich.tree.Tree` usage L960) |
