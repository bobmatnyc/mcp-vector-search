# Duetto Code Intelligence: Hourly Incremental Indexing Research

**Date**: 2026-02-19
**Project**: /Users/masa/Clients/Duetto/duetto-code-intelligence
**Objective**: Understand current capabilities and requirements for setting up hourly incremental indexing cron job

---

## Executive Summary

Duetto Code Intelligence has **comprehensive incremental indexing infrastructure already built**, but lacks automated scheduling. The system is production-ready for hourly cron jobs with minimal modifications needed.

**Key Findings**:
- ✅ Incremental indexing service exists with git-based change detection
- ✅ GitHub org sync with pull/clone logic is production-ready
- ✅ Knowledge source sync (JIRA/Confluence) with incremental support
- ✅ systemd service running on EC2 with EFS persistence
- ❌ NO existing cron or systemd timer for automated scheduling
- ⚠️ Missing: CLI command to combine sync + incremental index + knowledge sync in one operation

---

## 1. CLI Commands Available

### Current CLI Structure

**Main Entry Point**: `src/duetto_code_intelligence/cli/main.py`

```python
# Available command groups:
- duetto-intel sync      # Repository synchronization
- duetto-intel index     # Code indexing
- duetto-intel chat      # Chat with codebase
- duetto-intel knowledge # Knowledge source sync
- duetto-intel serve     # Start web server
```

### Index Subcommands

**File**: `src/duetto_code_intelligence/cli/index.py`

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `index status` | Show indexing status | - |
| `index reindex` | Full reindex (5-step pipeline) | `--force`, `--reset`, `--skip-atlassian`, `--skip-knowledge`, `--skip-ontology` |
| `index backup` | Backup index to S3 | `--name` |
| `index restore` | Restore index from S3 | `--name` |
| `index list-backups` | List S3 backups | - |
| `index reset` | Clear vector index (destructive) | `--yes` |
| `index atlassian` | Index JIRA/Confluence only | `--jira-days`, `--confluence-days`, `--jira-only`, `--confluence-only` |

**⚠️ Missing**: No `duetto-intel index incremental` command in CLI despite backend support

### Sync Subcommands

**File**: `src/duetto_code_intelligence/cli/sync.py`

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `sync list` | List all repos | `--language` |
| `sync status` | Show sync status | - |
| `sync repo <name>` | Sync single repo | - |
| `sync all` | Sync all repos | `--force`, `--language` |

### Knowledge Subcommands

**File**: `src/duetto_code_intelligence/cli/knowledge.py`

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `knowledge status` | Show knowledge sync status | - |
| `knowledge sync` | Sync knowledge sources | `--source`, `--full`, `--dry-run` |
| `knowledge list <source>` | List synced items | `--limit` |

---

## 2. Repo Syncing: GitHub Organization Integration

### Architecture

**Files**:
- Adapter: `src/duetto_code_intelligence/adapters/github_sync.py`
- Service: `src/duetto_code_intelligence/services/sync_service.py`

### Sync Flow

```python
# GitHubRepoSync adapter workflow:
1. list_repos() → GitHub API → List all repos in org
2. For each repo:
   a. Check if local_path exists (repos_path / repo.name)
   b. If exists → _pull_repo() (git pull with divergence handling)
   c. If not → _clone_repo() (git clone with token auth for private repos)
```

### Pull Logic (Existing Repos)

**Method**: `GitHubRepoSync._pull_repo()`

```python
def _pull_repo(local_path, repo):
    git_repo = Repo(local_path)
    old_sha = git_repo.head.commit.hexsha

    # Pull from origin
    origin.pull()

    # Divergence handling:
    if "diverged" in error:
        origin.fetch()
        git_repo.head.reset(f"origin/{repo.default_branch}",
                           index=True, working_tree=True)

    new_sha = git_repo.head.commit.hexsha
    changed = old_sha != new_sha

    if changed:
        diff = git_repo.commit(old_sha).diff(new_sha)
        files_changed = len(diff)

    return SyncResult(status=SUCCESS if changed else SKIPPED)
```

**Key Features**:
- Automatic divergence resolution (force reset to remote)
- Change detection via commit SHA comparison
- Returns SKIPPED status if no changes
- Counts changed files in diff

### Clone Logic (New Repos)

**Method**: `GitHubRepoSync._clone_repo()`

```python
def _clone_repo(repo, local_path):
    clone_url = repo.url
    if self.token and repo.is_private:
        clone_url = repo.url.replace("https://", f"https://{token}@")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    git_repo = Repo.clone_from(clone_url, local_path)

    return SyncResult(status=SUCCESS, files_changed=file_count)
```

**Key Features**:
- Token injection for private repos
- Creates parent directories automatically
- Counts initial files

### Sync Service Orchestration

**Method**: `SyncService.sync_all()`

```python
async def sync_all(force=False, filter_language=None):
    repos = await self.repo_sync.list_repos()

    if filter_language:
        repos = [r for r in repos if r.language == filter_language]

    for repo in repos:
        result = await self.repo_sync.sync_repo(repo)

        # Track success/failed/skipped counts
        yield result
```

**Returns**: AsyncIterator of SyncResult with:
- `status`: SUCCESS, FAILED, SKIPPED
- `repository`: Repo metadata
- `commit_sha`: Current HEAD
- `files_changed`: Number of files changed (0 if skipped)
- `error`: Error message if failed

### Configuration

**File**: `src/duetto_code_intelligence/config.py`

```python
class GitHubSettings(BaseSettings):
    token: str | None  # GitHub personal access token
    org: str = "duettoresearch"  # GitHub organization

class AppSettings(BaseSettings):
    repos_path: Path = Path("/mnt/data/repos")  # EFS mount
```

**Environment Variables**:
- `GITHUB_TOKEN`: GitHub API token
- `GITHUB_ORG`: Organization name (default: duettoresearch)
- `REPOS_PATH`: Local clone directory (default: /mnt/data/repos)

---

## 3. Knowledge Sources: JIRA and Confluence Integration

### Architecture

**Files**:
- Orchestrator: `src/duetto_code_intelligence/services/knowledge_sync/orchestrator.py`
- Base: `src/duetto_code_intelligence/services/knowledge_sync/base.py`
- Confluence: `src/duetto_code_intelligence/services/knowledge_sync/confluence_sync.py`
- JIRA: `src/duetto_code_intelligence/services/knowledge_sync/jira_sync.py`
- Notion: `src/duetto_code_intelligence/services/knowledge_sync/notion_sync.py`

### Sync Orchestration

**Service**: `KnowledgeSyncService`

```python
class KnowledgeSyncService:
    def __init__(self, data_path, confluence, jira, notion):
        self.data_path = data_path  # /mnt/data/knowledge
        self.sync_state_file = data_path / ".sync_state.json"
        self.sources = {"confluence": ..., "jira": ..., "notion": ...}

    async def sync_source(source, full=False, dry_run=False):
        sync_adapter = self.sources[source]
        output_dir = self.data_path / source.capitalize()

        # Incremental sync by default
        since = None if full else self.get_last_sync(source)

        result = await sync_adapter.sync(
            output_dir=output_dir,
            since=since,  # Only fetch items modified after this date
            dry_run=dry_run
        )

        if not dry_run:
            self.update_sync_state(source, result)

        return result
```

**State Tracking**: `.sync_state.json`

```json
{
  "confluence": {
    "last_sync": "2026-02-19T10:00:00Z",
    "items_synced": 342,
    "status": "success"
  },
  "jira": {
    "last_sync": "2026-02-19T10:05:00Z",
    "items_synced": 1284,
    "status": "success"
  }
}
```

### Confluence Sync

**Configuration**: `ConfluenceSettings`

```python
spaces: list[str] = ["ENGINEERIN", "PROD", "DevOps", "DATA", "DP2026", "SC"]
recency_days: int = 365  # Pull pages modified within N days
version_threshold: int = 10  # Pull pages with N+ versions
```

**Sync Logic**:
1. Queries Confluence API with space filters
2. Fetches pages modified within `recency_days` OR with `version_threshold+` versions
3. Converts pages to markdown files
4. Saves to `/mnt/data/knowledge/Confluence/<space>/<page-title>.md`

### JIRA Sync

**Configuration**: `JiraSettings`

```python
projects: list[str] = ["SRE", "PM", "GC", "BB", "ESS", "ML", "IC", ...]
history_months: int = 6  # Pull tickets from last N months
include_open_older: bool = True  # Also include open tickets older than history_months
```

**Sync Logic**:
1. Queries JIRA API with project filters
2. Fetches tickets from last `history_months` months
3. If `include_open_older=True`, also fetches old open tickets
4. Converts tickets to markdown with comments and metadata
5. Saves to `/mnt/data/knowledge/Jira/<project>/<ticket-key>.md`

### Atlassian API Configuration

**Settings**: `AtlassianSettings`

```python
email: str  # Atlassian account email
api_token: str  # Atlassian API token
base_url: str = "https://duettoresearch.atlassian.net"
```

**Environment Variables**:
- `ATLASSIAN_EMAIL`: Account email
- `ATLASSIAN_API_TOKEN`: API token for auth
- `ATLASSIAN_BASE_URL`: Instance URL

### Knowledge Sync in Reindex Flow

**File**: `src/duetto_code_intelligence/cli/index.py` (line 84-116)

```python
# Step 1: Sync knowledge sources (JIRA/Confluence) → markdown files
if not skip_knowledge:
    from ..services.knowledge_sync import run_knowledge_sync

    knowledge_result = await run_knowledge_sync(full=force)

    if status == "success":
        console.print(f"Knowledge sync complete: {total_synced} items synced")
    elif status == "skipped":
        console.print(f"Knowledge sync skipped: {reason}")
```

**Integration Points**:
- `duetto-intel index reindex` → Runs knowledge sync first (Step 1)
- `duetto-intel knowledge sync` → Standalone knowledge sync
- Both support `--full` flag to ignore `since` timestamp

---

## 4. The Reindex Flow: 5-Step Pipeline

**Command**: `duetto-intel index reindex [--force] [--reset]`

**File**: `src/duetto_code_intelligence/cli/index.py` (line 47-225)

### Full Pipeline Steps

```python
async def reindex():
    # Step 1: Sync knowledge sources (JIRA/Confluence) → markdown
    if not skip_knowledge:
        knowledge_result = await run_knowledge_sync(full=force)
        # Creates markdown files in /mnt/data/knowledge/

    # Step 2: Index code + knowledge markdown files (mcp-vector-search)
    result = await service.reindex(force=force, reset=reset)
    # Calls mcp-vector-search indexer on repos_path + knowledge_path

    # Step 3: Build knowledge graph (ontology)
    if not skip_ontology:
        ontology_result = await ontology_service.build_ontology()
        # Builds entity-relationship graph

    # Step 4: Index Atlassian for vector search
    if not skip_atlassian:
        atlassian_result = await atlassian_indexer.index_all()
        # Pulls fresh JIRA/Confluence content and indexes for search

    # Step 5: Export visualization graph
    graph_result = await export_visualization_graph(repos_path)
    # Generates D3.js graph for web UI
```

### Force vs. Normal Reindex

**`--force` flag**:
- Forces reprocessing of all files even if unchanged
- Passed to `index_project(force_reindex=True)`
- **Does NOT delete existing index** (preserve_existing=True by default)

**`--reset` flag**:
- Clears index directory before reindex
- Fixes "Table already exists" errors
- Destructive: deletes all indexed data first

**Normal reindex** (no flags):
- Skips unchanged files via hash comparison
- Additive: adds new files, updates changed files
- Fastest option for regular updates

### Backend Implementation

**File**: `src/duetto_code_intelligence/services/index_service.py`

```python
async def reindex(force=False, reset=False):
    if reset:
        reset_result = await self.vector_search.reset_index()

    result = await self.vector_search.index_project(force=force)
    return result
```

**File**: `src/duetto_code_intelligence/adapters/vector_search.py` (line 319-452)

```python
async def index_project(force=False, preserve_existing=True):
    # Data protection check
    protection_status = check_data_protection(index_path)
    if protection and force and not preserve_existing:
        # Override to safe mode
        preserve_existing = True

    indexer = SemanticIndexer(
        database=self._database,
        project_root=self.repos_path,
        file_extensions=CODE_FILE_EXTENSIONS,
        skip_blame=True
    )

    chunks_indexed = await indexer.index_project(force_reindex=force)
    return {"status": "complete", "indexed": chunks_indexed}
```

### Incremental Indexing Backend

**File**: `src/duetto_code_intelligence/adapters/vector_search.py` (line 454-585)

**Method**: `index_incremental()`

```python
async def index_incremental():
    incremental_service = get_incremental_index_service()
    repos_to_index = await incremental_service.get_repos_needing_index()

    if not repos_to_index:
        return {"status": "complete", "repos_updated": 0}

    for repo_info in repos_to_index:
        repo_name = repo_info["name"]
        changed_files = repo_info["changed_files"]

        # Index only changed files
        indexer = SemanticIndexer(
            database=self._database,
            project_root=repo_path,
            file_extensions=CODE_FILE_EXTENSIONS
        )
        chunks_indexed = await indexer.index_project(force_reindex=False)

        # Update state with new commit
        incremental_service.set_last_indexed_commit(
            repo_name=repo_name,
            commit_sha=repo_info["current_commit"],
            files_indexed=chunks_indexed
        )

    return {
        "status": "complete",
        "mode": "incremental",
        "repos_updated": len(repos_updated),
        "files_indexed": total_files_indexed
    }
```

**Incremental Index Service**: `src/duetto_code_intelligence/services/incremental_index.py`

```python
class IncrementalIndexService:
    def __init__(self, repos_path, index_path):
        self.state_file = index_path / ".last_indexed"
        # Tracks last-indexed commit per repo

    def get_changed_files(self, repo_path, since_commit):
        # git diff --name-only since_commit HEAD
        result = subprocess.run(["git", "diff", "--name-only", since_commit, "HEAD"])
        return [repo_path / f for f in result.stdout.split("\n")]

    def get_repos_needing_index(self):
        for repo_dir in self.repos_path.iterdir():
            current_commit = self.get_current_commit(repo_dir)
            last_indexed = self.get_last_indexed_commit(repo_dir.name)

            if last_indexed == current_commit:
                continue  # Already indexed

            changed_files = self.get_changed_files(repo_dir, last_indexed)
            yield {
                "name": repo_dir.name,
                "path": repo_dir,
                "changed_files": changed_files,
                "is_new": last_indexed is None
            }
```

**State File**: `/mnt/data/index/.last_indexed`

```json
{
  "backend-api": {
    "last_indexed_commit": "a1b2c3d4",
    "last_indexed_at": "2026-02-19T10:15:00Z",
    "files_indexed": 342
  },
  "frontend-app": {
    "last_indexed_commit": "e5f6g7h8",
    "last_indexed_at": "2026-02-19T10:20:00Z",
    "files_indexed": 127
  }
}
```

---

## 5. Existing systemd/cron Setup

### systemd Service

**File**: `infra/ec2/duetto.service`

**Current Configuration**:

```ini
[Unit]
Description=Duetto Code Intelligence
After=network.target network-online.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/duetto-code-intelligence

# Starts web server only (no indexing)
ExecStart=/home/ubuntu/duetto-code-intelligence/.venv/bin/uvicorn \
    duetto_code_intelligence.web.app:create_app \
    --factory \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 2

Restart=always
RestartSec=5

Environment=INDEX_PATH=/mnt/data/index
Environment=REPOS_PATH=/mnt/data/repos
Environment=GITHUB_ORG=duettoresearch
Environment=AWS_REGION=us-east-1

EnvironmentFile=-/home/ubuntu/duetto-code-intelligence/.env

[Install]
WantedBy=multi-user.target
```

**Key Details**:
- **Current Purpose**: Web server only (port 8000)
- **User**: ubuntu
- **Working Dir**: /home/ubuntu/duetto-code-intelligence
- **Restart Policy**: Always restart on failure
- **Environment**: Loads from `.env` file + inline vars

### Existing Cron/Timer Configuration

**Finding**: ❌ **NO existing cron or systemd timer files**

**Search Results**:
```bash
# Searched for:
find infra -name "*.timer" -o -name "*cron*"
grep -r "cron\|hourly\|schedule" infra/

# Result: No matches found
```

**Conclusion**: Automated scheduling must be built from scratch

### Manual Update Script

**File**: `update-mcp-and-reindex.sh` (root directory)

**Purpose**: Manual full reindex for mcp-vector-search updates

```bash
#!/bin/bash
# Manual script for full reindex after mcp-vector-search upgrades
# NOT suitable for hourly cron (too slow, full reindex)

# 1. Update mcp-vector-search package
uv pip install --upgrade mcp-vector-search

# 2. Check index status
duetto-intel index status

# 3. Full reindex (30-60+ minutes)
duetto-intel index reindex --full

# 4. Verify results
duetto-intel index status
duetto-intel chat search "authentication"
```

**Not Suitable for Hourly Cron Because**:
- Uses `--full` flag (reprocesses all files)
- Takes 30-60+ minutes
- Updates mcp-vector-search package (unnecessary hourly)
- No incremental mode

---

## 6. Configuration Deep Dive

**File**: `src/duetto_code_intelligence/config.py`

### Key Configuration Classes

#### GitHub Settings

```python
class GitHubSettings(BaseSettings):
    token: str | None = None  # GITHUB_TOKEN env var
    org: str = "duettoresearch"  # GITHUB_ORG env var
```

#### Atlassian Settings

```python
class AtlassianSettings(BaseSettings):
    email: str | None = None  # ATLASSIAN_EMAIL
    api_token: str | None = None  # ATLASSIAN_API_TOKEN
    base_url: str = "https://duettoresearch.atlassian.net"
```

#### Knowledge Sync Settings

```python
class KnowledgeSyncSettings(BaseSettings):
    data_path: Path = Path("/mnt/data/knowledge")
    enabled_sources: list[str] = ["confluence", "jira", "notion"]
    sync_on_index: bool = True  # Run knowledge sync before indexing
```

#### App Paths (EFS Mounts)

```python
class AppSettings(BaseSettings):
    index_path: Path = Path("/mnt/data/index")  # Vector index
    repos_path: Path = Path("/mnt/data/repos")  # Git clones
    cache_path: Path = Path("/data/cache")  # Local cache
```

### Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `GITHUB_TOKEN` | None | GitHub API authentication |
| `GITHUB_ORG` | duettoresearch | GitHub organization |
| `REPOS_PATH` | /mnt/data/repos | Repository clones (EFS) |
| `INDEX_PATH` | /mnt/data/index | Vector index (EFS) |
| `ATLASSIAN_EMAIL` | None | Atlassian account |
| `ATLASSIAN_API_TOKEN` | None | Atlassian API token |
| `CONFLUENCE_SPACES` | ["ENGINEERIN", ...] | Confluence spaces to sync |
| `JIRA_PROJECTS` | ["SRE", "PM", ...] | JIRA projects to sync |
| `AWS_REGION` | us-east-1 | AWS region |

---

## 7. What Exists vs. What's Missing

### ✅ What Already Exists

| Component | Status | Location |
|-----------|--------|----------|
| **GitHub Sync** | ✅ Production-ready | `adapters/github_sync.py` |
| **Incremental Index Service** | ✅ Fully implemented | `services/incremental_index.py` |
| **Incremental Index Backend** | ✅ `index_incremental()` exists | `adapters/vector_search.py:454` |
| **Knowledge Sync (JIRA/Confluence)** | ✅ Production-ready | `services/knowledge_sync/` |
| **State Tracking** | ✅ `.last_indexed`, `.sync_state.json` | Persistent on EFS |
| **systemd Service** | ✅ Running web server | `infra/ec2/duetto.service` |
| **EFS Persistent Storage** | ✅ Mounted at /mnt/data | - |
| **CLI Framework** | ✅ Typer-based commands | `cli/*.py` |

### ❌ What's Missing

| Component | Status | Impact |
|-----------|--------|--------|
| **Hourly Cron Job** | ❌ Not configured | No automation |
| **systemd Timer** | ❌ Doesn't exist | No timer-based execution |
| **CLI `index incremental` Command** | ❌ Backend exists, CLI missing | Can't invoke from shell easily |
| **Combined Sync + Index Command** | ❌ Need wrapper | Multi-step manual process |
| **CloudWatch Scheduled Rule** | ❌ Not configured | AWS-native option not used |
| **Logging for Cron** | ❌ Need structured logs | Debugging scheduled jobs |
| **Health Check for Cron** | ❌ Need monitoring | Detect silent failures |

---

## 8. Recommended Approach for Hourly Incremental Indexing

### Option 1: systemd Timer (Recommended for EC2)

**Why systemd Timer**:
- Native integration with systemd logging (journalctl)
- Better than cron for long-running tasks
- Automatic retry on failure
- Can specify dependencies (After=network.target)

**Files to Create**:

#### 1. Timer Unit File

**Path**: `infra/ec2/duetto-incremental-index.timer`

```ini
[Unit]
Description=Duetto Code Intelligence Hourly Incremental Index
Requires=duetto-incremental-index.service

[Timer]
# Run every hour at :05 past the hour
OnCalendar=hourly
# Offset by 5 minutes to avoid startup conflicts
AccuracySec=1min
Persistent=true

[Install]
WantedBy=timers.target
```

#### 2. Service Unit File

**Path**: `infra/ec2/duetto-incremental-index.service`

```ini
[Unit]
Description=Duetto Code Intelligence Incremental Index Job
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/duetto-code-intelligence

# Run incremental index script
ExecStart=/home/ubuntu/duetto-code-intelligence/scripts/incremental-index.sh

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=duetto-incremental-index

# Environment
Environment=INDEX_PATH=/mnt/data/index
Environment=REPOS_PATH=/mnt/data/repos
Environment=GITHUB_ORG=duettoresearch
Environment=AWS_REGION=us-east-1
EnvironmentFile=-/home/ubuntu/duetto-code-intelligence/.env

# Timeout after 30 minutes (safety)
TimeoutStartSec=1800
```

#### 3. Incremental Index Script

**Path**: `scripts/incremental-index.sh`

```bash
#!/bin/bash
# Hourly incremental indexing job for Duetto Code Intelligence
# Syncs repos, indexes changed files, syncs knowledge sources

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[$(date -Iseconds)] Starting incremental index job"

cd "$PROJECT_DIR"
source .venv/bin/activate

# Step 1: Sync all repos (git pull for existing, clone for new)
echo "[$(date -Iseconds)] Step 1: Syncing repositories"
duetto-intel sync all 2>&1 | tee -a /var/log/duetto-incremental-index.log

# Step 2: Incremental knowledge sync (JIRA/Confluence)
echo "[$(date -Iseconds)] Step 2: Syncing knowledge sources (incremental)"
duetto-intel knowledge sync 2>&1 | tee -a /var/log/duetto-incremental-index.log

# Step 3: Incremental code indexing (only changed files)
echo "[$(date -Iseconds)] Step 3: Incremental indexing"
duetto-intel index incremental 2>&1 | tee -a /var/log/duetto-incremental-index.log

# Step 4: Index Atlassian content (last 24 hours)
echo "[$(date -Iseconds)] Step 4: Indexing recent JIRA/Confluence"
duetto-intel index atlassian --jira-days 1 --confluence-days 1 2>&1 | tee -a /var/log/duetto-incremental-index.log

# Step 5: Health check
echo "[$(date -Iseconds)] Step 5: Health check"
duetto-intel index status 2>&1 | tee -a /var/log/duetto-incremental-index.log

echo "[$(date -Iseconds)] Incremental index job complete"
```

**Installation**:

```bash
# Copy files
sudo cp infra/ec2/duetto-incremental-index.timer /etc/systemd/system/
sudo cp infra/ec2/duetto-incremental-index.service /etc/systemd/system/
sudo chmod +x scripts/incremental-index.sh

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable duetto-incremental-index.timer
sudo systemctl start duetto-incremental-index.timer

# Check status
systemctl status duetto-incremental-index.timer
systemctl list-timers duetto-incremental-index.timer

# View logs
journalctl -u duetto-incremental-index -f
```

### Option 2: Traditional Cron (Alternative)

**Crontab Entry**:

```bash
# /etc/cron.d/duetto-incremental-index
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# Run hourly at :05 past the hour
5 * * * * ubuntu /home/ubuntu/duetto-code-intelligence/scripts/incremental-index.sh >> /var/log/duetto-cron.log 2>&1
```

**Pros**: Simple, widely understood
**Cons**: No automatic restart, poor logging integration, no dependencies

### Option 3: AWS EventBridge + Lambda (Cloud-Native)

**Architecture**:
```
EventBridge (hourly) → Lambda Function → SSH to EC2 → Run incremental-index.sh
```

**Pros**: Cloud-native monitoring, retry logic
**Cons**: More complex, requires Lambda setup, SSH key management

---

## 9. Required CLI Command Addition

### New Command: `duetto-intel index incremental`

**File to Modify**: `src/duetto_code_intelligence/cli/index.py`

**Add Command**:

```python
@app.command("incremental")
def index_incremental() -> None:
    """Perform incremental indexing (only changed files).

    Uses git to detect changes per repo and only re-indexes files that have
    changed since the last successful index. Much faster than full reindex.
    """

    async def _incremental() -> None:
        service = get_index_service()

        console.print("[bold]Incremental Indexing[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting changed files...", total=None)
            result = await service.vector_search.index_incremental()
            progress.update(task, completed=True)

        console.print("\n[bold]Incremental Index Complete[/bold]")
        console.print(f"  Repos updated: {result.get('repos_updated', 0)}")
        console.print(f"  Files indexed: {result.get('files_indexed', 0)}")

        if result.get('repos'):
            console.print("\n[bold]Updated Repos:[/bold]")
            for repo in result['repos']:
                console.print(f"  - {repo['name']}: {repo['files_indexed']} files")

    asyncio.run(_incremental())
```

**Backend Already Exists**:
- `VectorSearchAdapter.index_incremental()` in `adapters/vector_search.py:454`
- `IncrementalIndexService` in `services/incremental_index.py`

### New Command: `duetto-intel sync-and-index`

**Convenience Command for Hourly Job**:

```python
@app.command("sync-and-index")
def sync_and_index(
    full: bool = typer.Option(False, "--full", help="Full sync and reindex"),
) -> None:
    """Sync repos and run incremental index in one command."""

    async def _sync_and_index() -> None:
        from ..adapters.github_sync import GitHubRepoSync
        from ..services.sync_service import SyncService
        from ..services.knowledge_sync import run_knowledge_sync

        settings = get_settings()

        # Step 1: Sync repos
        console.print("[bold]Step 1: Syncing repositories[/bold]")
        repo_sync = GitHubRepoSync(
            token=settings.github.token,
            org=settings.github.org,
            repos_path=settings.repos_path,
        )
        sync_service = SyncService(repo_sync)

        success = 0
        async for result in sync_service.sync_all(force=full):
            if result.status.value == "success":
                success += 1
                console.print(f"[green]✓[/green] {result.repository.name}")

        console.print(f"\n[green]{success} repos synced[/green]")

        # Step 2: Sync knowledge
        console.print("\n[bold]Step 2: Syncing knowledge sources[/bold]")
        knowledge_result = await run_knowledge_sync(full=full)
        console.print(f"[green]{knowledge_result.get('total_synced', 0)} items synced[/green]")

        # Step 3: Incremental index
        console.print("\n[bold]Step 3: Incremental indexing[/bold]")
        service = get_index_service()
        index_result = await service.vector_search.index_incremental()
        console.print(f"[green]{index_result.get('files_indexed', 0)} files indexed[/green]")

    asyncio.run(_sync_and_index())
```

---

## 10. Monitoring and Logging Recommendations

### CloudWatch Logs Integration

**Log Group**: `/ec2/duetto-code-intelligence/incremental-index`

**Configuration**: `infra/ec2/cloudwatch-agent-config.json` (add stream)

```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/duetto-incremental-index.log",
            "log_group_name": "/ec2/duetto-code-intelligence/incremental-index",
            "log_stream_name": "{instance_id}",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
```

### CloudWatch Alarms

**Recommended Alarms**:

1. **Incremental Index Failure**
   - Metric: Log filter for "ERROR" in incremental-index logs
   - Threshold: 1+ errors in 1 hour
   - Action: SNS notification to DevOps

2. **Incremental Index Duration**
   - Metric: Custom metric for job duration
   - Threshold: > 20 minutes (should be ~5 min normally)
   - Action: SNS notification (indicates performance issue)

3. **Repos Not Synced**
   - Metric: Log filter for "sync failed" count
   - Threshold: > 5 failed repos in 1 hour
   - Action: SNS notification

### Structured Logging

**Add to incremental-index.sh**:

```bash
# CloudWatch custom metrics
aws cloudwatch put-metric-data \
  --namespace "Duetto/CodeIntelligence" \
  --metric-name "IncrementalIndexDuration" \
  --value $ELAPSED_SECONDS \
  --unit Seconds

aws cloudwatch put-metric-data \
  --namespace "Duetto/CodeIntelligence" \
  --metric-name "FilesIndexed" \
  --value $FILES_INDEXED \
  --unit Count
```

---

## 11. Testing and Validation

### Manual Testing

```bash
# Test incremental index service
cd /home/ubuntu/duetto-code-intelligence
source .venv/bin/activate

# 1. Test repo sync
duetto-intel sync status
duetto-intel sync all

# 2. Test knowledge sync
duetto-intel knowledge status
duetto-intel knowledge sync

# 3. Test incremental index (after adding CLI command)
duetto-intel index incremental

# 4. Test combined command (after adding)
duetto-intel sync-and-index
```

### Timer Testing

```bash
# Test systemd service directly
sudo systemctl start duetto-incremental-index.service
sudo systemctl status duetto-incremental-index.service
journalctl -u duetto-incremental-index -n 100

# Test timer scheduling
sudo systemctl start duetto-incremental-index.timer
systemctl status duetto-incremental-index.timer
systemctl list-timers --all | grep duetto

# Trigger manually (without waiting for schedule)
sudo systemctl start duetto-incremental-index.service
```

### Performance Benchmarks

**Expected Performance** (based on typical GitHub org):

| Operation | Duration | Notes |
|-----------|----------|-------|
| Repo sync (all repos) | 2-5 min | Network-bound, parallel |
| Knowledge sync (incremental) | 1-3 min | API rate limits |
| Incremental index (5 changed repos) | 2-4 min | CPU-bound, depends on changes |
| Atlassian index (1 day) | 1-2 min | API rate limits |
| **Total Hourly Job** | **6-14 min** | Well under 1 hour |

**Full Reindex** (comparison):
- 30-60+ minutes (all files, all repos)
- Only needed weekly or after infrastructure changes

---

## 12. Rollout Plan

### Phase 1: CLI Command Addition (Week 1)

1. Add `duetto-intel index incremental` command
2. Add `duetto-intel sync-and-index` command
3. Test manually on EC2 instance
4. Deploy to production

**Success Criteria**: Commands run successfully, performance acceptable

### Phase 2: systemd Timer Setup (Week 1)

1. Create timer and service unit files
2. Create incremental-index.sh script
3. Install on EC2 instance
4. Enable timer, verify first run
5. Monitor logs for 24 hours

**Success Criteria**: Timer runs hourly, jobs complete successfully

### Phase 3: Monitoring and Alerting (Week 2)

1. Configure CloudWatch log collection
2. Create CloudWatch alarms
3. Set up SNS notifications
4. Create runbook for failure handling

**Success Criteria**: Alerts fire on failures, logs accessible in CloudWatch

### Phase 4: Optimization (Week 2-3)

1. Profile job duration
2. Optimize sync parallelism if needed
3. Tune incremental index thresholds
4. Add caching where beneficial

**Success Criteria**: Job duration < 10 minutes consistently

---

## 13. Specific Files to Modify/Create

### Files to Modify

| File | Changes | Purpose |
|------|---------|---------|
| `src/duetto_code_intelligence/cli/index.py` | Add `incremental` command | Expose incremental indexing in CLI |
| `src/duetto_code_intelligence/cli/index.py` | Add `sync-and-index` command | One-command hourly job |
| `infra/ec2/cloudwatch-agent-config.json` | Add log stream | Collect incremental-index logs |
| `infra/ec2/README.md` | Document timer setup | Operations guide |

### Files to Create

| File | Purpose |
|------|---------|
| `infra/ec2/duetto-incremental-index.timer` | systemd timer unit |
| `infra/ec2/duetto-incremental-index.service` | systemd service unit |
| `scripts/incremental-index.sh` | Hourly indexing script |
| `scripts/test-incremental-index.sh` | Testing script |
| `docs/operations/incremental-indexing.md` | Operations guide |

### File Contents Already Exist

| Component | File | Status |
|-----------|------|--------|
| GitHub sync adapter | `adapters/github_sync.py` | ✅ Production-ready |
| Incremental index service | `services/incremental_index.py` | ✅ Complete |
| Incremental index backend | `adapters/vector_search.py:454` | ✅ Implemented |
| Knowledge sync orchestrator | `services/knowledge_sync/orchestrator.py` | ✅ Production-ready |
| Sync service | `services/sync_service.py` | ✅ Production-ready |

---

## 14. Risk Mitigation

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **EFS mount failure** | Job fails, no indexing | Health check at start of script, alert on mount status |
| **GitHub API rate limit** | Sync fails | Use conditional requests (If-Modified-Since), respect rate limit headers |
| **Long-running job** | Hourly overlap | Set 30-minute timeout, profile and optimize |
| **Index corruption** | Search fails | Daily backup to S3, restore mechanism |
| **Memory leak in indexer** | OOM crash | Monitor memory, restart after N jobs |
| **Atlassian API timeout** | Knowledge sync fails | Retry with backoff, graceful degradation |

### Failure Recovery

**Automatic Recovery**:
- systemd Restart=on-failure (service unit)
- Persistent timer (catches up if missed)
- State tracking (resumes from last success)

**Manual Recovery**:
```bash
# Check logs
journalctl -u duetto-incremental-index -n 200

# Restart timer
sudo systemctl restart duetto-incremental-index.timer

# Force run immediately
sudo systemctl start duetto-incremental-index.service

# Reset index state (last resort)
rm /mnt/data/index/.last_indexed
duetto-intel index reindex --force
```

---

## 15. Cost Analysis

### Compute Costs

**Current Setup**: g4dn.xlarge (on-demand)
- $0.526/hour = ~$380/month

**Hourly Job Impact**:
- Job duration: ~10 minutes/hour
- CPU utilization: High during job, idle otherwise
- **No additional cost** (using existing instance)

**Optimization Opportunity**:
- Use t3.xlarge ($0.1664/hour = ~$120/month) for serving
- Spin up g4dn.xlarge spot instance for nightly full reindex
- **Potential savings**: $260/month (68%)

### EFS Costs

**Current Usage** (estimated):
- Repos: ~50GB
- Index: ~10GB
- Knowledge: ~5GB
- **Total**: ~65GB

**Cost**:
- 65GB × $0.30/GB/month = ~$19.50/month

**Hourly Job Impact**:
- Minimal incremental growth (~100MB/month)
- Knowledge sources add ~5GB/month
- **Additional cost**: ~$2/month

### Network Costs

**GitHub API**: Free (within rate limits)
- 5,000 requests/hour limit
- Conditional requests don't count against limit
- **Cost**: $0

**Atlassian API**: Free (Cloud plans include API access)
- **Cost**: $0

---

## 16. Alternative Approaches Considered

### Approach 1: Webhook-Driven Indexing

**Pros**: Near real-time updates, no polling
**Cons**: Complex setup, webhook endpoint security, GitHub org config

**Verdict**: ⚠️ Good for future, overkill for MVP

### Approach 2: AWS Lambda + EventBridge

**Pros**: Serverless, no EC2 overhead, AWS-native
**Cons**: Cold start latency, SSH key management, Lambda timeout (15 min)

**Verdict**: ⚠️ Consider for multi-instance deployments

### Approach 3: Airflow/Dagster Orchestration

**Pros**: Enterprise-grade workflow orchestration, DAG visualization
**Cons**: Heavy infrastructure, operational overhead

**Verdict**: ❌ Overkill for single-instance deployment

### Approach 4: GitHub Actions (CI/CD)

**Pros**: No server management, runs on GitHub infrastructure
**Cons**: Requires self-hosted runner on EC2, same result as cron

**Verdict**: ❌ Doesn't reduce operational complexity

---

## Conclusion

Duetto Code Intelligence has **excellent foundational infrastructure** for hourly incremental indexing:

✅ **Backend Complete**: Incremental indexing service with git-based change detection
✅ **Sync Ready**: GitHub org sync with pull/clone logic
✅ **Knowledge Sync**: JIRA/Confluence with incremental support
✅ **State Tracking**: Persistent state files on EFS

🔧 **Minimal Work Required**:
1. Add `duetto-intel index incremental` CLI command (30 lines of code)
2. Create systemd timer + service units (2 files, ~40 lines total)
3. Write incremental-index.sh script (~50 lines)
4. Configure monitoring and logging (~1 hour)

⏱️ **Expected Timeline**: 1-2 weeks from start to production monitoring

**Recommended Next Steps**:
1. Implement CLI commands (Phase 1)
2. Test manually on EC2 (Phase 1)
3. Deploy systemd timer (Phase 2)
4. Monitor for 48 hours (Phase 3)
5. Add CloudWatch alarms (Phase 3)
6. Document operations procedures (Phase 4)
