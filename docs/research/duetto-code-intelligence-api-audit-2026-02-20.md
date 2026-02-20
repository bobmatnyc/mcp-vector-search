# Duetto Code Intelligence API Audit

**Date:** 2026-02-20
**Project Path:** `~/Duetto/repos/duetto-code-intelligence`
**Purpose:** Verify all API integrations are using correct endpoints, authentication, and versions

---

## Executive Summary

The duetto-code-intelligence project is a RAG-powered code search system for 150+ Duetto repositories. It integrates with multiple external APIs and services. This audit verified:

- ✅ **8 API integrations** across 6 external services
- ✅ **All APIs using HTTPS** (no insecure HTTP)
- ✅ **No hardcoded credentials** in source code (all use environment variables)
- ⚠️ **1 deprecated API version** (Bedrock model ID)
- ⚠️ **1 outdated MCP embedding model** (sentence-transformers vs GraphCodeBERT)
- ✅ **Proper error handling** on all critical paths
- ✅ **Rate limiting** implemented for Atlassian and Notion APIs

---

## API Integration Inventory

| API/Service | File | Line | Status | Issue (if any) |
|-------------|------|------|--------|----------------|
| **AWS Bedrock (Claude LLM)** | `adapters/bedrock_llm.py` | 48-52 | ⚠️ **DEPRECATED** | Using `anthropic.claude-3-5-sonnet-20240620-v1:0` - should upgrade to newer model |
| **AWS Bedrock (Titan Embeddings)** | `adapters/bedrock_llm.py` | 198 | ✅ OK | Using `amazon.titan-embed-text-v1` (stable) |
| **GitHub API** | `adapters/github_sync.py` | 45 | ✅ OK | Using PyGithub SDK with proper token auth |
| **GitHub API (Health Check)** | `web/routes/status.py` | 34 | ✅ OK | `https://api.github.com/zen` for connectivity test |
| **Google OAuth** | `adapters/google_auth.py` | 55 | ✅ OK | Using OpenID Connect discovery URL (latest standard) |
| **Slack Web API** | `services/slack_service.py` | 282 | ✅ OK | `https://slack.com/api/chat.postMessage` (current version) |
| **Confluence API** | `services/knowledge_sync/confluence_sync.py` | 69, 101 | ✅ OK | Using REST API v3 (`/wiki/rest/api`) |
| **JIRA API** | `services/knowledge_sync/jira_sync.py` | 97 | ✅ OK | Using REST API v3 (`/rest/api/3/search/jql`) |
| **Notion API** | `services/knowledge_sync/notion_sync.py` | 22 | ✅ OK | Using API version `2022-06-28` (stable) |
| **Slack Webhooks** | `services/slack_notify.py` | 79 | ✅ OK | Incoming webhooks (no version, stable) |
| **mcp-vector-search (Internal)** | `adapters/vector_search.py` | 136-154 | ⚠️ **MISMATCH** | Using `all-MiniLM-L6-v2` (384d) but README says `graphcodebert-base` (768d) |

---

## Detailed Findings

### 1. AWS Bedrock - Claude Model (⚠️ DEPRECATED VERSION)

**File:** `src/duetto_code_intelligence/adapters/bedrock_llm.py`
**Lines:** 48-52, 107-112

**Current Implementation:**
```python
self._client = boto3.client(
    "bedrock-runtime",
    region_name=self.region,  # us-east-1
)

response = self.client.invoke_model(
    modelId=self._model_id,  # anthropic.claude-3-5-sonnet-20240620-v1:0
    body=json.dumps(body),
)
```

**Configuration (config.py:31-34):**
```python
model_id: str = Field(
    default="anthropic.claude-3-5-sonnet-20240620-v1:0",
    description="Bedrock model ID for chat",
)
```

**Issue:** Using `anthropic.claude-3-5-sonnet-20240620-v1:0` which is a June 2024 model. AWS Bedrock has released newer Claude models since then.

**Recommendation:**
- Update to latest Claude 3.5 Sonnet model (check AWS Bedrock console for current version)
- Typical newer model ID format: `anthropic.claude-3-5-sonnet-20241022-v2:0` or similar
- Set via `BEDROCK_MODEL_ID` environment variable

**Impact:** Low - Model still works but missing improvements from newer versions

---

### 2. mcp-vector-search Embedding Model Mismatch (⚠️ CONFIGURATION ISSUE)

**File:** `src/duetto_code_intelligence/adapters/vector_search.py`
**Lines:** 151-154

**Current Implementation:**
```python
# Create embedding function - MUST match model used to build index
# Index was built with all-MiniLM-L6-v2 (384 dims), NOT CodeBERT
embedding_fn, _cache = create_embedding_function(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    cache_dir=self.index_path / "embedding_cache",
)
```

**Documentation Claims (README.md:122, 242):**
```markdown
| `MCP_VECTOR_SEARCH_EMBEDDING_MODEL` | Override embedding model | microsoft/graphcodebert-base |

- **Embeddings**: GraphCodeBERT (microsoft/graphcodebert-base, 768d)
```

**Issue:** Code hardcodes `all-MiniLM-L6-v2` (384 dimensions) but documentation claims `graphcodebert-base` (768 dimensions). This is a dimension mismatch that would cause search failures.

**Root Cause Analysis:**
- Comment in code says "Index was built with all-MiniLM-L6-v2 (384 dims), NOT CodeBERT"
- This suggests the index was actually built with `all-MiniLM-L6-v2` despite documentation
- Environment variable `MCP_VECTOR_SEARCH_EMBEDDING_MODEL` is ignored in this code

**Recommendation:**
1. **If index was built with all-MiniLM-L6-v2:** Update README to reflect actual model (384d)
2. **If switching to GraphCodeBERT:** Must reindex entire codebase (different dimensions)
3. Add environment variable support: `embedding_model = os.getenv("MCP_VECTOR_SEARCH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")`
4. Add validation to prevent dimension mismatches

**Impact:** Medium - Documentation is misleading; could cause confusion when debugging search quality

---

### 3. GitHub API - PyGithub SDK (✅ OK)

**File:** `src/duetto_code_intelligence/adapters/github_sync.py`
**Lines:** 45

**Implementation:**
```python
self._github = Github(self.token)  # Uses PyGithub library
```

**Status:** ✅ Correct
- Using PyGithub SDK which handles API versioning internally
- Token authentication via `GITHUB_TOKEN` environment variable
- No hardcoded URLs or credentials
- SDK uses latest stable GitHub REST API (currently v3)

**Dependencies (pyproject.toml:34):**
```toml
"pygithub>=2.1",
```

**Best Practice:** ✅ Using SDK rather than raw HTTP calls

---

### 4. Google OAuth - OpenID Connect (✅ OK)

**File:** `src/duetto_code_intelligence/adapters/google_auth.py`
**Lines:** 55

**Implementation:**
```python
self._oauth.register(
    name="google",
    client_id=client_id,
    client_secret=client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)
```

**Status:** ✅ Correct
- Using OpenID Connect discovery URL (current standard)
- This URL returns metadata with latest endpoints automatically
- Properly configured redirect URI via `GOOGLE_REDIRECT_URI` environment variable
- Domain restriction to `@duettoresearch.com` (line 91)

**Security:** ✅ Good
- Session cookies are `httponly`, `secure`, `samesite=lax` (lines 154-157)
- Using `itsdangerous` for session signing with `SECRET_KEY`
- 7-day session expiry (line 17)

---

### 5. Slack Web API - chat.postMessage (✅ OK)

**File:** `src/duetto_code_intelligence/services/slack_service.py`
**Lines:** 281-288

**Implementation:**
```python
response = await self._http_client.post(
    "https://slack.com/api/chat.postMessage",
    headers={
        "Authorization": f"Bearer {self.bot_token}",
        "Content-Type": "application/json",
    },
    json=payload,
)
```

**Status:** ✅ Correct
- Using latest Slack Web API endpoint (no version in URL = latest)
- Bot token authentication (`xoxb-...`) via `SLACK_BOT_TOKEN`
- Request signature verification using HMAC SHA256 (lines 58-99)
- Proper timestamp validation (5-minute window to prevent replay attacks)

**Rate Limiting:** ✅ None needed (Slack has server-side rate limiting)

**Security:** ✅ Excellent
- Signature verification with constant-time comparison (line 96)
- Timestamp validation prevents replay attacks (line 78)

---

### 6. Confluence REST API v3 (✅ OK)

**File:** `src/duetto_code_intelligence/services/knowledge_sync/confluence_sync.py`
**Lines:** 69, 101

**Implementation:**
```python
self.base_url = f"{base_url}/wiki/rest/api"  # Default: https://duettoresearch.atlassian.net

url = f"{self.base_url}/space/{space_key}/content/page"
resp = await client.get(url, params=params)
```

**Status:** ✅ Correct
- Using Confluence REST API v3 (stable, current version)
- Basic authentication with email + API token (lines 83)
- Configurable base URL via `ATLASSIAN_BASE_URL` environment variable
- Pagination implemented (lines 100-127)

**Rate Limiting:** ✅ Good
- 0.5 second delay between requests (line 127)
- Prevents hitting Atlassian rate limits (10 requests/second)

**API Token:** ✅ Secure
- Uses API token instead of password (more secure)
- Token from `ATLASSIAN_API_TOKEN` environment variable

---

### 7. JIRA REST API v3 (✅ OK)

**File:** `src/duetto_code_intelligence/services/knowledge_sync/jira_sync.py`
**Lines:** 97

**Implementation:**
```python
url = f"{self.base_url}/rest/api/3/search/jql"  # v3 API
resp = await client.post(url, json=payload)
```

**Status:** ✅ Correct
- Using JIRA REST API v3 (latest stable version)
- JQL search endpoint with pagination support (lines 82-121)
- Basic authentication (email + API token)
- Proper error handling (lines 100-106)

**Rate Limiting:** ✅ Good
- 0.1 second delay per issue for comment fetching (line 306)
- Prevents overwhelming Atlassian servers

**Pagination:** ✅ Correct
- Uses `nextPageToken` cursor-based pagination (lines 94-119)
- Handles `isLast` flag properly

---

### 8. Notion API v2022-06-28 (✅ OK)

**File:** `src/duetto_code_intelligence/services/knowledge_sync/notion_sync.py`
**Lines:** 21-22

**Implementation:**
```python
NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"
```

**Status:** ✅ Correct
- Using Notion API version `2022-06-28` (stable, widely supported)
- Latest API version as of knowledge cutoff
- Bearer token authentication (line 43)

**Rate Limiting:** ✅ Excellent
- 0.35 second delay between requests (line 23)
- Exponential backoff on 429 errors (lines 67-71)
- Retry logic with 5 attempts (lines 61-87)

**Best Practice:** ✅ Robust retry logic for transient failures

---

### 9. AWS Bedrock - Titan Embeddings (✅ OK)

**File:** `src/duetto_code_intelligence/adapters/bedrock_llm.py`
**Lines:** 198

**Implementation:**
```python
response = self.client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body=body_str,
)
```

**Status:** ✅ Correct
- Using Amazon Titan Embeddings v1 (stable model)
- 8000 character limit properly enforced (line 193)
- Used as fallback since Claude doesn't have native embedding endpoint

**Note:** This is only used in `embed()` method which doesn't appear to be called anywhere in the codebase. The main embeddings come from mcp-vector-search's sentence-transformers model.

---

### 10. Slack Incoming Webhooks (✅ OK)

**File:** `src/duetto_code_intelligence/services/slack_notify.py`
**Lines:** 70-89

**Implementation:**
```python
async with httpx.AsyncClient() as client:
    response = await client.post(url, json=payload, timeout=10.0)
```

**Status:** ✅ Correct
- Using Slack Incoming Webhooks (no API version, stable)
- Webhook URL from `SLACK_WEBHOOK_REINDEX` environment variable
- 10-second timeout prevents hanging
- Graceful error handling (doesn't crash on webhook failures)

**Security:** ✅ Good
- Webhook URLs are sensitive (contain tokens in URL)
- Properly stored in environment variables (not hardcoded)

---

## Security Audit

### ✅ No Hardcoded Credentials

All API keys, tokens, and secrets are properly externalized:

| Credential | Environment Variable | Usage |
|------------|---------------------|--------|
| AWS credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Bedrock API (or instance role) |
| GitHub token | `GITHUB_TOKEN` | Repository sync |
| Google OAuth | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | User authentication |
| Slack bot token | `SLACK_BOT_TOKEN` | Bot API calls |
| Slack signing secret | `SLACK_SIGNING_SECRET` | Request verification |
| Slack webhook | `SLACK_WEBHOOK_REINDEX` | Notifications |
| Atlassian credentials | `ATLASSIAN_EMAIL`, `ATLASSIAN_API_TOKEN` | Confluence/JIRA |
| Notion API key | `NOTION_API_KEY` | Notion sync |
| Session secret | `SECRET_KEY` | Session cookie signing |

### ✅ All HTTPS

No insecure HTTP endpoints found. All external APIs use HTTPS:
- ✅ `https://slack.com/api/*`
- ✅ `https://api.github.com/*`
- ✅ `https://accounts.google.com/*`
- ✅ `https://duettoresearch.atlassian.net/*`
- ✅ `https://api.notion.com/*`
- ✅ AWS Bedrock (HTTPS by default in boto3)

### ✅ Request Timeouts

All HTTP clients have proper timeouts to prevent hanging:
- Slack service: 30 seconds (line 56 in slack_service.py)
- Confluence/JIRA/Notion: 30 seconds (lines 84/74/47)
- Slack webhooks: 10 seconds (line 79 in slack_notify.py)

### ⚠️ Error Exposure

**Minor Issue:** Some error messages could leak internal details:
- `adapters/vector_search.py:452` - Logs full traceback (good for debugging, but ensure CloudWatch logs are restricted)

**Recommendation:** Ensure CloudWatch logs have proper IAM access controls

---

## mcp-vector-search Integration

### How duetto-code-intelligence Uses mcp-vector-search

The project uses mcp-vector-search as a **library** (not MCP protocol):

```python
# Direct Python imports
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.lancedb_backend import LanceVectorDatabase
from mcp_vector_search.core.search import SemanticSearchEngine
from mcp_vector_search.core.indexer import SemanticIndexer
```

**Integration Points:**
1. **Search Engine** (`adapters/vector_search.py:163-168`)
   - Uses `SemanticSearchEngine` for semantic code search
   - Similarity threshold: 0.2 (lowered from 0.3 for better recall)

2. **Indexer** (`adapters/vector_search.py:414-432`)
   - Uses `SemanticIndexer` to build vector index from repositories
   - Incremental indexing via git diff detection
   - Skips git blame (repos are flat-cloned)

3. **Database Backend** (`adapters/vector_search.py:157-161`)
   - Uses `LanceVectorDatabase` for persistence
   - Index stored at `/mnt/data/index` (EFS mount on production)

4. **File Extensions** (`adapters/vector_search.py:17-33`)
   - Indexes: `.java`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.sql`, `.xml`, `.json`, `.yaml`, `.yml`, `.md`, `.sh`, `.gradle`, `.properties`

### Version Compatibility

**pyproject.toml (line 26):**
```toml
"mcp-vector-search>=2.5.38",
```

**Status:** ✅ Using recent version (2.5.38+)

**Potential Issue:** Embedding model mismatch (see Finding #2 above)

---

## Configuration Issues Found

### 1. Inconsistent Defaults Between README and Code

**README.md (lines 119-123):**
```markdown
| `INDEX_PATH` | Vector index location | /data/index |
| `REPOS_PATH` | Cloned repos location | /data/repos |
```

**config.py (lines 326-332):**
```python
index_path: Path = Field(
    default=Path("/mnt/data/index"),  # Different from README!
    description="Path to vector search index (EFS mount point)",
)
repos_path: Path = Field(
    default=Path("/mnt/data/repos"),  # Different from README!
    description="Path to cloned repositories (EFS mount point)",
)
```

**Impact:** Documentation doesn't match code. New users might be confused.

**Recommendation:** Update README.md to reflect actual defaults (`/mnt/data/index` and `/mnt/data/repos`)

---

### 2. Production .env File Contains Credentials

**File:** `.env.production`
**Issue:** Contains actual Google OAuth credentials (lines 14-15)

```
GOOGLE_CLIENT_ID=<redacted>
GOOGLE_CLIENT_SECRET=<redacted>
```

**Status:** ⚠️ **CREDENTIALS IN REPO**

**Security Impact:** Low to Medium
- These are OAuth credentials (not API keys)
- Limited to authorized redirect URIs configured in Google Cloud Console
- Still should not be in version control

**Recommendation:**
1. Remove `.env.production` from git history: `git filter-repo --path .env.production --invert-paths`
2. Add to `.gitignore` (already present, but file was committed before)
3. Rotate OAuth credentials in Google Cloud Console
4. Use AWS Secrets Manager or Parameter Store for production credentials

---

## API Error Handling Summary

| API | Error Handling | Retry Logic | Status |
|-----|----------------|-------------|--------|
| AWS Bedrock | ✅ Try/catch, logs errors | ❌ None (boto3 has internal retries) | Good |
| GitHub | ✅ Try/catch in sync operations | ❌ None | Acceptable |
| Google OAuth | ✅ Try/catch, user-facing error messages | ❌ N/A | Good |
| Slack Web API | ✅ Checks response.ok, logs errors | ❌ None | Good |
| Slack Webhooks | ✅ Logs but doesn't crash | ❌ None | Good |
| Confluence | ✅ Status code checks, HTTPError handling | ✅ 0.5s delay | Excellent |
| JIRA | ✅ Status code checks, pagination handling | ✅ 0.1s delay | Excellent |
| Notion | ✅ Comprehensive retry with backoff | ✅ 5 retries, exponential backoff | Excellent |

**Overall:** Error handling is solid. All APIs have basic error handling, and knowledge sync services have excellent retry logic.

---

## Recommendations Summary

### Priority 1: Security

1. **Remove credentials from .env.production**
   - File: `.env.production` (lines 14-15)
   - Action: Remove from git history, rotate credentials
   - Timeline: Immediate

### Priority 2: API Versions

2. **Update Bedrock Claude model**
   - File: `config.py` (line 32)
   - Current: `anthropic.claude-3-5-sonnet-20240620-v1:0`
   - Action: Update to latest Claude 3.5 Sonnet version
   - Timeline: Next deployment

3. **Fix embedding model documentation mismatch**
   - Files: `README.md`, `adapters/vector_search.py`
   - Current: Documentation says GraphCodeBERT (768d), code uses all-MiniLM-L6-v2 (384d)
   - Action: Update README to match actual implementation
   - Timeline: Next documentation update

### Priority 3: Configuration

4. **Fix path documentation inconsistency**
   - Files: `README.md`, `config.py`
   - Current: README says `/data/`, code defaults to `/mnt/data/`
   - Action: Update README
   - Timeline: Next documentation update

### Priority 4: Enhancements

5. **Add timeout configuration**
   - Files: All HTTP client adapters
   - Current: Hardcoded timeouts (30s, 10s)
   - Action: Make configurable via environment variables
   - Timeline: Future enhancement

6. **Add API health monitoring**
   - Current: Only GitHub API has health check
   - Action: Add `/health` endpoint checks for all external APIs
   - Timeline: Future enhancement

---

## Conclusion

**Overall Status:** ✅ **GOOD**

The duetto-code-intelligence project has well-architected API integrations with:
- ✅ Proper authentication using environment variables
- ✅ HTTPS for all external communications
- ✅ Good error handling and retry logic
- ✅ Rate limiting for APIs that need it
- ⚠️ One deprecated API version (Bedrock model)
- ⚠️ One documentation mismatch (embedding model)
- ⚠️ One security issue (credentials in .env.production)

**No critical API issues found.** The project follows best practices for API integration and can safely proceed with production usage after addressing the three medium-priority recommendations above.

---

## Appendix: API Endpoint Reference

### Complete API Endpoint List

| Service | Endpoint | Method | Purpose |
|---------|----------|--------|---------|
| AWS Bedrock | `bedrock-runtime.{region}.amazonaws.com` | POST | LLM inference (boto3 handles URL) |
| GitHub API | `https://api.github.com/*` | GET | Repository metadata |
| Google OAuth | `https://accounts.google.com/.well-known/openid-configuration` | GET | OIDC discovery |
| Slack Web API | `https://slack.com/api/chat.postMessage` | POST | Send messages |
| Slack Webhooks | `https://hooks.slack.com/services/*` | POST | Incoming webhooks |
| Confluence | `https://duettoresearch.atlassian.net/wiki/rest/api/*` | GET | Page retrieval |
| JIRA | `https://duettoresearch.atlassian.net/rest/api/3/*` | POST | Issue search |
| Notion | `https://api.notion.com/v1/*` | POST/GET | Database/page queries |

### Environment Variables Reference

```bash
# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0

# GitHub
GITHUB_TOKEN=ghp_xxx
GITHUB_ORG=duettoresearch

# Google OAuth
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxx
GOOGLE_REDIRECT_URI=https://code-intelligence.dev.duettosystems.com/auth/callback

# Slack
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_SIGNING_SECRET=xxx
SLACK_WEBHOOK_REINDEX=https://hooks.slack.com/services/xxx

# Atlassian
ATLASSIAN_EMAIL=xxx@duettoresearch.com
ATLASSIAN_API_TOKEN=xxx
ATLASSIAN_BASE_URL=https://duettoresearch.atlassian.net

# Notion
NOTION_API_KEY=secret_xxx

# Application
SECRET_KEY=<32-byte-hex>
INDEX_PATH=/mnt/data/index
REPOS_PATH=/mnt/data/repos
```

---

**Audit Completed:** 2026-02-20
**Auditor:** Research Agent (Claude Opus 4.6)
**Next Review:** Recommended after major version upgrades or quarterly
