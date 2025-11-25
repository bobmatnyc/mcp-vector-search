# Homebrew Tap Token Investigation

**Date:** 2025-11-25
**Context:** Release v0.12.9 - Phase 5 (Homebrew tap update) was skipped
**Issue:** HOMEBREW_TAP_TOKEN missing - Need to determine if required and how to obtain

---

## Executive Summary

**Finding:** The `HOMEBREW_TAP_TOKEN` requirement is **NEW** - it was introduced in commit `b8a0513` on **November 19, 2025** as part of a new automated Homebrew Formula update workflow.

**Key Points:**
- ✅ Token is **required** for automated updates to the Homebrew tap repository
- ✅ This is a **GitHub Personal Access Token (PAT)**, not a Homebrew-specific token
- ✅ The script can run without it (dry-run mode), but **cannot push changes** without the token
- ✅ This is a **recent addition** - you've never needed it before because the automation is brand new
- ✅ Manual updates to the tap repository can still be done using existing git credentials

**Impact:** The automated Homebrew Formula update will be skipped in releases until the token is configured.

---

## Analysis

### 1. When Was This Introduced?

The HOMEBREW_TAP_TOKEN requirement was introduced in **commit b8a0513** (November 19, 2025):

```
commit b8a0513efa0aa1ed1713fde2de5fdbe4562b0e13
Author: Bob Matsuoka <bob@matsuoka.com>
Date:   Wed Nov 19 17:17:11 2025 -0500

feat: add automated Homebrew Formula update workflow
```

This commit added:
- `/Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py` - Python automation script
- `/Users/masa/Projects/mcp-vector-search/.github/workflows/update-homebrew.yml` - GitHub Actions workflow
- Makefile targets: `homebrew-update`, `homebrew-update-dry-run`, `homebrew-test`
- Comprehensive documentation (5+ guides)

### 2. What Is HOMEBREW_TAP_TOKEN?

**Type:** GitHub Personal Access Token (PAT)

**Purpose:** Authenticate to GitHub when pushing changes to the tap repository (`bobmatnyc/homebrew-mcp-vector-search`)

**Used By:**
1. **Script:** `/Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py`
   - Line 113: `self.github_token = os.getenv("HOMEBREW_TAP_TOKEN")`
   - Line 485-487: Warns if not set: "HOMEBREW_TAP_TOKEN not set - push may fail"
   - Line 491-503: Configures git to use token for authentication

2. **GitHub Actions:** `/Users/masa/Projects/mcp-vector-search/.github/workflows/update-homebrew.yml`
   - Line 45: `HOMEBREW_TAP_TOKEN: ${{ secrets.HOMEBREW_TAP_TOKEN }}`
   - Used to authenticate when workflow runs automatically

3. **Makefile:** Lines 488-502
   - `homebrew-update-dry-run`: Checks token is set
   - `homebrew-update`: Requires token to push changes

### 3. Is Token Actually Required?

**For Automation:** YES - Required to push changes to the tap repository.

**For Manual Updates:** NO - You can update the tap repository manually using your existing git credentials.

**Behavior Without Token:**
- Script will **warn** but continue execution
- Can fetch PyPI info and update formula file locally
- **Cannot push** changes to remote repository
- Exit code 5 (authentication error) if push attempted without token

**Code Evidence:**
```python
# Line 485-487 in update_homebrew_formula.py
if not self.github_token:
    self.log("HOMEBREW_TAP_TOKEN not set - push may fail", "warning")
    self.log("Set HOMEBREW_TAP_TOKEN environment variable for authentication", "info")
```

### 4. How the Token Works

**Authentication Flow:**
1. Script reads token from environment: `os.getenv("HOMEBREW_TAP_TOKEN")`
2. Extracts repo path from URL: `bobmatnyc/homebrew-mcp-vector-search`
3. Creates authenticated URL: `https://{token}@github.com/{repo_path}.git`
4. Configures git remote: `git remote set-url origin <authenticated-url>`
5. Pushes changes: `git push` (now authenticated)

**Security Features:**
- Token read from environment variable (not hardcoded)
- Never logged or displayed in output
- Only requires `repo` scope (minimal permissions)
- Used only for push operations

### 5. What Repository Does It Access?

**Target Repository:** `https://github.com/bobmatnyc/homebrew-mcp-vector-search.git`

**What It Updates:**
- Formula file: `Formula/mcp-vector-search.rb` (or `mcp-vector-search.rb` in root)
- Updates version number and SHA256 hash
- Commits with message: `chore: update formula to {version}`
- Pushes to `origin/main`

**Local Clone Location:** `~/.homebrew_tap_update/homebrew-mcp-vector-search`

### 6. Current State of Your Environment

**Token Status:** NOT SET
```bash
$ env | grep -i homebrew
# Only shows Homebrew installation paths, no HOMEBREW_TAP_TOKEN
```

**Makefile Checks:**
- Line 488-491: `homebrew-update-dry-run` fails if token not set
- Line 497-500: `homebrew-update` exits with error if token not set
- Line 569-573: `full-release` skips Homebrew update with warning if token not set

### 7. Why You Haven't Needed This Before

**Timeline:**
- **Before Nov 19, 2025:** No automation existed - manual updates only
- **Nov 19, 2025:** Automation introduced in commit b8a0513
- **Current:** You're seeing the requirement for the first time in release v0.12.9

**Previous Workflow:**
- Manually update tap repository using git credentials
- No automation script
- No GitHub Actions workflow
- No token needed

**Current Workflow (Automated):**
- Script fetches version/hash from PyPI automatically
- Updates formula file automatically
- Commits and pushes automatically
- **Requires token for unattended push operations**

---

## Solution: How to Obtain and Configure Token

### Step 1: Create GitHub Personal Access Token

1. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/tokens/new
   - Or: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Configure Token:**
   - **Name:** `homebrew-tap-updater` (or any descriptive name)
   - **Expiration:** 90 days (recommended) or custom
   - **Scopes:** Select **ONLY** `repo` (full repository access)
     - ✅ `repo` - Full control of private repositories
     - ❌ Leave all other scopes unchecked

3. **Generate Token:**
   - Click "Generate token"
   - **IMPORTANT:** Copy token immediately (shown only once)
   - Token format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Set Environment Variable Locally

**Temporary (Current Session Only):**
```bash
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"
```

**Permanent (Recommended):**
```bash
# Add to shell configuration
echo 'export HOMEBREW_TAP_TOKEN="ghp_your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

**Verify:**
```bash
# Check token is set (should show token value)
echo $HOMEBREW_TAP_TOKEN

# Test with dry-run (should not error about missing token)
./scripts/update_homebrew_formula.py --dry-run
```

### Step 3: Configure GitHub Actions Secret

For automated workflows in GitHub Actions:

1. **Go to Repository Settings:**
   - Visit: https://github.com/bobmatnyc/mcp-vector-search/settings/secrets/actions

2. **Add Secret:**
   - Click "New repository secret"
   - **Name:** `HOMEBREW_TAP_TOKEN` (must match exactly)
   - **Value:** Paste your token (`ghp_...`)
   - Click "Add secret"

3. **Verify:**
   - Secret should appear in the list
   - Workflow will use it automatically in future runs

---

## Alternative Approaches

### Option 1: Use Existing Git Credentials (Manual)

You can continue updating the tap manually without the token:

```bash
# Clone tap repository
git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git ~/homebrew-tap

# Update formula manually
cd ~/homebrew-tap
# Edit Formula/mcp-vector-search.rb
# Update version and sha256

# Commit and push (uses your existing git credentials)
git commit -am "chore: update formula to 0.12.9"
git push
```

**Pros:**
- No token needed
- Uses existing GitHub authentication (SSH keys or credential manager)

**Cons:**
- Manual process
- No automation
- Prone to errors

### Option 2: Skip Homebrew Update in Releases

The Makefile already handles this gracefully:

```makefile
# Line 569-573 in Makefile
@if [ -n "$(HOMEBREW_TAP_TOKEN)" ]; then \
    $(MAKE) homebrew-update; \
else \
    echo "$(YELLOW)⚠️  Skipping Homebrew update (HOMEBREW_TAP_TOKEN not set)$(RESET)"; \
fi
```

**Pros:**
- No configuration needed
- Release workflow continues
- Homebrew update skipped with warning

**Cons:**
- Tap not automatically updated
- Users don't get latest version via Homebrew
- Manual update still needed

### Option 3: Token-Free Automation (Not Recommended)

Theoretically possible but not recommended:

```bash
# Use SSH authentication instead of HTTPS
export HOMEBREW_TAP_REPO="git@github.com:bobmatnyc/homebrew-mcp-vector-search.git"
```

**Cons:**
- Script not designed for SSH authentication
- Requires code modifications
- Less reliable in CI/CD environments
- Not supported by current implementation

---

## Recommendation

**Primary Solution:** Set up the HOMEBREW_TAP_TOKEN

**Reasoning:**
1. **Automation Benefit:** One-command releases without manual tap updates
2. **CI/CD Integration:** GitHub Actions can update tap automatically on releases
3. **Error Reduction:** Eliminates manual formula updates (version, SHA256)
4. **Future-Proof:** Supports the new automation workflow as intended
5. **Low Effort:** 5-minute one-time setup

**Steps:**
1. Create GitHub PAT with `repo` scope
2. Set `HOMEBREW_TAP_TOKEN` in `~/.zshrc`
3. Add `HOMEBREW_TAP_TOKEN` to GitHub Actions secrets
4. Test with: `make homebrew-update-dry-run`
5. Next release will automatically update Homebrew tap

---

## Testing Instructions

### Test 1: Verify Token Configuration

```bash
# Check token is set
echo $HOMEBREW_TAP_TOKEN

# Should output: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# If empty, token not configured
```

### Test 2: Dry-Run Update

```bash
# Test without making changes
./scripts/update_homebrew_formula.py --dry-run --verbose

# Expected output:
# ✓ Found version: 0.12.9
# ✓ SHA256 hash verified successfully
# [DRY RUN] Would update formula file
# [DRY RUN] Would commit and push changes
```

### Test 3: Actual Update

```bash
# Update to latest PyPI version
make homebrew-update

# Expected output:
# ✓ Formula file updated
# ✓ Changes committed
# ✓ Changes pushed successfully
```

### Test 4: GitHub Actions Integration

```bash
# Trigger workflow manually or via release
# Check: https://github.com/bobmatnyc/mcp-vector-search/actions

# Workflow should:
# 1. Extract version from tag
# 2. Update Homebrew formula
# 3. Push to tap repository
# 4. Create success notification
```

---

## Security Considerations

### Token Permissions

**Required Scope:**
- ✅ `repo` - Full control of repositories

**NOT Required:**
- ❌ `admin:org` - Not needed
- ❌ `workflow` - Not needed
- ❌ `write:packages` - Not needed
- ❌ Any other scopes

**Why `repo` scope:**
- Needed to push commits to tap repository
- Cannot use read-only scope (read:org) for pushes
- Minimal required permission for automation

### Token Storage

**Secure Storage:**
- ✅ Environment variable in shell config (~/.zshrc)
- ✅ GitHub Actions secrets (encrypted)
- ✅ Password manager (1Password, LastPass, etc.)

**NEVER:**
- ❌ Commit token to git repository
- ❌ Share token in public channels
- ❌ Use token in scripts that are committed
- ❌ Log token value in console output

### Token Rotation

**Best Practices:**
1. Set expiration to 90 days
2. Rotate token before expiration
3. Update environment variable and GitHub secret
4. Revoke old token after rotation
5. Monitor token usage in GitHub Settings → Developer settings

### Token Compromise

**If Token Leaked:**
1. Immediately revoke in GitHub Settings
2. Create new token with different name
3. Update environment variable and GitHub secret
4. Review repository access logs
5. Check for unauthorized pushes

---

## Documentation References

**Comprehensive Guides:**
1. `/Users/masa/Projects/mcp-vector-search/scripts/README_HOMEBREW_FORMULA.md` - Full documentation (536 lines)
2. `/Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_QUICKSTART.md` - Quick reference (119 lines)
3. `/Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_WORKFLOW.md` - Workflow guide (463 lines)
4. `/Users/masa/Projects/mcp-vector-search/docs/HOMEBREW_INTEGRATION.md` - Integration guide (467 lines)
5. `/Users/masa/Projects/mcp-vector-search/docs/HOMEBREW_QUICKSTART.md` - Quick start (157 lines)

**Key Sections:**
- Token setup: `scripts/README_HOMEBREW_FORMULA.md` lines 42-75
- Environment variables: Line 113-119
- Authentication: Lines 484-503
- Troubleshooting: Lines 315-392

---

## Summary

**Question:** Is HOMEBREW_TAP_TOKEN actually required?

**Answer:** Yes, for automated updates. No, for manual updates.

**Why You Haven't Needed It Before:**
- Automation was introduced on **November 19, 2025** (6 days ago)
- This is the **first release** using the new automation
- Previously, updates were done manually using existing git credentials

**What To Do:**
1. **Immediate:** Set up GitHub Personal Access Token with `repo` scope
2. **Configure:** Add to `~/.zshrc` and GitHub Actions secrets
3. **Test:** Run `make homebrew-update-dry-run`
4. **Verify:** Next release will automatically update Homebrew tap

**Impact If Not Configured:**
- ⚠️ Homebrew updates skipped during releases (warning shown)
- ⚠️ Users can't install latest version via Homebrew immediately
- ⚠️ Manual tap updates still needed
- ✅ Release workflow continues normally (non-blocking)

---

## Next Steps

1. **Create Token:** https://github.com/settings/tokens/new (repo scope)
2. **Set Locally:** `echo 'export HOMEBREW_TAP_TOKEN="ghp_..."' >> ~/.zshrc`
3. **Set GitHub:** https://github.com/bobmatnyc/mcp-vector-search/settings/secrets/actions
4. **Test:** `make homebrew-update-dry-run`
5. **Release:** Next release will auto-update Homebrew tap

**Estimated Time:** 5-10 minutes for complete setup

---

## Files Analyzed

1. `/Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py` - 719 lines
2. `/Users/masa/Projects/mcp-vector-search/.github/workflows/update-homebrew.yml` - 122 lines
3. `/Users/masa/Projects/mcp-vector-search/Makefile` - Lines 485-514 (Homebrew targets)
4. `/Users/masa/Projects/mcp-vector-search/scripts/README_HOMEBREW_FORMULA.md` - 536 lines
5. `/Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_QUICKSTART.md` - 119 lines
6. Git history: Commits b8a0513, c617e43, f97c21a

**Research Methodology:**
- Code analysis of automation scripts
- Git history examination
- Documentation review
- Environment variable inspection
- Makefile workflow analysis

**Memory Usage:** Efficient - Used Grep and Glob for discovery, strategic file reading
