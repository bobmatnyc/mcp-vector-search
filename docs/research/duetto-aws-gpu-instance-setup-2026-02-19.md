# AWS GPU Instance Setup Research - Duetto Code Intelligence

**Research Date**: 2026-02-19
**Project**: duetto-code-intelligence
**Instance**: i-02d8caca52b2e209b (g4dn.xlarge GPU)
**Purpose**: Understand how the AWS GPU instance was originally provisioned for rebuilding

---

## Executive Summary

The AWS GPU instance for duetto-code-intelligence is a **g4dn.xlarge** EC2 instance in us-east-1 used for vector indexing workloads. The instance was provisioned using either cloud-init or a manual setup script, with repositories cloned from the **duettoresearch** GitHub organization using GitHub CLI (gh) with token authentication.

---

## 1. GitHub Repository Configuration

### GitHub Organization
- **Organization**: `duettoresearch`
- **Source**: Configured via environment variable `GITHUB_ORG=duettoresearch`

### Repository Cloning Method

**Primary Method: GitHub CLI (gh)**
```bash
# Authentication via token
export GH_TOKEN="<github_token>"
# OR
export GITHUB_TOKEN="<github_token>"

# Clone command (used in startup script)
gh repo clone duettoresearch/<repo_name> <dest_path> -- --depth 1 --quiet
```

**Fallback Method: curl + GitHub API**
```bash
# If gh CLI fails, curl is used as fallback
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/orgs/duettoresearch/repos?per_page=200"
```

### Repository Discovery
```bash
# List all repos in org (used in startup.sh)
gh repo list duettoresearch --json name --limit 200 -q '.[].name'
```

### Authentication Details
- **Token Type**: GitHub Personal Access Token (PAT) or GitHub App token
- **Environment Variables**: `GH_TOKEN` or `GITHUB_TOKEN`
- **Storage**: AWS Secrets Manager (`duetto-code-intel/secrets`)
- **Permissions Required**:
  - `repo` scope (read access to repositories)
  - Organization member access

---

## 2. duetto-code-intelligence Installation

### Installation Method: Local Editable Install

```bash
# Clone the application repository
cd /home/ubuntu
git clone https://github.com/duettoresearch/duetto-code-intelligence.git

# Create Python virtual environment
cd duetto-code-intelligence
python3.11 -m venv .venv

# Install in editable mode with dev dependencies
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install -e ".[dev]"
```

**NOT installed from PyPI** - The application is installed as a local editable package from the git repository.

### Key Dependency
From `pyproject.toml`:
```toml
dependencies = [
    "mcp-vector-search>=2.5.19",
    # ... other dependencies
]
```

The `mcp-vector-search` package **IS installed from PyPI** (not from git).

### CLI Command Registration
After installation, the following CLI is available:
```bash
duetto-intel  # Main CLI entry point
```

Commands include:
- `duetto-intel sync all` - Sync repositories
- `duetto-intel index reindex` - Reindex codebase
- `duetto-intel serve` - Start web server

---

## 3. Complete Setup Process

### Option A: Cloud-Init (Automated)

The instance can be launched with cloud-init configuration that automatically:
1. Installs system packages
2. Mounts EFS
3. Clones application repository
4. Sets up Python environment
5. Fetches secrets from AWS Secrets Manager
6. Configures systemd service
7. Configures nginx with SSL

**File**: `infra/ec2/cloud-init.yaml` (referenced but not examined in this research)

### Option B: Manual Setup Script

**Script**: `/Users/masa/Clients/Duetto/duetto-code-intelligence/infra/ec2/setup.sh`

#### Step-by-Step Process

**Step 1: Install System Packages**
```bash
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    curl \
    jq \
    htop \
    nfs-common \
    awscli \
    unzip \
    amazon-efs-utils
```

**Step 2: Mount EFS**
```bash
# EFS ID and mount point
EFS_ID="fs-07fdcd4a3f1ba7482"
MOUNT_POINT="/mnt/data"

# Add to fstab
echo "$EFS_ID:/ $MOUNT_POINT efs _netdev,tls,iam 0 0" >> /etc/fstab

# Mount EFS
mount -t efs -o tls,iam "$EFS_ID":/ "$MOUNT_POINT"

# Create directories
mkdir -p "$MOUNT_POINT/repos" "$MOUNT_POINT/index"
chown -R ubuntu:ubuntu "$MOUNT_POINT"
```

**Step 3: Clone Application Repository**
```bash
REPO_URL="https://github.com/duettoresearch/duetto-code-intelligence.git"
APP_DIR="/home/ubuntu/duetto-code-intelligence"

cd /home/ubuntu
sudo -u ubuntu git clone "$REPO_URL"
```

**Step 4: Setup Python Environment**
```bash
cd "$APP_DIR"

# Create virtual environment
sudo -u ubuntu python3.11 -m venv .venv

# Install dependencies
sudo -u ubuntu .venv/bin/pip install --upgrade pip wheel
sudo -u ubuntu .venv/bin/pip install -e ".[dev]"
```

**Step 5: Fetch Secrets from AWS Secrets Manager**
```bash
SECRET_ID="duetto-code-intel/secrets"
AWS_REGION="us-east-1"

# Fetch secrets
SECRETS=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_ID" \
    --region "$AWS_REGION" \
    --query SecretString \
    --output text)

# Create .env file
cat > "$APP_DIR/.env" << EOF
# Application paths
INDEX_PATH=/mnt/data/index
REPOS_PATH=/mnt/data/repos

# GitHub
GITHUB_ORG=duettoresearch

# AWS
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0

# Secrets from AWS Secrets Manager
EOF

# Append secrets
echo "$SECRETS" | jq -r 'to_entries[] | "\(.key)=\(.value)"' >> "$APP_DIR/.env"

chown ubuntu:ubuntu "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"
```

**Step 6: Install Systemd Service**
```bash
cp "$APP_DIR/infra/ec2/duetto.service" /etc/systemd/system/duetto.service
systemctl daemon-reload
systemctl enable duetto
systemctl start duetto
```

**Step 7: Configure Nginx**
```bash
cp "$APP_DIR/infra/ec2/nginx.conf" /etc/nginx/sites-available/duetto
ln -sf /etc/nginx/sites-available/duetto /etc/nginx/sites-enabled/duetto
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
```

**Step 8: Configure SSL (Optional)**
```bash
DOMAIN="code.duettoresearch.com"
certbot --nginx -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    --email infra@duettoresearch.com \
    --redirect
```

---

## 4. Repository Sync Process

### Startup Script Behavior

**Script**: `/Users/masa/Clients/Duetto/duetto-code-intelligence/scripts/startup.sh`

The startup script implements **INCREMENTAL sync** for fast deployments:

1. **Start app FIRST** (for health checks)
2. **Sync repos in background**
3. **Trigger incremental indexing** after sync completes

### Sync Algorithm

```bash
# For each repository in GitHub org:

if [ -d "$dest_path/.git" ]; then
    # INCREMENTAL: Existing repo - just fetch and reset
    cd "$dest_path"
    git fetch origin --depth 1 --quiet
    git reset --hard "origin/${default_branch}" --quiet
else
    # INITIAL: New repo - clone
    gh repo clone "duettoresearch/${repo_name}" "$dest_path" -- --depth 1 --quiet
fi
```

### Parallel Cloning

```bash
MAX_PARALLEL_CLONES="${MAX_PARALLEL_CLONES:-4}"

# Clone/sync repos in parallel (up to 4 concurrent)
while IFS= read -r repo_name; do
    # Wait if too many parallel jobs
    while [ $RUNNING -ge $MAX_PARALLEL_CLONES ]; do
        wait -n 2>/dev/null || true
        RUNNING=$((RUNNING - 1))
    done

    # Sync in background
    sync_repo_incremental "$repo_name" &
    RUNNING=$((RUNNING + 1))
done
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_ORG` | `duettoresearch` | GitHub organization name |
| `REPOS_PATH` | `/mnt/data/repos` | Repository clone destination |
| `INDEX_PATH` | `/mnt/data/index` | Vector search index path |
| `MAX_PARALLEL_CLONES` | `4` | Concurrent clone/sync jobs |
| `SKIP_CLONE` | `false` | Skip repo sync (use cached) |
| `SKIP_AUTO_INDEX` | `false` | Skip auto-indexing after sync |

---

## 5. Current Instance Configuration

### Instance Details

| Property | Value |
|----------|-------|
| **Instance ID** | `i-02d8caca52b2e209b` |
| **Instance Type** | `g4dn.xlarge` (Tesla T4 GPU) |
| **Region** | `us-east-1` |
| **OS** | Ubuntu 22.04 LTS |
| **OS User** | `ubuntu` |
| **Public IP** | `100.29.9.224` (changes on restart) |

### Storage Configuration

| Path | Purpose | Storage Type |
|------|---------|--------------|
| `/mnt/data/` | EFS mount point | EFS (fs-07fdcd4a3f1ba7482) |
| `/mnt/data/repos/` | Cloned repositories | EFS shared storage |
| `/mnt/data/index/` | Vector search index | EFS shared storage |
| `/home/ubuntu/duetto-code-intelligence/` | Application code | Local EBS |
| `/home/ubuntu/duetto-code-intelligence/.venv/` | Python virtual environment | Local EBS |

### Connection Methods

**Primary Method: AWS SSM (VERIFIED WORKING)**
```bash
# Interactive shell session
aws ssm start-session --target i-02d8caca52b2e209b --region us-east-1

# Run single command
COMMAND_ID=$(aws ssm send-command \
  --instance-ids i-02d8caca52b2e209b \
  --region us-east-1 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["YOUR_COMMAND_HERE"]' \
  --output text --query 'Command.CommandId')
```

**Note**: EC2 Instance Connect and direct SSH do NOT work for this instance. Use AWS SSM only.

---

## 6. Rebuild Instance Steps

### Complete Rebuild Procedure

To rebuild the AWS GPU instance from scratch:

#### Prerequisites
1. AWS CLI configured with appropriate credentials
2. IAM permissions for:
   - EC2 management
   - EFS access
   - Secrets Manager read access
   - SSM Session Manager
3. GitHub token with repo access to duettoresearch organization

#### Rebuild Steps

**Step 1: Launch New EC2 Instance**
```bash
# Launch g4dn.xlarge instance in us-east-1
# - AMI: Ubuntu 22.04 LTS
# - Instance type: g4dn.xlarge
# - VPC: Same VPC as EFS
# - Security group: Allow NFS (2049) to EFS security group
# - IAM role: With Bedrock, S3, Secrets Manager, EFS, CloudWatch permissions
# - Storage: 50GB gp3
```

**Step 2: Connect via SSM**
```bash
# Get instance ID (example: i-NEW-INSTANCE-ID)
aws ssm start-session --target i-NEW-INSTANCE-ID --region us-east-1
```

**Step 3: Clone Setup Repository**
```bash
cd /tmp
git clone https://github.com/duettoresearch/duetto-code-intelligence.git
cd duetto-code-intelligence/infra/ec2
```

**Step 4: Run Setup Script**
```bash
# Run with sudo
sudo ./setup.sh
```

This will:
- Install all system packages
- Mount EFS at /mnt/data
- Clone application repository to /home/ubuntu/duetto-code-intelligence
- Create Python virtual environment
- Install dependencies
- Fetch secrets from AWS Secrets Manager
- Configure systemd service
- Configure nginx

**Step 5: Verify Installation**
```bash
# Check service status
sudo systemctl status duetto

# Check EFS mount
df -h /mnt/data

# Check application installation
source /home/ubuntu/duetto-code-intelligence/.venv/bin/activate
duetto-intel --version
```

**Step 6: Sync Repositories (if needed)**
```bash
# The startup script syncs repos automatically on service start
# To manually trigger sync:
cd /home/ubuntu/duetto-code-intelligence
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
duetto-intel sync all
```

**Step 7: Trigger Indexing**
```bash
# Full reindex
source /home/ubuntu/duetto-code-intelligence/.venv/bin/activate
export $(grep -v '^#' .env | xargs)
export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=25
duetto-intel index reindex --full
```

---

## 7. Environment Variables Required

### From AWS Secrets Manager (`duetto-code-intel/secrets`)

- `GITHUB_TOKEN` - GitHub API token (for repository sync)
- `GH_TOKEN` - GitHub CLI token (used by gh CLI)
- `SECRET_KEY` - Application secret key
- `GOOGLE_CLIENT_ID` - OAuth client ID
- `GOOGLE_CLIENT_SECRET` - OAuth client secret

### Configured in .env (Generated by Setup Script)

```bash
# Application paths
INDEX_PATH=/mnt/data/index
REPOS_PATH=/mnt/data/repos

# GitHub
GITHUB_ORG=duettoresearch

# AWS
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0

# Additional secrets appended from Secrets Manager
# GITHUB_TOKEN=...
# GH_TOKEN=...
# SECRET_KEY=...
# etc.
```

---

## 8. Key Files and Locations

### Setup Files
| File | Purpose |
|------|---------|
| `infra/ec2/setup.sh` | Manual setup script (alternative to cloud-init) |
| `infra/ec2/cloud-init.yaml` | Automated cloud-init configuration |
| `infra/ec2/duetto.service` | Systemd service unit file |
| `infra/ec2/nginx.conf` | Nginx configuration with SSL and SSE support |
| `infra/ec2/cloudwatch-agent-config.json` | CloudWatch agent configuration |
| `infra/ec2/logrotate.conf` | Log rotation configuration |

### Deployment Scripts
| Script | Purpose |
|--------|---------|
| `scripts/startup.sh` | Application startup script (runs on service start) |
| `scripts/sync-repos-to-s3.sh` | Sync repositories to S3 (for backup/caching) |
| `scripts/build-and-deploy.sh` | Build and deploy to ECS (deprecated, not used for GPU instance) |
| `scripts/local-dev.sh` | Local development startup script |

### Configuration Files
| File | Purpose |
|------|---------|
| `.env.example` | Template for environment variables |
| `pyproject.toml` | Python package configuration and dependencies |
| `docker-compose.yml` | Docker Compose configuration (not used on GPU instance) |

---

## 9. Differences from PyPI mcp-vector-search

The duetto-code-intelligence setup is **DIFFERENT** from a standard mcp-vector-search installation:

### Standard mcp-vector-search (from PyPI)
```bash
pip install mcp-vector-search
mcp-vector-search index /path/to/repos
```

### Duetto Code Intelligence (Custom Application)
```bash
git clone https://github.com/duettoresearch/duetto-code-intelligence.git
cd duetto-code-intelligence
pip install -e ".[dev]"  # Installs duetto-intel CLI
duetto-intel sync all    # Custom sync logic with GitHub org
duetto-intel index reindex  # Wraps mcp-vector-search with custom config
```

**Key Differences:**
1. **Custom CLI**: `duetto-intel` (not `mcp-vector-search`)
2. **GitHub Integration**: Automatic org-wide repository sync
3. **Web UI**: FastAPI web application with chat interface
4. **AWS Integration**: Bedrock for chat, S3 for backups
5. **OAuth**: Google OAuth for authentication
6. **Dependencies**: mcp-vector-search is a dependency, not the main application

---

## 10. Repository Provisioning Methods

### Method 1: Automatic Sync via Startup Script (Recommended)

The application automatically syncs repositories on startup:

```bash
# Triggered by systemd service start
# See: scripts/startup.sh

# Environment variables:
GITHUB_ORG=duettoresearch
REPOS_PATH=/mnt/data/repos
GH_TOKEN=<token_from_secrets_manager>

# Behavior:
# 1. Fetches list of all repos in org (gh repo list)
# 2. Clones new repos (gh repo clone)
# 3. Updates existing repos (git fetch + reset)
# 4. Runs in parallel (4 concurrent jobs)
# 5. Triggers incremental indexing after sync
```

### Method 2: Manual CLI Sync

```bash
cd /home/ubuntu/duetto-code-intelligence
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

# Sync all repos
duetto-intel sync all

# Sync specific repo
duetto-intel sync repo <repo_name>

# Check sync status
duetto-intel sync status
```

### Method 3: S3 Backup/Restore (For Faster Setup)

```bash
# Script: scripts/sync-repos-to-s3.sh
# This script can be used to:
# 1. Clone all repos locally
# 2. Upload to S3 bucket (duetto-code-intel-repos)
# 3. Restore from S3 on new instance (faster than cloning)

# To restore from S3:
aws s3 sync s3://duetto-code-intel-repos/repos/ /mnt/data/repos/ \
    --region us-east-1
```

---

## 11. Indexing Configuration

### Current Version
```bash
# From GPU_INSTANCE_CONNECTION.md
mcp-vector-search version: 2.5.19 (deadlock fix)
```

### Upgrade Command
```bash
# On the instance
source /home/ubuntu/duetto-code-intelligence/.venv/bin/activate
pip install --no-cache-dir --upgrade mcp-vector-search
```

### Indexing Commands

```bash
# Set environment variables
source /home/ubuntu/duetto-code-intelligence/.venv/bin/activate
export $(grep -v '^#' .env | xargs)
export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=25

# Incremental reindex (only changed files)
duetto-intel index reindex

# Full reindex (rebuild entire index)
duetto-intel index reindex --full

# Check index status
duetto-intel index status
```

### Indexing Logs
```bash
# Current indexing log location
/tmp/reindex-*.log

# Example:
tail -f /tmp/reindex-20260218-193114-gpu-v2.5.19.log
```

---

## 12. Troubleshooting Common Issues

### Issue: Cannot Connect to Instance

**Solution**: Use AWS SSM (not SSH or EC2 Instance Connect)
```bash
aws ssm start-session --target i-02d8caca52b2e209b --region us-east-1
```

### Issue: EFS Not Mounted

**Solution**: Check and remount EFS
```bash
# Check mount status
mount | grep efs
df -h /mnt/data

# Remount if needed
sudo mount -t efs -o tls,iam fs-07fdcd4a3f1ba7482:/ /mnt/data
```

### Issue: GitHub Authentication Failed

**Solution**: Check tokens in AWS Secrets Manager
```bash
# Verify secrets are configured
aws secretsmanager get-secret-value \
    --secret-id duetto-code-intel/secrets \
    --region us-east-1 \
    --query SecretString \
    --output text | jq -r '.GITHUB_TOKEN'

# Check gh CLI authentication
gh auth status
```

### Issue: Service Won't Start

**Solution**: Check logs and environment
```bash
# Check service status
sudo systemctl status duetto

# View logs
sudo journalctl -u duetto -n 100 --no-pager

# Check .env file
cat /home/ubuntu/duetto-code-intelligence/.env

# Test manually
cd /home/ubuntu/duetto-code-intelligence
source .venv/bin/activate
python -c "from duetto_code_intelligence.web.app import create_app; print('OK')"
```

---

## Summary

The AWS GPU instance for duetto-code-intelligence is provisioned using:

1. **GitHub Org**: `duettoresearch`
2. **Clone Method**: GitHub CLI (`gh`) with token authentication (GH_TOKEN or GITHUB_TOKEN)
3. **Installation**: Local editable install (`pip install -e ".[dev]"`) from git repository
4. **Setup Scripts**: `infra/ec2/setup.sh` or cloud-init automation
5. **Provisioning**: Manual setup script or cloud-init for automated deployment

The complete rebuild process involves:
- Launching g4dn.xlarge instance
- Running setup.sh script
- Mounting EFS
- Installing application from git
- Fetching secrets from AWS Secrets Manager
- Syncing repositories from GitHub org
- Triggering vector indexing

All repository sync is handled automatically by the startup script using GitHub CLI with token authentication against the duettoresearch organization.

---

## Research Metadata

- **Researcher**: Claude Code (Research Agent)
- **Research Date**: 2026-02-19
- **Project Path**: /Users/masa/Clients/Duetto/duetto-code-intelligence
- **Key Files Examined**:
  - GPU_INSTANCE_CONNECTION.md
  - infra/ec2/setup.sh
  - scripts/startup.sh
  - scripts/sync-repos-to-s3.sh
  - pyproject.toml
  - .env.example
  - README.md
  - infra/ec2/README.md
  - infra/README.md
  - docs/LOCAL_DEVELOPMENT.md

**Research Classification**: Informational (no immediate action required)
