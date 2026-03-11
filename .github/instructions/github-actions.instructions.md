---
name: GitHub Actions
description: Authoring, reviewing, and linting GitHub Actions workflow files and helper scripts in this repository.
applyTo: ".github/workflows/**"
---

# GitHub Actions -- datas-world/.github

Reference: <https://docs.github.com/en/actions/writing-workflows>

---

## Mandatory: run actionlint before every commit and during every review

`actionlint` is the authoritative linter for all workflow files.  Run it on
every file you touch **before committing** and verify its output **during code
review**.

```bash
# Lint all workflows (run from the repository root)
actionlint .github/workflows/*.yml

# Lint a single file
actionlint .github/workflows/<filename>.yml
```

The `actionlint` CI workflow runs automatically on every push or pull request
that touches `.github/workflows/**` and is a **mandatory status check** -- it
must pass before merging.

### Local installation

```bash
go install github.com/rhysd/actionlint/cmd/actionlint@latest
```

`shellcheck` is pre-installed on the `ubuntu-slim` runner and picked up by
`actionlint` automatically to validate all inline `run:` shell scripts.

---

## Runner selection

| Runner | When to use |
|---|---|
| `ubuntu-slim` | API calls, `gh`/`jq`/`curl` tasks, lightweight scripts -- anything that comfortably fits in 15 minutes on 1 vCPU |
| `ubuntu-latest` | `npm ci`, TypeScript builds, CodeQL analysis -- toolchain-heavy jobs where the 15-minute limit on slim would be risky |

Self-hosted runners (`tmk-wkst-4` Ubuntu workstation, `tuk-nb-3` Fedora notebook)
are labelled by the startup hook in `runner-hooks/label-sync.sh`.  The Ubuntu
workstation carries the `ubuntu-slim` label and accepts those jobs automatically.

### ubuntu-slim -- pre-installed tools (image 20260120, Ubuntu 24.04)

These tools are **guaranteed to be present**.  Never add install-guard steps
(`type -p <tool> || apt-get install ...`) for any of them.

| Tool | Version |
|---|---|
| GitHub CLI (`gh`) | 2.85.0 |
| `jq` | 1.7.1 |
| `yq` | 4.50.1 |
| `shellcheck` | 0.9.0 |
| `curl` | 8.5.0 |
| `git` | 2.52.0 |
| Node.js | 24.13.0 |
| `npm` | 11.6.2 |

Full list: <https://github.com/actions/runner-images/blob/main/images/ubuntu-slim/ubuntu-slim-Readme.md>

---

## SHA-pinned actions

Pin every `uses:` reference to a **full 40-character commit SHA**; add the
human-readable version as a comment.

```yaml
# ? correct
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2

# ? mutable -- never use tags or branch names
- uses: actions/checkout@v4
- uses: actions/checkout@main
```

To resolve the commit SHA for a given tag:

```bash
gh api /repos/actions/checkout/git/ref/tags/v6.0.2 --jq '.object.sha'
# If the tag points to a tag object rather than a commit, dereference it:
gh api /repos/actions/checkout/git/tags/<tag-sha> --jq '.object.sha'
```

---

## Explicit permissions (minimum viable)

Every workflow must declare `permissions: {}` at the top level; each job opts
in to only what it needs.

```yaml
permissions: {}   # deny all by default at workflow level

jobs:
  my-job:
    permissions:
      contents: read   # only what this job actually requires
```

Common permission sets:

| Use case | Permissions |
|---|---|
| API read-only | `contents: read` |
| Write issue or PR | `issues: write` / `pull-requests: write` |
| Publish npm package | `contents: write`, `id-token: write` |
| CodeQL | `contents: read`, `actions: read`, `security-events: write` |
| GitHub Pages | `contents: read`, `pages: write`, `id-token: write` |

---

## Reusable workflow anatomy

Every reusable workflow must:
1. Use `on: workflow_call`
2. Declare `permissions: {}` at the workflow level
3. Explicitly declare all `inputs:` and `secrets:`
4. Have each job declare its own `permissions:`

```yaml
name: My Reusable Workflow

on:
  workflow_call:
    inputs:
      my_input:
        description: Description of the input.
        required: false
        type: string
        default: 'default-value'
    secrets:
      MY_SECRET:
        description: Description of the secret.
        required: false

permissions: {}

jobs:
  my-job:
    name: Job display name
    runs-on: ubuntu-slim
    permissions:
      contents: read
    steps:
      - ...
```

---

## Job `name` = status-check context string

The `name:` field is the exact string GitHub uses in required status checks.
Renaming a job silently breaks branch protection rules.  When renaming, update
`PUT .../branches/main/protection` in every repository that calls this workflow.

---

## GitHub Apps -- inventory and principle of least privilege

Three GitHub Apps are registered in the `datas-world` organisation.  Each has only
the permissions it needs and nothing more.

| App | Secrets | Permissions | Purpose |
|---|---|---|---|
| `datas-world-sync-bot` | `SYNC_APP_ID`, `SYNC_APP_PRIVATE_KEY` | Contents: write ? Pull requests: write ? Workflows: write | BetaHuhn file sync ? creates PRs in target repos |
| `datas-world-project-assign` | `PROJECT_APP_ID`, `PROJECT_APP_PRIVATE_KEY` | Org Projects: write ? Issues: read ? Pull requests: read | Adds issues/PRs to the `.github` Governance project |
| `datas-world-label-sync` | `LABEL_SYNC_APP_ID`, `LABEL_SYNC_APP_PRIVATE_KEY` | Issues: write ? Org Members: read | Syncs canonical labels from `labels.json` to all repos |

**Creating a new App** (one-time setup): use the manifest-flow script at
`~/.copilot/session-state/*/files/create-label-sync-app.py` which opens a single
browser form, captures the callback, stores secrets automatically, and prints
installation instructions.

**Never** add permissions to an existing App to handle a new use-case -- create a
dedicated App instead.  `GITHUB_TOKEN` cannot write to org-level Projects V2; always
mint a short-lived installation token via `actions/create-github-app-token`.

```yaml
# Pattern -- mint token then pass to the step that needs it:
- uses: actions/create-github-app-token@29824e69f54612133e76f7eaac726eef6c875baf  # v2.2.1
  id: app-token
  with:
    app-id: ${{ vars.LABEL_SYNC_APP_ID }}   # replace with correct App secret
    private-key: ${{ secrets.LABEL_SYNC_APP_PRIVATE_KEY }}
    owner: datas-world
```

---

## GitHub App token for Projects V2

`GITHUB_TOKEN` cannot write to org-level Projects V2.  Use
`actions/create-github-app-token` with `owner: datas-world`:

```yaml
- uses: actions/create-github-app-token@29824e69f54612133e76f7eaac726eef6c875baf  # v2.2.1
  id: app-token
  with:
    app-id: ${{ vars.PROJECT_APP_ID }}
    private-key: ${{ secrets.PROJECT_APP_PRIVATE_KEY }}
    owner: datas-world
```

---

## `milestone-assign.yml` and `sync-topics-node.yml` use TypeScript actions

These two workflows delegate to private org TypeScript actions rather than embedding
shell scripts. The actions run as `node20`; Node.js 24.13.0 on `ubuntu-slim` satisfies
this.

| Workflow | Action | Version |
|---|---|---|
| `milestone-assign.yml` | `datas-world/action-milestone-assign` | `0.1.0` |
| `sync-topics-node.yml` | `datas-world/action-sync-topics-node` | `0.1.0` |

To upgrade: build an orphan release commit in the action repo (only `action.yml` +
`dist/`), push as the new tag, then update the SHA-pinned `uses:` line in the relevant
workflow file. Do not embed inline shell logic for either workflow.

---

## Runner hooks

Self-hosted runners carry the `ubuntu-slim` label (and optionally `ubuntu-latest`) by
running `runner-hooks/label-sync.sh` as an `ExecStartPre` systemd step wired in by
`runner-hooks/install.sh`.  Both scripts live in the repository root.  See
[AGENTS.md](../../AGENTS.md) -- "Runner strategy" -- for the full installation story.

---

## Testing `run:` scripts without side effects

`actionlint` and `shellcheck` catch syntax and structural problems but cannot verify
runtime API logic.  The options below test the actual shell code without making
real changes to GitHub issues, projects, or pull requests.

### Option 1 -- `bats` unit tests with a mock `gh` (recommended for script logic)

Extract complex `run:` blocks to standalone `.sh` files under `scripts/`.  Write
[`bats`](https://bats-core.readthedocs.io/en/stable/) tests that put a mock `gh`
binary earlier on `$PATH`.  The mock returns canned JSON for read calls and
prints-but-does-not-execute write calls.

```bash
# scripts/mock/gh -- put this on $PATH in bats tests
#!/usr/bin/env bash
echo "gh called with: $*" >&2   # log the call for assertions
# Return canned data for GET-like subcommands:
case "$*" in
  *"api /repos/"*"milestones"*) cat tests/fixtures/milestones.json ;;
  *) echo "[]" ;;
esac
```

Advantages: fast, no Docker, no network, no GitHub credentials required.

### Option 2 -- `act` locally (full workflow simulation)

[`act`](https://github.com/nektos/act) runs workflows locally in Docker without consuming
GitHub-hosted runner minutes.  Pass a read-only or empty `GITHUB_TOKEN` so that any
accidental write call receives a `403` rather than modifying data.

```bash
# Dry-run a workflow: prints the execution plan without running steps
act --dryrun -W .github/workflows/pr-lint.yml

# Actually run with a read-only token -- writes will 403, reads succeed
act -W .github/workflows/sync-topics-node.yml \
    -s GITHUB_TOKEN="$(gh auth token)" \
    -e tests/events/push.json
```

Use `--dryrun` for structural validation; use a read-only token for logic testing.

### Option 3 -- sandbox repository

Create a dedicated `datas-world/.github-sandbox` repository with throwaway issues
and PRs.  Trigger workflow caller stubs against it.  Real GitHub API behaviour,
no risk to production data.

### Option 4 -- `DRY_RUN` env var in scripts

For scripts that make a mix of reads and writes, honour a `DRY_RUN` environment
variable: when set, log the `gh api` write call instead of executing it.

```bash
gh_write() {
  if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "[DRY_RUN] gh $*" >&2
  else
    gh "$@"
  fi
}
```

Set `DRY_RUN=1` in CI jobs triggered manually (via `workflow_dispatch`) during
development to validate the logic path without touching data.

---

## Canonical org labels (`.github/labels.json`)

All repositories in the `datas-world` org share a canonical label set defined in
[`.github/labels.json`](../../.github/labels.json).  The
[`sync-labels.yml`](sync-labels.yml) workflow applies these labels to every org repo
using [`srealmoreno/label-sync-action@v2`](https://github.com/srealmoreno/label-sync-action)
using an App installation token (calling `/installation/repositories`) -- no manual repo list needed.  It also supports
`dry-run` preview via `workflow_dispatch`.

**Rules:**
1. Every label used by any workflow -- PR labels, issue labels, tracking labels -- **must
   be in `labels.json`**.  Add it there first, then use it.
2. Aliases listed in `labels.json` are **renamed** in-place in each repo (issues/PRs
   keep their labels), so rename inconsistent labels by adding the old name to the
   `aliases` array of the canonical entry.
3. `clean-labels: false` is intentional -- repos may have their own project-specific
   labels that should not be deleted.  Set to `true` only for a deliberate cleanup run.

> **Note:** GitHub's org-level "default labels" REST API returns 404 on this org --
> it is only available on GitHub Enterprise Cloud.  `sync-labels.yml` replaces that
> functionality; trigger a manual `workflow_dispatch` run after creating a new repo.

**Current canonical labels -- 22 entries (prefixed by category):**

Labels follow a `<category>:<name>` prefix convention.  Exceptions: GitHub's special
labels (`good first issue`, `help wanted`) and close-reason labels (`duplicate`,
`invalid`, `wontfix`) stay unprefixed because tools and GitHub itself treat them
specially.

**No prefix -- GitHub specials + close reasons:**

| Name | Color | Description |
|---|---|---|
| `duplicate` | `#cfd3d7` | This issue or pull request already exists |
| `good first issue` | `#7057ff` | Good for newcomers |
| `help wanted` | `#008672` | Extra attention is needed |
| `invalid` | `#e4e669` | This doesn't seem right |
| `wontfix` | `#ffffff` | This will not be worked on |

**`type:` -- Conventional Commits change type** (use `aliases` to rename legacy flat names):

| Name | Color | Description | Aliases (renamed on sync) |
|---|---|---|---|
| `type:breaking-change` | `#b60205` | Incompatible API change -- semver major bump | `breaking change` |
| `type:bug` | `#d73a4a` | Something isn't working | `bug` |
| `type:build` | `#fbca04` | Build system or external dependency changes | `build` |
| `type:chore` | `#ededed` | Maintenance task; no production code change | `chore` |
| `type:ci` | `#006b75` | CI/CD workflow or pipeline changes | `ci`, `ci/cd`, `github-actions`, `github_actions` |
| `type:docs` | `#0075ca` | Documentation only | `documentation`, `docs` |
| `type:feature` | `#a2eeef` | New feature or request | `enhancement`, `feat` |
| `type:perf` | `#e99695` | Performance improvement | `performance` |
| `type:question` | `#d876e3` | Further information is requested | `question` |
| `type:refactor` | `#c5def5` | Code restructuring without behavior change | `refactor`, `refactoring` |
| `type:revert` | `#ee9922` | Reverts a prior commit | `revert` |
| `type:style` | `#fef2c0` | Formatting or lint only; no logic change | `style` |
| `type:test` | `#0e8a16` | Test additions or corrections | `test`, `testing` |

**`auto:` -- applied by automation / bots:**

| Name | Color | Description | Aliases (renamed on sync) |
|---|---|---|---|
| `auto:bot` | `#f9d71c` | Created or updated automatically by a workflow | `automated`, `automation` |
| `auto:dependencies` | `#0366d6` | Dependency update | `dependencies` |
| `auto:sync` | `#1d76db` | File synced from datas-world/.github | `sync` |

**`governance:` -- cross-repo governance tracking:**

| Name | Color | Description | Aliases (renamed on sync) |
|---|---|---|---|
| `governance:branch-protection` | `#e11d48` | Branch protection incomplete -- required status checks missing | `branch-protection-needed` |

---

## Adding issues and PRs to the .github Governance project

Every workflow that **creates or modifies** an issue or pull request -- whether in
`datas-world/.github` itself or in any other org repo -- **must add that item to the
[.github Governance project](https://github.com/orgs/datas-world/projects/4)**.

`GITHUB_TOKEN` cannot write to org-level Projects V2.  Always mint a short-lived
installation token for `datas-world-project-assign` (org secrets `PROJECT_APP_ID` +
`PROJECT_APP_PRIVATE_KEY`) and call the `addProjectV2ItemById` GraphQL mutation.

```yaml
- name: Generate project App token
  uses: actions/create-github-app-token@29824e69f54612133e76f7eaac726eef6c875baf  # v2.2.1
  id: project-token
  with:
    app-id: ${{ vars.PROJECT_APP_ID }}
    private-key: ${{ secrets.PROJECT_APP_PRIVATE_KEY }}
    owner: datas-world

- name: Add to .github Governance project
  env:
    GH_TOKEN: ${{ steps.project-token.outputs.token }}
    ITEM_NODE_ID: ${{ steps.create-issue.outputs.node_id }}   # replace as needed
    PROJECT_ID: ${{ vars.DOT_GITHUB_PROJECT_ID }}             # PVT_kwDODbKvSM4BRVaC
  run: |
    gh api graphql \
      -f query='mutation($proj:ID!,$item:ID!){addProjectV2ItemById(input:{projectId:$proj,contentId:$item}){item{id}}}' \
      -f proj="$PROJECT_ID" \
      -f item="$ITEM_NODE_ID"
```

The reusable `project-assign.yml` workflow wraps the same pattern for callers that
prefer a `workflow_call` interface:

```yaml
jobs:
  add-to-project:
    uses: datas-world/.github/.github/workflows/project-assign.yml@main
    with:
      project_url: ${{ vars.DOT_GITHUB_PROJECT_URL }}
    secrets: inherit
```

> **Project metadata**
> URL: `https://github.com/orgs/datas-world/projects/4`  
> Node ID: stored in org variable `DOT_GITHUB_PROJECT_ID`  
> URL: stored in org variable `DOT_GITHUB_PROJECT_URL`

---

## Maintaining branch protection from the sync workflow

The `post-sync` job in [`sync-shared-files.yml`](sync-shared-files.yml) runs after
every sync and:

1. Adds all open `auto:sync`-labelled PRs in the org to the governance project (via
   `actions/github-script` + GraphQL `addProjectV2ItemById`).
2. For every repo in Group 2 that receives `.github/workflows/pr-lint.yml`, ensures
   `Validate PR title (conventional commits)` is a required status check on the default
   branch (inline bash + `gh` + `jq`).

**When automatic configuration fails** (empty repo, missing branch, permission error),
the inline bash step opens a single tracking issue labelled `governance:branch-protection`
in that repo, adds it to the governance project, and closes it with a resolution
comment when a later run succeeds.

If you add a new Group that syncs a workflow file whose job name should be a required
check, follow the same pattern in the `post-sync` job:

1. Parse the target repos from `sync.yml` using `yq`.
2. Call `PUT /repos/{owner}/{repo}/branches/{branch}/protection` -- always **merge** the
   new check into the existing `required_status_checks.checks` array rather than
   replacing it wholesale (preserve other required checks).
3. On failure, open / update a `branch-protection-needed` tracking issue; close it with
   a resolution comment when a later run succeeds.

---

## Conventional Commits

Commit messages for workflow changes must follow Conventional Commits 1.0.0.
See [conventional-commits.instructions.md](conventional-commits.instructions.md).

---

## Workflow file naming convention

Two kinds of files live in `.github/workflows/`:

### `reusable-*.yml` -- Reusable (callable) workflows
- Triggered via `workflow_call` only.
- Contain the actual logic; called from other org repositories via thin stubs in
  `.github/workflow-stubs/`.
- **Never rename** without simultaneously updating all callers across all repos.
- Prefer parameterisation via `inputs:` over duplicating logic per repo.

### All other `.yml` -- Repository-local workflows
- Run only in the `.github` repository itself (CI, sync, label management, etc.).
- Must **not** use `workflow_call` as their sole trigger.

See `.github/workflows/AGENTS.md` for the full table of reusable workflows and
`.github/workflow-stubs/AGENTS.md` for stub authoring guidance.
