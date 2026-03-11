# Copilot Instructions -- datas-world/.github

Organisation-level GitHub infrastructure: reusable workflows, community health files,
and AI agent support files.

**See also:** [AGENTS.md](../AGENTS.md) ? [.github/AGENTS.md](AGENTS.md)

## What this repository contains

- **Community health files** at the repo root: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `README.md`, `LICENSE`
- **Reusable workflows** in `.github/workflows/` -- all triggered by `on: workflow_call`
- **Issue templates** in `.github/ISSUE_TEMPLATE/`
- **AI agent files** in `.github/` (instructions, agents, prompts, this file)

There is **no application code** here -- no TypeScript, no Python, no build step, no tests.

## Workflow authoring rules

- **SHA-pin every `uses:`**: `actions/checkout@<40-char-SHA>  # v4.2.2`
- **`permissions: {}`** at workflow level; each job declares its own minimum.
- **`runs-on` for API-only jobs**: `runs-on: ubuntu-slim`
- **CodeQL and npm jobs** stay on `ubuntu-latest` -- they need specific tooling and
  may exceed the 15-minute slim limit.
- **Job `name:` is the status-check context** -- never rename without updating branch
  protection in all calling repos.
- Every reusable workflow **must** declare `on: workflow_call` and
  `permissions: {}` at the workflow level.
- For full authoring rules, runner selection details, and the ubuntu-slim pre-installed
  tool table see `.github/instructions/github-actions.instructions.md`.

## Community health files

- `CODE_OF_CONDUCT.md` includes an explicit scope clause: it does **not** apply to
  private repositories, personal projects, or proprietary work. Preserve this clause.
- `SECURITY.md` must never promise specific response timelines -- the org-wide policy
  is "no commitment to response timeline".
- `CONTRIBUTING.md` is access-controlled -- never use open-source-style language like
  "anyone can contribute" or "fork and send a PR".
- `README.md` is the org profile shown on `github.com/datas-world`. Keep it accurate
  and current.

## Conventional Commits

All commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
`chore`, `revert`. Breaking changes: `feat!:` or `BREAKING CHANGE:` footer.

**No `v` prefix** on any tag: `1.0.0`, not `v1.0.0`.

## Milestone and topic workflows use TypeScript actions

`milestone-assign.yml` and `sync-topics-node.yml` delegate to private org actions
(not inline shell). To upgrade either action, build a new orphan release commit
in the action repo and update the SHA-pinned `uses:` line here.

| Workflow | Action repo | Current version |
|---|---|---|
| `milestone-assign.yml` | `datas-world/action-milestone-assign` | `0.1.0` |
| `sync-topics-node.yml` | `datas-world/action-sync-topics-node` | `0.1.0` |

SemVer bump logic: `breaking change` ? major, `enhancement` ? minor, all other labels ? patch.
See the action repo for full implementation details and tests.

## When you add a new reusable workflow

1. Add it to `.github/workflows/` following the SHA-pinning and permissions rules above.
2. Update `AGENTS.md` (root) -- add it to the "Reusable workflow catalogue" table.
3. Update `.github/AGENTS.md` -- document any new patterns it introduces.
4. Update `.github/instructions/github-actions.instructions.md` if it uses new patterns.
5. Consider whether a new `.github/prompts/*.prompt.md` would help callers.
