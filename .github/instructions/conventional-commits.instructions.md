---
applyTo: "**"
---

<!--
  SPDX-License-Identifier: Apache-2.0

  !! MANAGED FILE -- do not edit this copy directly !!
  Synced from datas-world/.github -- this file is replaced on every sync run.
  To propose changes, open a PR against the source instead:
  https://github.com/datas-world/.github/blob/main/.github/instructions/conventional-commits.instructions.md
-->

# Conventional Commits Instructions -- datas-world org

Specification: <https://www.conventionalcommits.org/en/v1.0.0/>

---

## Commit message format

```
<type>(<scope>)[optional !]: <description>

[optional body]

[optional footers]
```

Both `type` and `scope` are **required**. Use `!` after the closing parenthesis
to signal breaking changes: `feat(api)!: rename input`.

## Valid types

| Type | When to use | SemVer bump |
|------|-------------|-------------|
| `feat` | New feature visible to users/callers | Minor |
| `fix` | Bug fix | Patch |
| `docs` | Documentation changes only | Patch |
| `style` | Formatting/whitespace -- no logic change | Patch |
| `refactor` | Code restructure that is neither fix nor feature | Patch |
| `perf` | Performance improvement | Patch |
| `test` | Adding or correcting tests | Patch |
| `build` | Build system, dependency manifest changes | Patch |
| `ci` | CI/CD workflow changes | Patch |
| `chore` | Maintenance tasks | Patch |
| `revert` | Reverts a prior commit | Patch |

## Scope

Scope identifies the subsystem or component being changed. It is **mandatory**
in all commits in this organisation.

Each repository defines its own scope list in its
`.github/instructions/conventional-commits.instructions.md`.
The following org-wide default scopes are valid in every repository:

| Scope | Applies to |
|-------|------------|
| `deps` | Dependency version bumps |
| `ci` | CI/CD pipeline (caller stubs, actionlint) |
| `docs` | Standalone documentation files |
| `config` | Repository or tool configuration |
| `release` | Release automation changes |

Use the scope that most precisely describes the changed area. Prefer a narrower
scope over a generic one when both apply.

Repo-specific scopes for **datas-world/.github**:

| Scope | Applies to |
|-------|------------|
| `workflow` | Reusable workflow files |
| `contributing` | CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md |
| `runner` | Self-hosted runner hook scripts |
| `instructions` | Copilot instruction files |
| `agents` | Copilot custom agent definition files |
| `prompts` | Copilot slash-command prompt files |
| `templates` | Issue and PR templates |
| `sync` | File-sync workflow and sync.yml configuration |

## Subject line rules

- **Imperative, present tense** -- "add feature", not "added feature" or "adds feature"
- **Lowercase first letter** ? enforced by `pr-lint.yml`
- **No trailing period**
- **? 72 characters** -- readable in `git log --oneline` and the GitHub UI
- **No issue number** in the subject -- issue references belong in the footer

## PR title = squash-merge commit message

Pull request titles are validated by the reusable `pr-lint.yml` workflow and
become the commit subject on the default branch when a PR is squash-merged.
PR titles must therefore follow all commit subject rules above.

### Work-in-progress PRs

A `[WIP] ` prefix (note the trailing space) is **required on draft PRs** and
**optional on non-draft PRs**:

| PR state | `[WIP] ` prefix | Enforced by |
|---|---|---|
| Draft | **Required** | `pr-lint.yml` step 1 |
| Ready for review | Optional | -- |

Non-draft PRs may carry `[WIP] ` as a deliberate signal -- for example, to
trigger an automatic reviewer-assignment workflow when the prefix is later
removed.  The `[WIP] ` prefix is stripped before Conventional Commits
validation regardless of draft state.

Valid examples:
```
[WIP] feat(sync): add private-repo support      ? draft PR (prefix required)
[WIP] feat(sync): add private-repo support      ? non-draft WIP (prefix optional)
feat(sync): add private-repo support            ? non-draft ready (no prefix)
```

For process rules (issue requirements, approval, draft lifecycle) see
[CONTRIBUTING.md -- Draft / work-in-progress PRs](../../CONTRIBUTING.md#draft--work-in-progress-prs).  
The enforcement workflow source is at
[`.github/workflows/pr-lint.yml`](../../.github/workflows/pr-lint.yml).

## Breaking changes

Use `!` after the closing scope parenthesis (`feat(api)!:`) **or** add a
`BREAKING CHANGE:` footer. Both trigger a major SemVer bump per
<https://semver.org/spec/v2.0.0.html>.

```
feat(api)!: rename workflow input node-version to node_version

Callers used `node-version`; renamed to `node_version` to match the
project-wide snake_case convention for all reusable workflow inputs.

BREAKING CHANGE: The `node-version` input has been renamed to `node_version`.
All callers must update their `with:` blocks.
```

## Commit body

Include a body when the subject alone does not fully explain the change.

- Explain **why** the change was made (context, problem being solved)
- Describe what behaviour changed -- before vs. after -- where it aids understanding
- Separate from the subject by **one blank line**
- Wrap lines at **72 characters**
- Use imperative, present tense throughout

```
fix(milestone-assign): skip SemVer parse for non-release milestones

Previously every milestone title was passed to `semver.parse`, which
threw on titles like "Backlog" or "Q4-2025". The fix guards the parse
with a format check so non-release milestones are silently skipped.
```

## Footers

Footers appear after the body, separated by one blank line. Each footer is a
token followed by `: ` or ` #` and a value.

### Issue / ticket references

| Footer | Effect |
|--------|--------|
| `Closes #NNN` | Auto-closes issue NNN on merge to default branch |
| `Fixes #NNN` | Identical to Closes |
| `Resolves #NNN` | Identical to Closes |
| `Refs #NNN` | References without closing |
| `Closes owner/repo#NNN` | Cross-repository issue close |

Use **`Closes`** as the org-default keyword; use `Refs` when the issue is only
related but not resolved by the change.

Multiple issues: `Closes #1, closes #2`.

The keywords activate only when the PR targets the repository's default branch.

#### Issue reference policy

- **PRs must reference at least one issue** using a GitHub closing or reference
  keyword.
- If no suitable issue exists, AI assistants must propose creating one and help
  the author open it before the PR is submitted.
- PRs without any issue reference are only permitted with explicit approval from
  the reviewer or merger.
- **PRs opened from forks always require a linked issue.**

### Other footers

| Footer | When to use |
|--------|-------------|
| `BREAKING CHANGE: <description>` | Required when `!` is used; explains the break and migration path |
| `Signed-off-by: Full Name <email>` | **Required** on all commits -- certifies the [DCO](https://developercertificate.org/) |
| `Co-authored-by: Name <email>` | Pair programming or AI-assisted work |

Configure git to add `Signed-off-by` automatically:

```bash
git config --global format.signOff true
```

For GitHub Copilot CLI co-authorship use:

```
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

### Footer ordering

When multiple footers are present, use this order:

1. `BREAKING CHANGE:` (if applicable)
2. `Closes` / `Fixes` / `Resolves` / `Refs` (issue references)
3. `Co-authored-by:` (co-authors)
4. `Signed-off-by:` (DCO sign-off, last)

## Full example

```
feat(sync-topics): add support for private repos via app token

Public repos were already supported via the default GITHUB_TOKEN.
Private repos require a GitHub App installation token with the
`repo:read` scope. This change adds an optional `app-token` input
that, when provided, replaces the default token for all API calls.

Closes #78
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
Signed-off-by: Torsten Knodt <torsknod2@users.noreply.github.com>
```

## Tag format

Tags carry **no `v` prefix**: `1.0.0`, not `v1.0.0`.

---

For the full contribution process and pull request rules see
[CONTRIBUTING.md](../../CONTRIBUTING.md).
