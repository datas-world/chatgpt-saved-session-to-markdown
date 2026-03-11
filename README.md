<!--
Copyright (C) 2025 Torsten Knodt and contributors
GNU General Public License
SPDX-License-Identifier: GPL-3.0-or-later
-->

# chatgpt-saved-session-to-markdown

Convert saved ChatGPT sessions (`.html` / `.mhtml`) and **PDF prints** into clean **Markdown**.

[![PyPI version][pypi-badge]][pypi-url]
[![Python versions][pyversions-badge]][pyversions-url]
[![License: GPL v3+][license-badge]][license-url]
[![CI][ci-badge]][ci-url]
[![CodeQL][codeql-badge]][codeql-url]
[![pre-commit][precommit-badge]][precommit-url]
[![Dependabot][dependabot-badge]][dependabot-url]

## Features

- **No temp files** for `.mhtml` — processed fully in memory, **attachments embedded** as data URIs.
- Robust role detection for `User` / `Assistant` via BeautifulSoup selectors.
- **PDF support** (via pypdf; best-effort text extraction; still recommends HTML/MHTML).
- **Hybrid executor**: threads for small batches, processes for large ones.
- **Strict failures**: no dummy outputs; non-zero exit if a file cannot be extracted.
- **Heuristic warnings** always on: suggests a better format (HTML vs. MHTML vs. PDF) even if only one is provided.

## Install

```bash
pipx install chatgpt-saved-session-to-markdown
# or
pip install chatgpt-saved-session-to-markdown
```

[pypi-badge]: https://badge.fury.io/py/chatgpt-saved-session-to-markdown.svg
[pypi-url]: https://badge.fury.io/py/chatgpt-saved-session-to-markdown
[pyversions-badge]: https://img.shields.io/pypi/pyversions/chatgpt-saved-session-to-markdown.svg
[pyversions-url]: https://pypi.org/project/chatgpt-saved-session-to-markdown/
[license-badge]: https://img.shields.io/badge/License-GPLv3+-blue.svg
[license-url]: LICENSE
[ci-badge]: https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]: https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/ci.yml
[codeql-badge]: https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/codeql.yml/badge.svg?branch=main
[codeql-url]: https://github.com/datas-world/chatgpt-saved-session-to-markdown/security/code-scanning
[precommit-badge]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[precommit-url]: .pre-commit-config.yaml
[dependabot-badge]: https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg
[dependabot-url]: https://github.com/datas-world/chatgpt-saved-session-to-markdown/network/updates
