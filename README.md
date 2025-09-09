<!--
Copyright (C) 2025 Torsten Knodt and contributors
GNU General Public License
SPDX-License-Identifier: GPL-3.0-or-later
-->
# chatgpt-saved-session-to-markdown

Convert saved ChatGPT sessions (`.html` / `.mhtml`) and **PDF prints** into clean **Markdown**.


[![PyPI version](https://img.shields.io/pypi/v/chatgpt-saved-session-to-markdown.svg)](https://pypi.org/project/chatgpt-saved-session-to-markdown/)
[![Python versions](https://img.shields.io/pypi/pyversions/chatgpt-saved-session-to-markdown.svg)](https://pypi.org/project/chatgpt-saved-session-to-markdown/)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3+-blue.svg)](LICENSE)
[![CI](https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/ci.yml/badge.svg)](https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/ci.yml)
[![CodeQL](https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/codeql.yml/badge.svg)](https://github.com/datas-world/chatgpt-saved-session-to-markdown/actions/workflows/codeql.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/datas-world/chatgpt-saved-session-to-markdown/network/updates)


## Features
- **No temp files** for `.mhtml` — processed fully in memory, **attachments embedded** as data URIs.
- Robust role detection for `User` / `Assistant` via BeautifulSoup selectors.
- **PDF support** (best-effort text extraction; still recommends HTML/MHTML).
- **Hybrid executor**: threads for small batches, processes for large ones.
- **Strict failures**: no dummy outputs; non-zero exit if a file cannot be extracted.
- **Heuristic warnings** always on: suggests a better format (HTML vs. MHTML vs. PDF) even if only one is provided.

## Install
```bash
pipx install chatgpt-saved-session-to-markdown
# or
pip install chatgpt-saved-session-to-markdown
```

## CLI
```bash
chatgpt-saved-session-to-markdown run -v --outdir ./out ~/Downloads/*.mhtml ~/Downloads/*.html ~/Downloads/*.pdf
# auto-choose threads/processes; always warns if another format is likely better

# license headers auto-fix (also used by pre-commit)
chatgpt-saved-session-to-markdown headers-fix
```

## Credits
- Author: Torsten Knodt <torsten.knodt+com.github.datas-world.chatgpt-saved-session-to-markdown@datas-world.de>
- License: GPL-3.0-or-later
