<!--
Copyright (C) 2025 Torsten Knodt and contributors
GNU General Public License
SPDX-License-Identifier: GPL-3.0-or-later
-->

# GitHub Copilot Development Instructions

This document provides mandatory development guidelines for GitHub Copilot when working on this project. These instructions ensure code quality, maintainability, and compliance with project standards.

## Quality Assurance Requirements

### Pre-commit Compliance

GitHub Copilot **SHALL** not consider any development task completed until all [`pre-commit`](https://pre-commit.com/) checks are passing successfully. This includes but is not limited to:

- [Code formatting with Black](https://black.readthedocs.io/)
- [Import sorting](https://pycqa.github.io/isort/) with [isort](https://pycqa.github.io/isort/)
- [Linting](https://docs.astral.sh/ruff/) with [Ruff](https://docs.astral.sh/ruff/)
- [Documentation style](http://www.pydocstyle.org/) with [pydocstyle](http://www.pydocstyle.org/)
- [Documentation coverage](https://interrogate.readthedocs.io/) with [interrogate](https://interrogate.readthedocs.io/)
- [Security scanning](https://bandit.readthedocs.io/) with [Bandit](https://bandit.readthedocs.io/)
- [License compliance](https://reuse.software/) with [REUSE](https://reuse.software/)
- [Dependency security](https://pypa.github.io/pip-audit/) with [pip-audit](https://pypa.github.io/pip-audit/)

### Prohibited Actions

The following actions are **STRICTLY FORBIDDEN** without explicit approval from a project maintainer:

1. **Disabling or weakening pre-commit checks** - All quality gates must remain active
1. **Modifying test configurations** to reduce coverage or skip tests
1. **Bypassing security checks** or ignoring security warnings
1. **Removing or weakening linting rules** without documented justification
1. **Altering files in `tests/data`** - Test data files (except `*.license` files) must not be modified to suppress warnings or errors

## Error and Warning Management

### Suppression Strategy

When addressing warnings or errors, proposals for suppression **SHALL** occur at the lowest possible level:

1. **Inline suppression** - Use specific `# noqa` comments with error codes when justified
1. **Function/class level** - Apply suppressions to the smallest possible scope
1. **File level** - Only when the entire file requires the exception
1. **Global level** - Never without maintainer approval

Example of proper inline suppression:

```python
# This is acceptable for legitimate cases
result = subprocess.run(command, shell=True)  # noqa: S602 # Intentional shell use for user input
```

### Test Failure Analysis

When tests fail, a **deep root-cause analysis** is required following this process:

1. **Determine failure type**:

   - Implementation error (code bug)
   - Test error (incorrect test assumptions)
   - Environment issue (dependency, configuration)

1. **Document findings** in commit messages or pull request descriptions

1. **Escalate decision** - The final determination of whether to fix implementation or modify tests **SHALL** be made by project maintainers

## Standards and References

All implemented solutions **SHALL** reference applicable standards with deep hyperlinks:

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [SPDX License Identifier Guidelines](https://spdx.github.io/spdx-spec/v2.3/SPDX-license-identifier/)
- [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Off-the-Shelf (COTS) Preference

**STRICTLY prefer** established, well-maintained off-the-shelf software over custom implementations:

### Preferred Tools and Libraries

- **Code Quality**: [Black](https://black.readthedocs.io/), [Ruff](https://docs.astral.sh/ruff/), [isort](https://pycqa.github.io/isort/)
- **Testing**: [pytest](https://docs.pytest.org/), [pytest-cov](https://pytest-cov.readthedocs.io/)
- **Documentation**: [pydocstyle](http://www.pydocstyle.org/), [interrogate](https://interrogate.readthedocs.io/)
- **Security**: [Bandit](https://bandit.readthedocs.io/), [pip-audit](https://pypa.github.io/pip-audit/)
- **Web Parsing**: [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/), [lxml](https://lxml.de/)
- **CLI Framework**: [Typer](https://typer.tiangolo.com/)
- **Markup Processing**: [markdownify](https://pypi.org/project/markdownify/)

### Custom Implementation Criteria

Custom implementations are only acceptable when:

1. **No suitable off-the-shelf solution exists** for the specific use case
1. **Performance requirements** cannot be met by existing solutions
1. **Licensing constraints** prevent use of available alternatives
1. **Maintainer approval** has been explicitly granted

## Compliance Verification

Before marking any task as complete:

1. **Run full pre-commit suite**: `pre-commit run --all-files`
1. **Execute test suite**: `python -m pytest`
1. **Verify documentation**: Ensure all public APIs are documented
1. **Check licensing**: Confirm SPDX headers are present and correct

## Escalation Process

When in doubt about any of these requirements:

1. **Document the concern** in detail
1. **Propose alternative approaches** with justification
1. **Request maintainer review** before proceeding
1. **Wait for explicit approval** before implementing controversial changes

______________________________________________________________________

**Note**: These instructions are binding for all GitHub Copilot interactions with this codebase. Non-compliance may result in rejected contributions and requests for complete rework.
