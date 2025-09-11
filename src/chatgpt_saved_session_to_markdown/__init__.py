# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""Top-level package for chatgpt-saved-session-to-markdown."""

__all__ = ["__version__"]

try:
    # Written by hatch-vcs at build time
    from ._version import __version__
except Exception:  # pragma: no cover
    try:
        # Editable installs: use importlib.metadata (PEP 566)
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("chatgpt-saved-session-to-markdown")
    except Exception:
        __version__ = "0.0.0-dev"
