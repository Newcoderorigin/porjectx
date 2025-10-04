"""Runtime guardrails for unsupported Python versions."""

from __future__ import annotations

import sys
import warnings


UNSUPPORTED_VERSION = (3, 12)


def warn_if_unsupported() -> None:
    """Emit a warning when running on unsupported interpreter versions."""

    version = sys.version_info
    if (version.major, version.minor) >= UNSUPPORTED_VERSION:
        warnings.warn(
            "Python 3.12+ is not yet validated for Toptek; unexpected behaviour may occur.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["warn_if_unsupported", "UNSUPPORTED_VERSION"]
