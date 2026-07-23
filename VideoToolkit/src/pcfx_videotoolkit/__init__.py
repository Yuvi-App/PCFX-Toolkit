"""Universal NEC PC-FX RAINBOW FMV tooling."""

from __future__ import annotations

__version__ = "0.1.0"


class PcfxError(RuntimeError):
    """Base error for every failure this toolkit raises deliberately."""


__all__ = ["PcfxError", "__version__"]
