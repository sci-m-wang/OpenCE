"""Composite CE methods shipping with OpenCE."""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import BaseMethod, MethodArtifacts, MethodRegistry, PillarBundle

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .ace_closed_loop import ACEClosedLoopMethod as _ACEClosedLoopMethod


def __getattr__(name: str):  # pragma: no cover - trivial proxy
    if name == "ACEClosedLoopMethod":
        module = import_module("opence.methods.ace_closed_loop")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "BaseMethod",
    "MethodArtifacts",
    "MethodRegistry",
    "PillarBundle",
    "ACEClosedLoopMethod",
]
