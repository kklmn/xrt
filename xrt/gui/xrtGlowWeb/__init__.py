# -*- coding: utf-8 -*-
"""
xrtGlowWeb -- experimental browser front end for xrtGlow-style scenes.

This package is intentionally separate from :mod:`xrt.gui.xrtGlow`.  It reuses
raycing serialization and propagation, while serving a lightweight browser
viewer for embedding in documentation systems.
"""

__author__ = "Roman Chernikov, OpenAI"
__date__ = "02 May 2026"

__all__ = ["serve"]


def serve(*args, **kwargs):
    from .server import serve as _serve
    return _serve(*args, **kwargs)
