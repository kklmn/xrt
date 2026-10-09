# -*- coding: utf-8 -*-
"""Literal formatting shared by the GUI code generators."""

import ast


def path_literal(value):
    """Prefer raw literals for paths, without changing their values."""
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(path_literal(v) for v in value) + ']'
    if isinstance(value, str):
        for quote in ('"', "'"):
            literal = 'r' + quote + value + quote
            try:
                if ast.literal_eval(literal) == value:
                    return literal
            except (SyntaxError, ValueError, UnicodeError):
                pass
    return repr(value)
