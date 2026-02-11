# -*- coding: utf-8 -*-
__versioninfo__ = (2, 0, 0, 'b0', 'post1')

_base = '.'.join(map(str, __versioninfo__[:3]))

_suffix = ''
for part in __versioninfo__[3:]:
    if part.startswith(('a', 'b', 'rc')):
        _suffix += part
    else:
        _suffix += '.' + part

__version__ = _base + _suffix

__date__ = "11 Feb 2026"
