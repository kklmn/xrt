# -*- coding: utf-8 -*-
__versioninfo__ = (2, 0, 0, 'b0')
_base = '.'.join(map(str, __versioninfo__[:3]))
if len(__versioninfo__) > 3:
    __version__ = _base + str(__versioninfo__[3])
else:
    __version__ = _base
__date__ = "05 Feb 2026"
