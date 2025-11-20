# -*- coding: utf-8 -*-
import colorama

try:  # for Python 3 compatibility:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    unicode = unicode
    basestring = basestring

_VERBOSITY_ = 10   # [0-100] Regulates the level of diagnostics printout

colors = 'BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', \
    'WHITE', 'RESET'

colorama.init(autoreset=True)


def colorPrint(s, fcolor=None, bcolor=None):
    style = getattr(colorama.Fore, fcolor) if fcolor in colors else \
        colorama.Fore.RESET
    style += getattr(colorama.Back, bcolor) if bcolor in colors else \
        colorama.Back.RESET
    print('{0}{1}'.format(style, s))


def is_sequence(arg):
    """Checks whether *arg* is a sequence."""
    result = (not hasattr(arg, "strip") and hasattr(arg, "__getitem__") or
              hasattr(arg, "__iter__"))
    if result:
        try:
            arg[0]
        except IndexError:
            result = False
        if result:
            result = not isinstance(arg, (basestring, unicode))
    return result
