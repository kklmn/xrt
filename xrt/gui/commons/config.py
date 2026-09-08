# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "8 Sep 2026"

import os

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # python 2

iniDir = os.path.expanduser(os.path.join('~', '.xrt'))
if not os.path.exists(iniDir):
    os.makedirs(iniDir)

encoding = 'utf-8'

iniPaths = os.path.join(iniDir, 'paths.ini')
configPaths = ConfigParser()
configPaths.read(iniPaths, encoding=encoding)
# configPaths.optionxform = str  # makes it case sensitive


def get(conf, section, entry, default=None):
    if conf.has_option(section, entry):
        res = conf.get(section, entry)
        # if isinstance(default, str):
        #     return res
        # else:
        #     try:
        #         return eval(res)
        #     except (SyntaxError, NameError):
        #         return res
        return res
    else:
        return default


def put(conf, section, entry, value):
    if not conf.has_section(section):
        conf.add_section(section)
    conf.set(section, entry, value)


def path(section, what):
    path = configPaths.get(section, what)
    try:
        return os.path.dirname(path)
    except Exception:
        return ''


def write_configs():  # in mainWindow's closeEvent
    with open(iniPaths, 'w+', encoding=encoding) as cf:
        configPaths.write(cf)
