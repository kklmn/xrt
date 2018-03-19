# -*- coding: utf-8 -*-
__copyright__ = u'2016 Konstantin Klementiev, MIT License'
__date__ = "19 Mar 2018"

#import os
import shutil
from docutils import nodes
from docutils.parsers.rst import Directive


class addverify(nodes.General, nodes.Element):
    pass


class AddVerify(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False

    def run(self):
        if not lapp.builder.format.startswith('html'):
            return [nodes.raw('', '')]
        uri = "googlefc270ce8cd25e311.html"
        htmldest = lapp.outdir
        shutil.copy2(uri, htmldest)
        return [nodes.raw('', '')]


def setup(app):
    global lapp
    lapp = app
    app.add_node(addverify)
    app.add_directive('addverify', AddVerify)
    return {'version': '0.1'}   # identifies the version of our extension
