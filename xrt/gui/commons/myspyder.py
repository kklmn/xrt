# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "18 Oct 2017"

import re

#  Spyderlib modules can reside in either Spyder or Spyderlib, so we check both
#  It's definitely not the optimal solution, but it works.
try:
    from spyderlib.widgets.sourcecode import codeeditor  # analysis:ignore
    isSpyderlib = True
except ImportError:
    try:
        from spyder.widgets.sourcecode import codeeditor  # analysis:ignore
        isSpyderlib = True
    except ImportError:
        isSpyderlib = False

try:
    from spyderlib.widgets.externalshell import pythonshell
    isSpyderConsole = True
except ImportError:
    try:
        from spyder.widgets.externalshell import pythonshell  # analysis:ignore
        isSpyderConsole = True
    except ImportError:
        isSpyderConsole = False

try:
#    from spyderlib.utils.inspector.sphinxify import (CSS_PATH, sphinxify,
#                                                     generate_context)
    from spyderlib.utils.inspector.sphinxify import CSS_PATH, generate_context
    spyderHelpPath = "spyderlib.utils.inspector"
    isSphinx = True
except (ImportError, TypeError):
    try:
#        from spyder.utils.inspector.sphinxify import (CSS_PATH, sphinxify,
#                                                      generate_context)
        from spyder.utils.inspector.sphinxify import CSS_PATH, generate_context
        spyderHelpPath = "spyder.utils.inspector"
        isSphinx = True
    except ImportError:
        CSS_PATH = None
        sphinxify = None
        isSphinx = False

if not isSphinx:
    try:
#        from spyderlib.utils.help.sphinxify import (CSS_PATH, sphinxify,  # analysis:ignore
#                                                    generate_context)  # analysis:ignore
        from spyderlib.utils.help.sphinxify import CSS_PATH, generate_context  # analysis:ignore
        spyderHelpPath = "spyderlib.utils.help"
        isSphinx = True
    except (ImportError, TypeError):
        try:
#            from spyder.utils.help.sphinxify import (CSS_PATH, sphinxify,  # analysis:ignore
#                                                        generate_context)  # analysis:ignore
            from spyder.utils.help.sphinxify import CSS_PATH, generate_context  # analysis:ignore
            spyderHelpPath = "spyder.utils.help"
            isSphinx = True
        except ImportError:
            pass

if isSphinx:
    if CSS_PATH is not None:
        CSS_PATH = re.sub('\\\\', '/', CSS_PATH)

# imports for sphinxify
import os.path as osp
import shutil
from tempfile import mkdtemp
from xml.sax.saxutils import escape
from docutils.utils import SystemMessage
from sphinx.application import Sphinx
from spyder.config.base import get_module_source_path
import codecs
from spyder.utils import encoding


def sphinxify(docstring, context, buildername='html'):
    """
    Reimplemented Spyder's sphinxify in order to fix its ExtensionError:
    "Could not import extension sphinx.ext.autosummary"
    on Python 2.7 and Spyder or on anaconda.
    The solution is to add 'extensions' list to confoverrides dict.
    """
    srcdir = mkdtemp()
    srcdir = encoding.to_unicode_from_fs(srcdir)

    base_name = osp.join(srcdir, 'docstring')
    rst_name = base_name + '.rst'

    if buildername == 'html':
        suffix = '.html'
    else:
        suffix = '.txt'
    output_name = base_name + suffix

    # This is needed so users can type \\ on latex eqnarray envs inside raw
    # docstrings
    if context['right_sphinx_version'] and context['math_on']:
        docstring = docstring.replace('\\\\', '\\\\\\\\')

    # Add a class to several characters on the argspec. This way we can
    # highlight them using css, in a similar way to what IPython does.
    # NOTE: Before doing this, we escape common html chars so that they
    # don't interfere with the rest of html present in the page
    argspec = escape(context['argspec'])
    for char in ['=', ',', '(', ')', '*', '**']:
        argspec = argspec.replace(
            char, '<span class="argspec-highlight">' + char + '</span>')
    context['argspec'] = argspec

    doc_file = codecs.open(rst_name, 'w', encoding='utf-8')
    doc_file.write(docstring)
    doc_file.close()

    confdir = osp.join(get_module_source_path(spyderHelpPath))

    confoverrides = {'html_context': context,
                     'extensions': ['sphinx.ext.mathjax']}

    doctreedir = osp.join(srcdir, 'doctrees')

    sphinx_app = Sphinx(srcdir, confdir, srcdir, doctreedir, buildername,
                        confoverrides, status=None, warning=None,
                        freshenv=True, warningiserror=False, tags=None)
    try:
        sphinx_app.build(None, [rst_name])
    except SystemMessage:
        output = ("It was not possible to generate rich text help for this "
                  "object.</br>Please see it in plain text.")
        return output

    if osp.exists(output_name):
        output = codecs.open(output_name, 'r', encoding='utf-8').read()
        output = output.replace('<pre>', '<pre class="literal-block">')
    else:
        output = ("It was not possible to generate rich text help for this "
                  "object.</br>Please see it in plain text.")
        return output

    shutil.rmtree(srcdir, ignore_errors=True)

    return output
