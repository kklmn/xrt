# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "18 Oct 2017"

import re
import sys
import os
import os.path as osp
import shutil

#  Spyderlib modules can reside in either Spyder or Spyderlib, so we check both
#  It's definitely not the optimal solution, but it works.

try:
    from spyder.widgets.sourcecode import codeeditor  # analysis:ignore
    isSpyderlib = True
except ImportError:
    try:
        from spyderlib.widgets.sourcecode import codeeditor  # analysis:ignore
        isSpyderlib = True
    except ImportError:
        isSpyderlib = False

try:
    from spyder.widgets.externalshell import pythonshell
    isSpyderConsole = True
except (ImportError, KeyError):
    try:
        from spyderlib.widgets.externalshell import pythonshell  # analysis:ignore
        isSpyderConsole = True
    except (ImportError, KeyError):
        isSpyderConsole = False

CONFDIR = osp.dirname(osp.abspath(__file__))
DOCDIR = osp.expanduser(osp.join('~', '.xrt', 'doc'))

shutil.rmtree(osp.join(DOCDIR, '_images'))
shutil.copytree(osp.join(CONFDIR, '_images'), osp.join(DOCDIR, '_images'))
shutil.rmtree(osp.join(DOCDIR, '_themes'))
shutil.copytree(osp.join(CONFDIR, '_themes'), osp.join(DOCDIR, '_themes'))
shutil.copy2(osp.join(CONFDIR, 'conf.py'), osp.join(DOCDIR, 'conf.py'))

CSS_PATH = osp.join(DOCDIR, '_static')
CSS_PATH = re.sub('\\\\', '/', CSS_PATH)
JS_PATH = CSS_PATH

xrtQookPageName = 'xrtQookPage'
xrtQookPage = 'file:///' + osp.join(DOCDIR, xrtQookPageName+'.html')
xrtQookPage = re.sub('\\\\', '/', xrtQookPage)

from . import qt
shouldScaleMath = qt.QtName == "PyQt4" and sys.platform == 'win32'


try:
    from xml.sax.saxutils import escape
    from docutils.utils import SystemMessage
    from sphinx.application import Sphinx
    import codecs
    isSphinx = True
except Exception:
    isSphinx = False


def generate_context(name='', argspec='', note=''):
    context = {'name': name,
               'argspec': argspec,
               'note': note,
               'css_path': CSS_PATH,
               'js_path': JS_PATH,
               'shouldScaleMath': 'true' if shouldScaleMath else ''}
    return context


def sphinxify(docstring, context, buildername='html', img_path=''):
    """
    Largely modified Spyder's sphinxify.
    """
    if img_path:
        if os.name == 'nt':
            img_path = img_path.replace('\\', '/')
        leading = '/' if os.name.startswith('posix') else ''
        docstring = docstring.replace('_images', leading+img_path)

    srcdir = osp.join(DOCDIR, '_sources')
    if not osp.exists(srcdir):
        os.makedirs(srcdir)
    base_name = osp.join(srcdir, xrtQookPageName)
    rst_name = base_name + '.rst'

    # This is needed so users can type \\ on latex eqnarray envs inside raw
    # docstrings
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

    confoverrides = {'html_context': context,
                     'extensions': ['sphinx.ext.mathjax']}

    doctreedir = osp.join(DOCDIR, 'doctrees')
    sphinx_app = Sphinx(srcdir, DOCDIR, DOCDIR, doctreedir, buildername,
                        confoverrides, status=None, warning=None,
                        freshenv=True, warningiserror=False, tags=None)

    try:
        sphinx_app.build(None, [rst_name])
    except SystemMessage:
        pass
#        output = ("It was not possible to generate rich text help for this "
#                  "object.</br>Please see it in plain text.")
