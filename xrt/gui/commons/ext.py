# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Mar 2025"

import re
import sys
import os
import os.path as osp
import shutil

import http.server
import socketserver
import threading
from functools import partial

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
# try:
#     shutil.rmtree(DOCDIR)
# except FileNotFoundError:
#     pass
shutil.copytree(osp.join(CONFDIR, '_images'), osp.join(DOCDIR, '_images'),
                dirs_exist_ok=True)
shutil.copytree(osp.join(CONFDIR, '_themes'), osp.join(DOCDIR, '_themes'),
                dirs_exist_ok=True)
shutil.copy2(osp.join(CONFDIR, 'conf.py'), osp.join(DOCDIR, 'conf.py'))

CSS_PATH = osp.join(DOCDIR, '_static')
CSS_PATH = re.sub('\\\\', '/', CSS_PATH)
JS_PATH = CSS_PATH

xrtQookPageName = 'xrtQookPage'

from . import qt
shouldScaleMath = qt.QtName == "PyQt4" and sys.platform == 'win32'

try:
    from xml.sax.saxutils import escape
    from docutils.utils import SystemMessage
    from sphinx.application import Sphinx
    import sphinx  # analysis:ignore
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


def sphinxify(docstring, context, buildername='html', img_path='',
              wantMessages=False):
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

    confoverrides = {'html_context': context}
    # confoverrides['extensions'] = [
    #     'sphinx.ext.mathjax', 'sphinxcontrib.jquery']

    doctreedir = osp.join(DOCDIR, 'doctrees')
    status, warning = [sys.stderr]*2 if wantMessages else [None]*2
    sphinx_app = Sphinx(srcdir, DOCDIR, DOCDIR, doctreedir, buildername,
                        confoverrides, status=status, warning=warning,
                        freshenv=True, warningiserror=False, tags=None)

    try:
        sphinx_app.build(None, [rst_name])
    except SystemMessage:
        pass
#        output = ("It was not possible to generate rich text help for this "
#                  "object.</br>Please see it in plain text.")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOCDIR, **kwargs)

    def log_message(self, format, *args):
        if "404" in format % args:
            print(format % args)


class LocalWebServer:
    HOST = "127.0.0.1"
    PORT = 8000  # =0: OS chooses a free port

    def __init__(self):
        self.httpd = None
        self.thread = None
        self.port = None

    def start(self):
        self.httpd = socketserver.TCPServer((self.HOST, self.PORT), Handler)
        self.host, self.port = self.httpd.server_address[:2]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join()
