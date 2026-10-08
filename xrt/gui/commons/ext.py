# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "8 Oct 2026"

import re
import sys
import os
import os.path as osp
import shutil
import io
import keyword
import token
import tokenize

import http.server
import socketserver
import threading

from . import qt
shouldScaleMath = qt.QtName == "PyQt4" and sys.platform == 'win32'

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


class LineNumberArea(qt.QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return qt.QSize(self.editor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.editor.lineNumberAreaPaintEvent(event)


class PythonTextEdit(qt.QPlainTextEdit):
    INDENT = " "*4
    LINE_LENGTH = 80
    COLOR_STRING = "#00AA00"
    COLOR_NUMBER = "#800000"
    COLOR_COMMENT = "#ADADAD"
    COLOR_KEYWORD = "#0000FF"
    COLOR_BUILTIN = "#A20090"
    COLOR_CLASS = "#000000"
    COLOR_FUNC = "#000000"
    COLOR_SELF = "#924939"
    COLOR_LINE = "#E8F2FE"
    COLOR_LINENUM = "#EFEFEF"
    COLOR_LINELENGTH = "#EEEEEE"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.defaultFontSize = None

        self.builtins = list(__builtins__.keys()) + ['True', 'False', 'None']
        self.formats = {}
        self.formats[token.STRING] = self.make_format(self.COLOR_STRING)
        self.formats[token.NUMBER] = self.make_format(self.COLOR_NUMBER)
        self.formats[token.COMMENT] = self.make_format(self.COLOR_COMMENT)

        self.keyword_format = self.make_format(self.COLOR_KEYWORD, bold=True)
        self.builtin_format = self.make_format(self.COLOR_BUILTIN, bold=False)
        self.class_fmt = self.make_format(self.COLOR_CLASS, bold=True)
        # self.class_fmt.setFontUnderline(True)
        self.func_fmt = self.make_format(self.COLOR_FUNC, bold=True)
        self.self_fmt = self.make_format(self.COLOR_SELF)
        self.self_fmt.setFontItalic(True)

        self.lineNumberArea = LineNumberArea(self)

        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)

        self.updateLineNumberAreaWidth(0)
        self.highlightCurrentLine()

        self.textChanged.connect(self.highlight)

        self._font_size = self.font().pointSizeF()

        self.zoomInAction = qt.QAction("Zoom In", self)
        self.zoomInAction.setShortcut(qt.QKeySequence.ZoomIn)
        self.zoomInAction.triggered.connect(self.zoomInEditor)

        self.zoomOutAction = qt.QAction("Zoom Out", self)
        self.zoomOutAction.setShortcut(qt.QKeySequence.ZoomOut)
        self.zoomOutAction.triggered.connect(self.zoomOutEditor)

        self.resetZoomAction = qt.QAction("Reset Zoom", self)
        self.resetZoomAction.setShortcut("Ctrl+0")
        self.resetZoomAction.triggered.connect(self.resetZoom)

        self.addAction(self.zoomInAction)
        self.addAction(self.zoomOutAction)
        self.addAction(self.resetZoomAction)

        self.setContextMenuPolicy(qt.Qt.DefaultContextMenu)

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        menu.addAction(self.zoomInAction)
        menu.addAction(self.zoomOutAction)
        menu.addAction(self.resetZoomAction)
        menu.exec(event.globalPos())

    def setFont(self, font):
        if self.defaultFontSize is None:
            self.defaultFontSize = font.pointSize()
        super().setFont(font)

    def zoomInEditor(self):
        self.changeFontSize(+1)

    def zoomOutEditor(self):
        self.changeFontSize(-1)

    def resetZoom(self):
        if self.defaultFontSize is None:
            return
        font = self.font()
        font.setPointSize(self.defaultFontSize)
        self.setFont(font)
        if hasattr(self, "lineNumberArea"):
            self.updateLineNumberAreaWidth(0)

    def changeFontSize(self, delta):
        font = self.font()
        size = max(6, min(48, font.pointSize() + delta))
        font.setPointSize(size)
        self.setFont(font)
        if hasattr(self, "lineNumberArea"):
            self.updateLineNumberAreaWidth(0)

    def wheelEvent(self, event):
        if event.modifiers() & qt.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoomInEditor()
            else:
                self.zoomOutEditor()
            event.accept()
            return
        super().wheelEvent(event)

    def make_format(self, color, bold=False):
        fmt = qt.QTextCharFormat()
        fmt.setForeground(qt.QColor(color))
        if bold:
            fmt.setFontWeight(75)

        return fmt

    def clear_formatting(self):
        cursor = qt.QTextCursor(self.document())
        cursor.select(qt.QTextCursor.Document)
        fmt = qt.QTextCharFormat()
        fmt.setForeground(qt.QColor("black"))
        cursor.setCharFormat(fmt)

    def highlight(self):
        text = self.toPlainText()
        self.blockSignals(True)
        cursor = self.textCursor()
        pos = cursor.position()
        self.clear_formatting()

        try:
            line_offsets = [0]

            for line in text.splitlines(True):
                line_offsets.append(line_offsets[-1] + len(line))

            prev_tok = None
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                tok_type = tok.type

                start_line, start_col = tok.start
                end_line, end_col = tok.end

                start = line_offsets[start_line-1] + start_col
                end = line_offsets[end_line-1] + end_col

                fmt = None
                if tok_type == token.NAME:
                    if tok.string in self.builtins:
                        fmt = self.builtin_format
                    elif tok.string in keyword.kwlist:
                        fmt = self.keyword_format
                    elif prev_tok == "class":
                        fmt = self.class_fmt
                    elif prev_tok == "def":
                        fmt = self.func_fmt
                    elif tok.string == "self":
                        fmt = self.self_fmt

                elif tok_type in self.formats:
                    fmt = self.formats[tok_type]

                if fmt is not None:
                    c = qt.QTextCursor(self.document())
                    c.setPosition(start)
                    c.setPosition(end, qt.QTextCursor.KeepAnchor)
                    c.mergeCharFormat(fmt)

                if tok_type == token.NAME:
                    prev_tok = tok.string
                else:
                    prev_tok = None

        except tokenize.TokenError, IndentationError:
            pass

        cursor.setPosition(pos)
        self.setTextCursor(cursor)

        self.blockSignals(False)

    def keyPressEvent(self, event):
        if event.key() == qt.Qt.Key_Tab:
            self.textCursor().insertText(self.INDENT)
            return
        super().keyPressEvent(event)

    def lineNumberAreaWidth(self):
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(
                0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(qt.QRect(
            cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))

    def highlightCurrentLine(self):
        selections = []
        if not self.isReadOnly():
            sel = qt.QTextEdit.ExtraSelection()
            sel.format.setBackground(qt.QColor(self.COLOR_LINE))
            sel.format.setProperty(sel.format.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            selections.append(sel)
        self.setExtraSelections(selections)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = qt.QPainter(self.viewport())
        fm = self.fontMetrics()

        # Width of one character in the current font
        char_width = fm.horizontalAdvance('9')
        x = int(self.LINE_LENGTH * char_width)
        pen = qt.QPen(qt.QColor(self.COLOR_LINELENGTH))
        pen.setWidthF(2.5)
        painter.setPen(pen)
        painter.drawLine(x, 0, x, self.viewport().height())

    def lineNumberAreaPaintEvent(self, event):
        painter = qt.QPainter(self.lineNumberArea)
        painter.fillRect(event.rect(), qt.QColor(self.COLOR_LINENUM))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        current_line = self.textCursor().blockNumber()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                if block_number == current_line:
                    painter.setPen(qt.Qt.black)
                    font = painter.font()
                    font.setBold(True)
                    painter.setFont(font)
                else:
                    painter.setPen(qt.Qt.darkGray)
                    font = painter.font()
                    font.setBold(False)
                    painter.setFont(font)

                painter.drawText(0, top, self.lineNumberArea.width() - 4,
                                 self.fontMetrics().height(), qt.Qt.AlignRight,
                                 str(block_number + 1))
            block = block.next()
            top = bottom
            if block.isValid():
                bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1


# Sphinx docs
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
    PORT = 0  # OS chooses a free port

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
