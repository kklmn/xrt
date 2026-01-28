# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 18:01:30 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import os  # analysis:ignore
import time  # analysis:ignore
from ..commons import qt  # analysis:ignore
from ..commons import ext  # analysis:ignore


class LevelRestrictedModel(qt.QStandardItemModel):
    def __init__(self):
        super().__init__()
        self._cached_drag_row = None
        self._cached_drag_parent = None

    def supportedDragActions(self):
        return qt.Qt.MoveAction

    def canDropMimeData(self, data, action, row, column, parent_index):
        if str(parent_index.data(role=qt.Qt.UserRole)) == 'top' and \
                column == 0 and row > 1:
            return True
        return False

    def mimeData(self, indexes):
        mime = super().mimeData(indexes)
        if indexes:
            index = indexes[0]
            self._cached_drag_row = index.row()
            self._cached_drag_parent = self.itemFromIndex(index.parent()) \
                if index.parent().isValid() else self.invisibleRootItem()
        return mime

    def dropMimeData(self, data, action, row, column, parent_index):
        if str(parent_index.data(role=qt.Qt.UserRole)) != 'top':
            return False

        if action != qt.Qt.MoveAction:
            return False

        # decode source
        if not data.hasFormat("application/x-qstandarditemmodeldatalist"):
            return False

        if self._cached_drag_row is None or self._cached_drag_parent is None:
            return False
        # locate original item
        # retrieve via internal stored property or last hovered index
        # OR reconstruct from data (advanced)

        # For now, assume we're tracking source during drag
        moved_row = self._cached_drag_row
        source_parent = self._cached_drag_parent

        # Move entire row (multi-column safe)
        items = source_parent.takeRow(moved_row)

        dest_parent = self.itemFromIndex(parent_index)

#        if parent_index.isValid():
#            dest_parent = self.itemFromIndex(parent_index)
#        else:
#            dest_parent = self.invisibleRootItem()

        if row < 0:
            dest_parent.appendRow(items)
        else:
            dest_parent.insertRow(row, items)

        self._cached_drag_row = None
        self._cached_drag_parent = None

        return True


class SphinxWorker(qt.QObject):
    html_ready = qt.Signal()

    def prepare(self, doc=None, docName=None, docArgspec=None,
                docNote=None, img_path=""):
        self.doc = doc
        self.docName = docName
        self.docArgspec = docArgspec
        self.docNote = docNote
        self.img_path = img_path

    def render(self):
        cntx = ext.generate_context(
            name=self.docName,
            argspec=self.docArgspec,
            note=self.docNote)
        ext.sphinxify(self.doc, cntx, img_path=self.img_path)
        self.thread().terminate()
        self.html_ready.emit()


class BusyIconWorker(qt.QObject):
    BUSYICONDT = 0.050  # s

    def prepare(self, parent):
        self.parent = parent  # XrtQook QMainWindow
        iconPath = os.path.join(parent.iconsDir, 'icon-busy.png')
        self.busyPixmap0 = qt.QPixmap(iconPath)
        self.busyVarLim = 0.01, 1.5, 0.2  # min, max, mult
        self.busyVar = self.busyVarLim[0]
        self.busyParticleScales = [1, 1.5, 1.5, 1.5, 1.5]
        self.shouldRedraw = True

    def render(self):
        while self.shouldRedraw:
            pixBusy = qt.QPixmap(self.busyPixmap0.size())
            pixBusy.fill(qt.QColor("#000000ff"))
            painter = qt.QPainter(pixBusy)
            # painter.setRenderHint(qt.QPainter.SmoothPixmapTransform)
            if self.busyVar > self.busyVarLim[1]:
                self.busyVar = self.busyVarLim[0]
            scale = self.busyVar
            painter.scale(scale, scale)
            for sc in self.busyParticleScales:
                painter.scale(sc, sc)
                painter.drawPixmap(32, 32, self.busyPixmap0)
            painter.end()
            self.busyVar += self.busyVar*self.busyVarLim[2]
            icon = qt.QIcon(pixBusy)

            # tab order is unknown:
            tabWidget = self.parent.tabWidget
            for itab in range(tabWidget.count()):
                if tabWidget.tabText(itab) == self.parent.tabNameGlow:
                    break
            else:
                return
            tabWidget.setTabIcon(itab, icon)

            time.sleep(self.BUSYICONDT)  # waiting in a separate thread is ok

    def halt(self):
        self.shouldRedraw = False
