# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 18:01:38 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

from ...commons import qt  # analysis:ignore
from ...xrtGlow import ConfigurablePlotWidget  # analysis:ignore
from ....backends import raycing  # analysis:ignore


try:
    QWebView = qt.QtWeb.QWebView
except AttributeError:
    # QWebKit deprecated in Qt 5.7
    # The idea and partly the code of the compatibility fix is borrowed from
    # spyderlib.widgets.browser
    class WebPage(qt.QtWeb.QWebEnginePage):
        """
        Web page subclass to manage hyperlinks for WebEngine

        Note: This can't be used for WebKit because the
        acceptNavigationRequest method has a different
        functionality for it.
        """
        linkClicked = qt.Signal(qt.QUrl)
        linkDelegationPolicy = 2

        def setLinkDelegationPolicy(self, policy):
            self.linkDelegationPolicy = policy

        def acceptNavigationRequest(self, url, navigation_type, isMainFrame):
            """
            Overloaded method to handle links ourselves
            """
            strURL = str(url.toString())
            if strURL.endswith('png'):
                return False
            elif strURL.startswith('file'):
                if strURL.endswith('tutorial.html') or\
                        strURL.endswith('tutorial'):
                    self.linkClicked.emit(url)
                    return False
                else:
                    return True
            else:
                self.linkClicked.emit(url)
                return False

    class QWebView(qt.QtWeb.QWebEngineView):
        """Web view"""

        def __init__(self):
            qt.QtWeb.QWebEngineView.__init__(self)
            web_page = WebPage(self)
            self.setPage(web_page)


class TreeViewEx(qt.QTreeView):
    objDoubleClicked = qt.Signal(str)

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.pos())

        if index.isValid():
            column = index.column()
            row = index.row()
            item = self.model().itemFromIndex(index)
            if column == 1:
                if str(item.text()).startswith("Instance"):
                    objIndex = index.sibling(row, 0)
                    objItem = self.model().itemFromIndex(objIndex)
                    objid = str(objItem.data(qt.Qt.UserRole))
                    if raycing.is_valid_uuid(objid):
                        self.objDoubleClicked.emit(objid)
                        return
                elif str(item.text()).startswith("Preview"):
                    objIndex = index.sibling(row, 0)
                    objItem = self.model().itemFromIndex(objIndex)
                    objid = str(objItem.text())
                    self.objDoubleClicked.emit(objid)
                    return
        super().mouseDoubleClickEvent(event)


class QDockWidgetNoClose(qt.QDockWidget):  # ignores Alt+F4 on undocked widget
    def closeEvent(self, evt):
        evt.setAccepted(not evt.spontaneous())

    def changeWindowFlags(self, evt):
        if self.isFloating():
            # The dockWidget will automatically regain it's Qt::widget flag
            # when it becomes docked again
            self.setWindowFlags(qt.Qt.Window |
                                qt.Qt.CustomizeWindowHint |
                                qt.Qt.WindowMaximizeButtonHint)
            # setWindowFlags calls setParent() when changing the flags for a
            # window, causing the widget to be hidden, so:
            self.show()

            # Custom title bar:
            self.titleBar = qt.QWidget(self)
            self.titleBar.setAutoFillBackground(True)
            # self.titleBar.setStyleSheet(
            #     "QWidget {font: bold; font-size: " + str(fontSize) + "pt;}")

            # pal = self.titleBar.palette()
            # pal.setColor(qt.QPalette.Window, qt.QColor("lightgray"))
            # self.titleBar.setPalette(pal)
            height = qt.QApplication.style().pixelMetric(
                qt.QStyle.PM_TitleBarHeight)
            self.titleBar.setMaximumHeight(height)
            layout = qt.QHBoxLayout()
            self.titleBar.setLayout(layout)

            bSize = height // 2
            self.buttonSize = qt.QSize(bSize, bSize)
            self.titleIcon = qt.QLabel()
            if hasattr(self, 'dockIcon'):
                self.titleIcon.setPixmap(self.dockIcon.pixmap(self.buttonSize))
            self.titleIcon.setVisible(True)
            layout.addWidget(self.titleIcon, 0)
            self.title = qt.QLabel(self.windowTitle())
            layout.addWidget(self.title, 0)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.addStretch()

            self.dockButton = qt.QToolButton(self)
            self.dockButton.setIcon(qt.QApplication.style().standardIcon(
                qt.QStyle.SP_ToolBarVerticalExtensionButton))
            self.dockButton.setMaximumSize(self.buttonSize)
            self.dockButton.setAutoRaise(True)
            self.dockButton.clicked.connect(self.toggleFloating)
            self.dockButton.setToolTip('dock into the main window')
            layout.addWidget(self.dockButton, 0)

            self.maxButton = qt.QToolButton(self)
            self.maxButton.setIcon(qt.QApplication.style().standardIcon(
                qt.QStyle.SP_TitleBarMaxButton))
            self.maxButton.setMaximumSize(self.buttonSize)
            self.maxButton.setAutoRaise(True)
            self.maxButton.clicked.connect(self.toggleMax)
            layout.addWidget(self.maxButton, 0)

            self.setTitleBarWidget(self.titleBar)
        else:
            self.setTitleBarWidget(None)
            self.parent().setTabIcons()

    def toggleFloating(self):
        self.setFloating(not self.isFloating())
        self.raise_()

    def toggleMax(self):
        if self.isMaximized():
            self.showNormal()
            self.maxButton.setIcon(qt.QApplication.style().standardIcon(
                qt.QStyle.SP_TitleBarMaxButton))
        else:
            self.showMaximized()
            self.maxButton.setIcon(qt.QApplication.style().standardIcon(
                qt.QStyle.SP_TitleBarNormalButton))


class PlotViewer(qt.QDialog):
    def __init__(self, plotProps, parent=None, viewOnly=False, beamLine=None,
                 plotId=None):
        super().__init__(parent)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Live Plot Builder")
        hiddenProps = {'_object', 'fluxUnit'}

        self.dynamicPlot = ConfigurablePlotWidget(
                plotProps, parent, viewOnly, beamLine,
                plotId, hiddenProps=hiddenProps)
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.dynamicPlot)

    def update_plot_param(self, paramTuple):
        self.dynamicPlot.update_plot_param(paramTuple)

    def update_beam(self, beamTag):
        self.dynamicPlot.update_beam(beamTag)
