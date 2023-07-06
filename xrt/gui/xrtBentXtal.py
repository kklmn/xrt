# -*- coding: utf-8 -*-
u"""
.. _xrtBentXtal:

xrtBentXtal -- a GUI for bent crystal calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to ray tracing applications with perfect and bent crystals, xrt has
a GUI widget xrtBentXtal to calculate reflectivity curves of bent crystals with
utilizing the power of modern GPUs. One can use this widget to conveniently
study the influence of crystal type, bending radius, asymmetry angle, thickness
and other parameters on reflectivity and compare multiple curves side by side.
We include the CPU-based PyTTE code [PyTTE1]_ [PyTTE2]_ for reference and
performance comparison. We also add the possibility to calculate transmitted
amplitudes in Bragg geometry, missing in the original PyTTE, though this mode
only works for CPU-based calculations and is not suitable for ray tracing due
to memory constraints.

.. imagezoom:: _images/xrtBentXtal.png
   :alt: &ensp;xrtBentXtal -- a Qt widget for bent crystal calculations in xrt.
       Shown are two reflectivity curves for a flat version of Si (10, 10, 0)
       crystal (blue) and a bent version of it (orange). In going to higher
       order reflexes the difference in the width of the reflection domain
       becomes increasingly pronounced.

"""

__author__ = "Roman Chernikov, Konstantin Klementiev, GPT-4"
__date__ = "6 Jul 2023"
__version__ = "1.0.0"
__license__ = "MIT license"

import os, sys; sys.path.append(os.path.join(*['..']*2))  # analysis:ignore
import re
import uuid
import time
import multiprocessing
import numpy as np
import copy
from functools import partial
from datetime import datetime

from scipy.interpolate import UnivariateSpline

QtName = "PyQt5"
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
    QPushButton, QMenu, QComboBox, QFileDialog, QToolButton,\
    QSplitter, QTreeView, QMessageBox, QProgressBar, QLabel, QFrame
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QBrush,\
    QPixmap, QColor
import matplotlib as mpl
from matplotlib.backend_tools import ToolBase

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from xrt.backends.raycing.pyTTE_x.elastic_tensors import CRYSTALS
from xrt.backends.raycing.pyTTE_x import TTcrystal, TTscan, Quantity
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy_qt import TakagiTaupin,\
    CalculateAmplitudes  # , integrate_single_scan_step
from xrt.backends.raycing import materials_crystals as rxtl
from xrt.backends.raycing.physconsts import CH
path_to_xrt = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

try:
    import pyopencl as cl
    import xrt.backends.raycing.myopencl as mcl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

HKLRE = r'\((\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})\)|(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})'

targetOpenCL = 'auto'

def parse_hkl(hklstr):
    matches = re.findall(HKLRE, hklstr)
    hklval = []
    for match in matches:
        for group in match:
            if group != '':
                hklval.append(int(group))
    return hklval


def run_calculation(params, progress_queue):
    calculation = CalculateAmplitudes(*params)
    result = calculation.run()
    progress_queue.put((result))


def raise_warning(text, infoText):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setInformativeText(infoText)
    msg.setWindowTitle(text)
    proceed_button = msg.addButton('Proceed', QMessageBox.AcceptRole)
    cancel_button = msg.addButton(QMessageBox.Cancel)
    msg.exec_()

    # Check which button was clicked
    if msg.clickedButton() == proceed_button:
        return True
    elif msg.clickedButton() == cancel_button:
        return False


class PlotWidget(QWidget):
    statusUpdate = Signal(tuple)

    allCrystals = []
    for ck in CRYSTALS.keys():
        if ck in rxtl.__all__:
            allCrystals.append(ck)

    allColors = []
    for color in mcolors.TABLEAU_COLORS.keys():
        colorName = color.split(":")[-1]
        allColors.append(colorName)

    allGeometries = ['Bragg reflected', 'Bragg transmitted',
                     'Laue reflected', 'Laue transmitted']

    allUnits = {'urad': 1e-6,
                'mrad': 1e-3,
                'deg': np.pi/180.,
                'mdeg': 1e-3*np.pi/180.,
                'arcsec': np.pi/180./3600.,
                'eV': 1}

    allUnitsStr = {'urad': r'µrad',
                   'mrad': r'mrad',
                   'deg': r'°',
                   'mdeg': r'm°',
                   'arcsec': r'arcsec',
                   'eV': r'eV'}

    allBackends = ['auto', 'pyTTE']

    allCurves = {'σ': '-', 'π': '--', 'σ*σ': '.-', 'π*π': '*--', 'Δφ': ':'}

    initParams = [
        # (param name, param value, copy from previous, param data (optional))
        ("Separator Structure", "Crystal Structure", False, '#dddddd'),  # 0
        ("Crystal", "Si", True, allCrystals),  # 1
        ("Geometry", "Bragg reflected", True, allGeometries),  # 2
        ("hkl", "1, 1, 1", True),  # 3
        ("Thickness (mm)", "1.", True),  # 4
        ("Asymmetry \u2220 (°)", "0.", True),  # 5
        ("Bending Rm (m)", "inf", False),  # 6
        ("Bending Rs (m)", "inf", False),  # 7
        ("Separator Scan", "Scan", False, '#dddddd'),  # 8
        ("Energy (eV)", "9000", True),  # 9
        ("Scan Range", "-80, 120", True),  # 10
        ("Scan Units", "urad", True, list(allUnits.keys())),  # 11
        ("Scan Points", "500", True),  # 12
        ("Calc Backend", "auto", False, allBackends),  # 13
        ("Separator Plot", "Plot", False, '#dddddd'),  # 14
        ("Curve Color", "blue", False, allColors),  # 15
        ("Curves", ['σ', ], True, allCurves)  # 16
        ]

    def __init__(self):
        super().__init__()

        try:
            self.setWindowIcon(QIcon(
                os.path.join('xrtQook', '_icons', 'xbcc.png')))
        except Exception:
            # icon not found. who cares?
            pass
        self.statusUpdate.connect(self.update_progress_bar)
        self.layout = QHBoxLayout()
        self.mainSplitter = QSplitter(Qt.Horizontal, self)

        self.proc_nr = max(multiprocessing.cpu_count()-2, 1)

        # Create a QVBoxLayout for the plot and the toolbar
        plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout()

        self.allIcons = {}
        for colorName, colorCode in zip(
                self.allColors, mcolors.TABLEAU_COLORS.values()):
            self.allIcons[colorName] = self.create_colored_icon(colorCode)

        self.poolsDict = {}

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.ax2 = self.axes.twinx()
        self.axes.set_ylabel('|Amplitude|²', color='k')
        self.axes.tick_params(axis='y', labelcolor='k')
        self.ax2.set_ylabel('Phase', color='b')
        self.ax2.tick_params(axis='y', labelcolor='b')
        self.figure.tight_layout()

        self.plot_lines = {}

        # Add the Matplotlib toolbar to the QVBoxLayout
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)
        self.aboutButton = QToolButton(self)
        icon = QIcon()
        icon.addPixmap(QPixmap(os.path.join(os.path.dirname(mpl.__file__),
                               'mpl-data', 'images', 'help_large.png')),
                       QIcon.Normal, QIcon.Off)
        # adding icon to the toolbutton
        self.aboutButton.setIcon(icon)
        self.aboutButton.clicked.connect(self.about)
        self.toolbar.addWidget(self.aboutButton)

        # Add the canvas to the QVBoxLayout
        self.plot_layout.addWidget(self.canvas)

        plot_widget.setLayout(self.plot_layout)
        self.mainSplitter.addWidget(plot_widget)

        tree_widget = QWidget(self)
        self.tree_layout = QVBoxLayout()
        self.model = QStandardItemModel()
        self.model.itemChanged.connect(self.on_tree_item_changed)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self.model)

        # self.model.setHorizontalHeaderLabels(["Name", "Value"])
        # self.model.setHorizontalHeaderLabels(["", ""])
        self.tree_view.setHeaderHidden(True)

        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
                self.show_context_menu)
        self.tree_view.clicked.connect(self.on_item_clicked)

        self.add_plot_button = QPushButton("Add curve")
        self.add_plot_button.clicked.connect(self.add_plot)

        self.export_button = QPushButton("Export curve")
        self.export_button.clicked.connect(self.export_curve)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.add_plot_button)
        self.buttons_layout.addWidget(self.export_button)
        self.tree_layout.addLayout(self.buttons_layout)
        self.tree_layout.addWidget(self.tree_view)
        self.progressBar = QProgressBar()
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, 100)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.tree_layout.addWidget(self.progressBar)

        tree_widget.setLayout(self.tree_layout)
        self.mainSplitter.addWidget(tree_widget)
        self.layout.addWidget(self.mainSplitter)
        self.setLayout(self.layout)

        self.xlabel_base_angle = r'$\theta-\theta_B$'
        self.xlabel_base_e = r'$E - E_B$'
        self.axes.set_xlabel('{0} (µrad)'.format(self.xlabel_base_angle))

        if isOpenCL:
            self.allBackends.append('xrtCL FP32')
            self.matCL = mcl.XRT_CL(r'materials.cl', precisionOpenCL='float32',
                                    targetOpenCL=targetOpenCL)
            self.isFP64 = False
            if hasattr(self.matCL, 'cl_ctx'):
                for ctx in self.matCL.cl_ctx:
                    for device in ctx.devices:
                        if device.double_fp_config == 63:
                            self.isFP64 = True
                if ":" in targetOpenCL:
                    self.isFP64 = True
            if self.isFP64:
                self.allBackends.append('xrtCL FP64')

        self.add_plot()
        self.resize(1100, 700)
        self.mainSplitter.setSizes([700, 400])
        self.tree_view.resizeColumnToContents(0)

    def about(self):
        # https://stackoverflow.com/a/69325836/2696065
        def isWin11():
            return True if sys.getwindowsversion().build > 22000 else False

        import platform
        locos = platform.platform(terse=True)
        if 'Linux' in locos:
            try:
                locos = " ".join(platform.linux_distribution())
            except AttributeError:  # no platform.linux_distribution in py3.8
                try:
                    import distro
                    locos = " ".join(distro.linux_distribution())
                except ImportError:
                    print("do 'pip install distro' for a better view of Linux"
                          " distro string")
        elif 'Windows' in locos:
            if isWin11():
                locos = 'Winows 11'
        from xrt.version import __version__ as xrtversion  # analysis:ignore
        strXrt = 'xrt {0} in {1}'.format(xrtversion, path_to_xrt)

        title = "About xrt bentXCalculator"
        txt = """<b>bentXCalculator(Qt)</b> v {0}
            <ul>
            <li>{1[0]}
            <li>{1[1]}
            <li>{1[2]}
            </ul>
            <p>Open source, {2}
            <p>Available on PyPI and GitHub as part of xrt<p>
            <p>Your system:
            <ul>
            <li>{3}
            <li>Python {4}
            <li>Qt {5}, {6} {7}
            <li>{8}
            </ul>""".format(
                __version__, __author__.split(','), __license__,
                locos, platform.python_version(),
                QT_VERSION_STR, QtName, PYQT_VERSION_STR,
                strXrt)

        msg = QMessageBox()
        msg.setStyleSheet("QLabel{min-width: 300px;}")
        msg.setWindowIcon(self.windowIcon())
        msg.setIconPixmap(QPixmap(os.path.join(
            'xrtQook', '_icons', 'xbcc.png')).scaledToHeight(
                256, Qt.SmoothTransformation))
        msg.setText(txt)
        msg.setWindowTitle(title)
        msg.exec_()

    def create_colored_icon(self, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def add_legend(self):
        if self.axes.get_legend() is not None:
            self.axes.get_legend().remove()
        lgs = []
        lns = []
        showPhase = False
        for plots in self.plot_lines.values():
            for line in plots:
                if line.get_visible():
                    lns.append(line)
                    if line.axes is self.ax2:
                        showPhase = True
                    lgs.append(line.get_label())
        leg = self.axes.legend(lns, lgs)
        for text in leg.get_texts():
            if 'Δφ' in text.get_text():
                text.set_color('b')

        self.ax2.set_visible(showPhase)

    def add_plot(self):
        plot_uuid = uuid.uuid4()
        plots = []
        for k, v in self.allCurves.items():
            if v[0] in Line2D.markers:
                kw = dict(linestyle=v[1:], marker=v[0])
            else:
                kw = dict(linestyle=v)
            line = Line2D([], [], label=k, **kw)
            plots.append(line)
            if k == 'Δφ':
                self.ax2.add_line(line)
            else:
                self.axes.add_line(line)
        self.plot_lines[plot_uuid] = plots

        previousPlot = None
        plot_item = QStandardItem()
        plot_item.setFlags(plot_item.flags() | Qt.ItemIsEditable)
        plot_item.plot_index = plot_uuid
        plot_item.skipRecalculation = False
        plot_item.prevUnits = "angle"
        plot_item.fwhms = [None for label in self.allCurves]

        cbk_item = QStandardItem()
        cbk_item.setFlags(Qt.NoItemFlags)
        self.model.appendRow([plot_item, cbk_item])
        plot_number = plot_item.row()

        if plot_number > 0:
            previousPlot = self.model.item(plot_number-1)

        for ii, params in enumerate(self.initParams):
            iname, ival, copyFromPrev = params[0:3]
            idata = params[3] if len(params) > 3 else None
            if previousPlot is not None:
                if copyFromPrev:
                    if iname == "Curves":
                        modelIndex = self.model.indexFromItem(
                            previousPlot.child(ii, 1))
                        w = self.tree_view.indexWidget(modelIndex)
                        ival = w.getActive()
                    else:
                        ival = previousPlot.child(ii, 1).text()
                else:
                    if iname == "Curve Color":
                        prevValue = previousPlot.child(ii, 1).text()
                        prevIndex = self.allColors.index(prevValue)
                        if prevIndex + 1 == len(self.allColors):
                            ival = self.allColors[0]
                        else:
                            ival = self.allColors[prevIndex+1]

            if iname.startswith("Separator"):
                sep_item = QStandardItem()
                sep_item.setText(ival)
                sep_item.setFlags(Qt.NoItemFlags)
                sep_item.setBackground(QBrush(QColor(idata)))
                sep_item2 = QStandardItem()
                sep_item2.setFlags(Qt.NoItemFlags)
                sep_item2.setBackground(QBrush(QColor(idata)))
                plot_item.appendRow([sep_item, sep_item2])
                lab = QLabel()
                self.tree_view.setIndexWidget(sep_item2.index(), lab)
            else:
                item_name = QStandardItem(iname)
                item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
                if iname.startswith("Curves"):
                    item_value = QStandardItem()
                    item_value.setFlags(item_value.flags())
                    w = StateButtons(self.tree_view, list(idata.keys()), ival)
                    w.statesActive.connect(partial(
                        self.on_tree_item_changed, item_value))
                    plot_item.appendRow([item_name, item_value])
                    self.tree_view.setIndexWidget(item_value.index(), w)
                else:
                    item_value = QStandardItem(str(ival))
                    item_value.setFlags(item_value.flags() | Qt.ItemIsEditable)
                    item_value.prevValue = str(ival)
                    plot_item.appendRow([item_name, item_value])

            if isinstance(idata, list):
                cb = QComboBox()
                cb.setMaxVisibleItems(25)
                if iname == "Curve Color":
                    model = QStandardItemModel()
                    cb.setModel(model)
                    for color in idata:
                        item = QStandardItem(color)
                        item.setIcon(self.allIcons[color])
                        model.appendRow(item)
                    plot_item.setIcon(self.allIcons[str(ival)])
                else:
                    cb.addItems(idata)
                cb.setCurrentText(ival)
                self.tree_view.setIndexWidget(item_value.index(), cb)
                # cb.currentTextChanged.connect(
                #         lambda text, item=item_value: item.setText(text))
                cb.currentTextChanged.connect(partial(
                    self.setItemData, iname, item_value))

            if iname == "Crystal":
                plot_name = ival
            elif iname == "hkl":
                nameList = [str(ind) for ind in parse_hkl(ival)]
                plot_name += "[{0}] flat".format(''.join(nameList))
                item_value.prevList = nameList
            elif iname == "Scan Units":
                if ival == 'eV':
                    self.get_range_item(plot_item).limRads = copy.copy(
                        self.get_range_item(previousPlot).limRads)
                else:
                    lims = self.parse_limits(
                        self.get_range_item(plot_item).text())
                    convFactor = self.allUnits[ival]
                    self.get_range_item(plot_item).limRads = lims*convFactor

            if iname == "Curve Color":
                color = "tab:" + ival
                for line in plots:
                    line.set_color(color)

        plot_item.setText(plot_name)
        self.add_legend()

        plot_index = self.model.indexFromItem(plot_item)
        self.tree_view.expand(plot_index)

        self.calculate_amps_in_thread(plot_item)

    def findIndexFromText(self, text):
        for i, initParam in enumerate(self.initParams):
            if initParam[0].startswith(text):
                return i
        print(f'Could not find index of "{text}"!')

    def setItemData(self, iname, item, txt):
        if iname == "Crystal":
            parent = item.parent()
            if item.text() in parent.text():
                newText = parent.text().replace(item.text(), txt)
                parent.setText(newText)
        item.setText(txt)

    def get_fwhm(self, x, y):
        # simple implementation, quantized by dx:
        # topHalf = np.where(y >= 0.5*np.max(y))[0]
        # return np.abs(x[topHalf[0]] - x[topHalf[-1]])

        # a better implementation, weakly dependent on dx size
        try:
            spline = UnivariateSpline(x, y - y.max()*0.5, s=0)
            roots = spline.roots()
            return max(roots) - min(roots)
        except ValueError:
            return 0.

    def get_energy(self, item):
        ind = self.findIndexFromText("Energy")
        try:
            return float(item.child(ind, 1).text())
        except ValueError:
            return float(self.initParams[ind][1])

    def get_range_item(self, item):
        ind = self.findIndexFromText("Scan Range")
        return item.child(ind, 1)

    def get_scan_range(self, item):
        return self.get_range_item(item).limRads

    def get_scan_points(self, item):
        ind = self.findIndexFromText("Scan Points")
        try:
            return int(item.child(ind, 1).text())
        except ValueError:
            return int(self.initParams[ind][1])

    def get_units_item(self, item):
        ind = self.findIndexFromText("Scan Units")
        return item.child(ind, 1)

    def get_units(self, item):
        return self.get_units_item(item).text()

    def get_backend(self, item):
        ind = self.findIndexFromText("Calc")
        return item.child(ind, 1).text()

    def get_color(self, item):
        ind = self.findIndexFromText("Curve Color")
        return item.child(ind, 1).text()

    def get_curve_types(self, item):
        ind = self.findIndexFromText("Curves")
        modelIndex = self.model.indexFromItem(item.child(ind, 1))
        w = self.tree_view.indexWidget(modelIndex)
        layout = w.layout()
        return [layout.itemAt(i).widget().isChecked()
                for i in range(layout.count())]

    def on_tree_item_changed(self, item):
        if item.index().column() == 0:
            plot_index = item.plot_index
            if plot_index is not None:
                self.update_legend(item)
                self.canvas.draw()
        else:
            parent = item.parent()
            if parent:
                plot_index = parent.plot_index
                lines = self.plot_lines[plot_index]
                param_name = parent.child(item.index().row(), 0).text()
                param_value = item.text()
                convFactor = self.allUnits[self.get_units(parent)]

                xaxis, curS, curP = copy.copy(parent.curves)

                if param_name == "hkl":
                    if ',' not in param_value:
                        hklList = list(param_value)
                    else:
                        hklList = [s.strip() for s in param_value.split(',')]
                    try:
                        test = [int(i) for i in hklList]  # analysis:ignore
                        legit = True
                    except Exception:
                        legit = False
                    if len(hklList) == 3 and legit:
                        item.setText(', '.join(hklList))
                        hklName = ''.join(hklList)
                        prevName = ''.join(item.prevList)
                        if prevName in parent.text():
                            newText = parent.text().replace(prevName, hklName)
                            parent.setText(newText)
                        item.prevList = hklList
                    else:
                        item.setText(', '.join(item.prevList))
                elif (param_name.startswith("Thick") or
                      param_name.startswith("Asymm") or
                      param_name.startswith("Energy")):
                    try:
                        test = float(param_value)  # analysis:ignore
                        legit = True
                        item.prevValue = param_value
                    except Exception:
                        legit = False
                        item.setText(item.prevValue)
                        return
                elif param_name.startswith("Scan Range"):
                    try:
                        test = [float(i) for i in param_value.split(',')]
                        legit = True
                        item.prevValue = param_value
                    except Exception:
                        legit = False
                        item.setText(item.prevValue)
                        return
                elif param_name.startswith("Scan Points"):
                    try:
                        test = int(param_value)  # analysis:ignore
                        legit = True
                        item.prevValue = param_value
                    except Exception:
                        legit = False
                        item.setText(item.prevValue)
                        return
                elif param_name.startswith("Bending"):
                    if param_value.startswith('inf'):
                        if 'bent' in parent.text():
                            newText = parent.text().replace('bent', 'flat')
                            parent.setText(newText)
                            item.prevValue = 'inf'
                    else:
                        try:
                            test = float(param_value)  # analysis:ignore
                            legit = True
                            item.prevValue = param_value
                        except Exception:
                            legit = False
                            item.setText(item.prevValue)
                        if 'flat' in parent.text():
                            newText = parent.text().replace('flat', 'bent')
                            parent.setText(newText)

                if param_name not in ["Scan Range", "Scan Units",
                                      "Curve Color", "Curves"]:
                    self.calculate_amps_in_thread(parent)

                elif param_name.endswith("Range"):
                    if parent.skipRecalculation:
                        parent.skipRecalculation = False
                    else:
                        rItem = self.get_range_item(parent)
                        units = self.get_units(parent)
                        convFactor = self.allUnits[units]
                        limits = self.parse_limits(param_value)
                        if units == "eV":
                            rItem.limRads = self.e2theta(limits, parent)
                        else:
                            rItem.limRads = limits*convFactor
                        self.calculate_amps_in_thread(parent)

                elif param_name.endswith("Units"):
                    convFactor = self.allUnits[param_value]
                    limRads = self.get_range_item(parent).limRads
                    if param_value == "eV":
                        xlabel_base = self.xlabel_base_e
                        newLims = self.theta2e(limRads, parent)
                        newUnits = "energy"
                    else:
                        xlabel_base = self.xlabel_base_angle
                        newLims = limRads/convFactor
                        newUnits = "angle"
                    if parent.prevUnits == newUnits:
                        parent.skipRecalculation = True
                    parent.prevUnits = newUnits
                    xLimMin, xLimMax = min(newLims), max(newLims)
                    self.axes.set_xlabel('{0} ({1})'.format(
                            xlabel_base, self.allUnitsStr[param_value]))
                    self.get_range_item(parent).setText(
                        "{0:.4g}, {1:.4g}".format(xLimMin, xLimMax))

                    xaxis = np.linspace(xLimMin, xLimMax,
                                        self.get_scan_points(parent))
                    for line in lines:
                        line.set_xdata(xaxis)
                    self.axes.set_xlim(xLimMin, xLimMax)
                    self.rescale_axes()

                    self.update_all_units(param_value)
                    self.update_legend(parent)
                    self.update_thetaB(parent)
                    self.canvas.draw()

                elif param_name.endswith("Color"):
                    for line in lines:
                        line.set_color("tab:"+param_value)
                    parent.setIcon(self.allIcons[param_value])
                    self.add_legend()
                    self.canvas.draw()

                else:
                    self.on_calculation_result((xaxis, curS, curP,
                                                parent.row()))

    def theta2e(self, theta, plot_item):
        thetaIn = plot_item.thetaB + theta
        E0 = self.get_energy(plot_item)
        d = plot_item.d
        return 0.5*CH/d/np.sin(thetaIn) - E0

    def e2theta(self, energy, plot_item):
        E0 = self.get_energy(plot_item)
        eIn = E0 + energy
        d = plot_item.d
        return np.arcsin(0.5*CH/d/eIn) - plot_item.thetaB

    def show_context_menu(self, point):
        index = self.tree_view.indexAt(point)
        if not index.isValid():
            return

        item = self.model.itemFromIndex(index)
        if item.parent() is not None:  # Ignore child items
            return

        context_menu = QMenu(self.tree_view)
        delete_action = context_menu.addAction("Delete plot")
        delete_action.triggered.connect(partial(self.delete_plot, item))
        context_menu.exec_(self.tree_view.viewport().mapToGlobal(point))

    def delete_plot(self, item):
        plot_index = item.plot_index
        lines = self.plot_lines[plot_index]

        # self.axes.lines has become immutable in matplotlib >= 3.7
        # self.axes.lines.remove(lines[0])
        # self.axes.lines.remove(lines[1])
        # self.ax2.lines.remove(lines[2])
        for line in lines:
            line.remove()

        self.plot_lines.pop(plot_index)

        row = item.row()
        self.model.removeRow(row)
        self.add_legend()

        self.canvas.draw()

    def rescale_axes(self):
        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(True)
        self.ax2.set_autoscalex_on(True)
        self.ax2.set_autoscaley_on(True)
        self.axes.relim()
        self.axes.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

    def export_curve(self):
        selected_indexes = self.tree_view.selectedIndexes()
        if not selected_indexes:
            QMessageBox.warning(self, "Warning", "No plot selected for export")
            return

        selected_index = selected_indexes[0]
        while selected_index.parent().isValid():
            selected_index = selected_index.parent()
        root_item = self.model.itemFromIndex(selected_index)

        ind = self.findIndexFromText("Crystal")
        crystal = root_item.child(ind, 1).text()
        ind = self.findIndexFromText("Geometry")
        geometry = root_item.child(ind, 1).text()
        ind = self.findIndexFromText("hkl")
        hkl = root_item.child(ind, 1).text()
        ind = self.findIndexFromText("Thickness")
        thck = float(root_item.child(ind, 1).text())
        ind = self.findIndexFromText("Asymmetry")
        asymmetry = float(root_item.child(ind, 1).text())
        ind = self.findIndexFromText("Bending Rm")
        Rm = root_item.child(ind, 1).text()
        RmStr = Rm if Rm == "inf" else Rm + "m"
        ind = self.findIndexFromText("Bending Rs")
        Rs = root_item.child(ind, 1).text()
        RsStr = Rs if Rs == "inf" else Rs + "m"
        energy = self.get_energy(root_item)
        thetaB = np.degrees(float(root_item.thetaB))

        units = self.get_units(root_item)
        convFactor = self.allUnits[units]
        xaxis = "dE" if units == 'eV' else "dtheta"

        theta = root_item.curves[0]

        fileName = re.sub(r'[^a-zA-Z0-9_\-.]+', '_', root_item.text())

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save File", fileName,
                "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            lines = self.plot_lines[root_item.plot_index]
            names = self.allCurves.keys()
            outLines, outNames = [theta/convFactor], [f"{xaxis}({units})"]
            for line, name in zip(lines, names):
                if line.get_visible():
                # if True:
                    outLines.append(line.get_ydata())
                    outNames.append(name)
            what = "Transmittivity" if geometry.endswith("mitted") else \
                "Reflectivity"
            now = datetime.now()
            nowStr = now.strftime("%d/%m/%Y %H:%M:%S")
            header = \
                f"{what} calculated by xrt bentXCalculator on {nowStr}\n"\
                f"Crystal: {crystal}[{hkl}]\tThickness: {thck:.8g}mm\n"\
                f"Asymmetry: {asymmetry:.8g}°\tRm: {RmStr}\tRs: {RsStr}\n"\
                f"Energy: {energy}eV\tθ_B: {thetaB:.8g}°\n"\
                f"Geometry: {geometry}\n"
            header += "\t".join(outNames)
            np.savetxt(file_name, np.array(outLines).T, fmt='%#.7g',
                       delimiter='\t', header=header, encoding='utf-8')

    def calculate_amplitudes(self, crystal, geometry, hkl, thickness,
                             asymmetry,  radius, energy, npoints, limits,
                             backendStr):  # Left here for debug purpose only
        useTT = False
        if backendStr == "auto":
            if radius == "inf":
                backend = "xrt"
            elif geometry.startswith("B") and geometry.endswith("mitted"):
                backend = "pytte"
            elif isOpenCL:
                useTT = True
                backend = "xrtCL"
                precision = "float64"
                if geometry.startswith("B"):
                    precision = "float32"
            else:
                backend = "pytte"
        elif backendStr == "pyTTE":
            backend = "pytte"
        elif backendStr == "xrtCL FP32":
            backend = "xrtCL"
            precision = "float32"
            useTT = True
        elif backendStr == "xrtCL FP64":
            backend = "xrtCL"
            precision = "float64"
            useTT = True
        else:
            return

        hklList = parse_hkl(hkl)
        crystalClass = getattr(rxtl, crystal)
        crystalInstance = crystalClass(hkl=hklList, t=float(thickness),
                                       geom=geometry,
                                       useTT=useTT)

        theta0 = crystalInstance.get_Bragg_angle(energy)
        alpha = np.radians(float(asymmetry))

        if limits is None:
            theta = np.linspace(-100, 100, npoints)*1e-6 + theta0
        else:
            theta = np.linspace(limits[0], limits[-1], npoints) + theta0

        if geometry.startswith("B"):
            gamma0 = -np.sin(theta+alpha)
            gammah = np.sin(theta-alpha)
        else:
            gamma0 = -np.cos(theta+alpha)
            gammah = -np.cos(theta-alpha)
        hns0 = np.sin(alpha)*np.cos(theta+alpha) -\
            np.cos(alpha)*np.sin(theta+alpha)

        if backend == "xrtCL":
            matCL = mcl.XRT_CL(r'materials.cl',
                               precisionOpenCL=precision,
                               targetOpenCL=targetOpenCL)
            ampS, ampP = crystalInstance.get_amplitude_pytte(
                    energy, gamma0, gammah, hns0, ucl=matCL, alphaAsym=alpha,
                    Ry=float(radius)*1000.)
#        elif backend == "pytte":
#            geotag = 0 if geometry.startswith('B') else np.pi*0.5
#            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
#                            thickness=Quantity(float(thickness), 'mm'),
#                            debye_waller=1, xrt_crystal=crystalInstance,
#                            Rx=Quantity(float(radius), 'm'),
#                            asymmetry=Quantity(alpha+geotag, 'rad'))
#            tts = TTscan(constant=Quantity(E, 'eV'),
#                         scan=Quantity(theta-theta0, 'rad'),
#                         polarization='sigma')
#            amps_calculator = TakagiTaupin(ttx, tts)
#            integrationParams = amps_calculator.prepare_arrays()

#            scan_tt_s = TakagiTaupin(ttx, tts)
#            scan_tt_p = TakagiTaupin(ttx, ttp)
#            scan_vector, Rs, Ts, ampS = scan_tt_s.run()
#            scan_vector, Rp, Tp, ampP = scan_tt_p.run()
        else:
            ampS, ampP = crystalInstance.get_amplitude(
                    energy, gamma0, gammah, hns0)
#        return theta - theta0, abs(ampS)**2, abs(ampP)**2
        return theta - theta0, ampS, ampP

#    def calculate_amps_in_thread(self, crystal, geometry, hkl, thickness,
#                                 asymmetry, radius, energy, npoints, limits,
#                                 backendStr, plot_nr):
    def calculate_amps_in_thread(self, plot_item):
        ind = self.findIndexFromText("Crystal")
        crystal = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("Geometry")
        geometry = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("hkl")
        hkl = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("Thickness")
        thickness = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("Asymmetry")
        asymmetry = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("Bending Rm")
        Rm = plot_item.child(ind, 1).text()
        ind = self.findIndexFromText("Bending Rs")
        Rs = plot_item.child(ind, 1).text()

        energy = self.get_energy(plot_item)

        npoints = self.get_scan_points(plot_item)
        units = self.get_units(plot_item)

        backendStr = self.get_backend(plot_item)
        plot_nr = plot_item.row()

        useTT = False
        backend = "xrt"
        precision = "float64"

        if float(Rm) == 0:
            Rm = "inf"
        if float(Rs) == 0:
            Rs = "inf"

        if backendStr == "auto":
            if Rm == Rs == "inf":
                backend = "xrt"
            elif geometry == "Bragg transmitted":
                backend = "pytte"
            elif isOpenCL:
                useTT = True
                backend = "xrtCL"
                if self.isFP64:
                    precision = "float64"
                else:
                    precision = "float32"
                if geometry.startswith("B"):
                    precision = "float32"
            else:
                backend = "pytte"
        elif backendStr == "pyTTE":
            backend = "pytte"
            if not raise_warning(
                    "Long Processing Time",
                    "The selected calculation mode may take a significant "
                    "amount of time to complete."):
                self.statusUpdate.emit(("Calculation cancelled", 100))
                return
        elif backendStr == "xrtCL FP32":
            self.statusUpdate.emit(("Calculating on GPU", 0))
            backend = "xrtCL"
            precision = "float32"
            useTT = True
            if geometry.startswith("Laue"):
                if not raise_warning(
                        "Insifficient Precision",
                        "Calculating Laue in FP32 may cause problems with "
                        "convergence"):
                    self.statusUpdate.emit(("Calculation cancelled", 100))
                    return
        elif backendStr == "xrtCL FP64":
            self.statusUpdate.emit(("Calculating on GPU", 0))
            backend = "xrtCL"
            precision = "float64"
            useTT = True
        else:
            return

        hklList = parse_hkl(hkl)
        crystalClass = getattr(rxtl, crystal)
        crystalInstance = crystalClass(hkl=hklList, t=float(thickness),
                                       geom=geometry,
                                       useTT=useTT)

        theta0 = crystalInstance.get_Bragg_angle(energy)
        plot_item.thetaB = theta0
        plot_item.d = getattr(crystalInstance, 'd')
        phi = np.radians(float(asymmetry))

        if units == "eV":
            tLimits = self.get_scan_range(plot_item)
            limits = self.theta2e(tLimits, plot_item)
            xenergy = np.linspace(
                    min(limits), max(limits), npoints) + energy
            theta = theta0 * np.ones_like(xenergy)
            xaxis = xenergy - energy
        else:
            limits = self.get_scan_range(plot_item)
            xenergy = energy
            theta = np.linspace(
                    min(limits), max(limits), npoints) + theta0
            xaxis = theta - theta0

        if geometry.startswith("B"):
            gamma0 = -np.sin(theta+phi)
            gammah = np.sin(theta-phi)
        else:
            gamma0 = -np.cos(theta+phi)
            gammah = -np.cos(theta-phi)
        hns0 = np.sin(phi)*np.cos(theta+phi) -\
            np.cos(phi)*np.sin(theta+phi)

        if backend == "xrtCL" and geometry == "Bragg transmitted":
            self.statusUpdate.emit(
                    ("Bragg transmitted not supported in OpenCL", 100))
            return

        if backend == "xrtCL":
            self.amps_calculator = AmpCalculator(
                    crystalInstance, xaxis, xenergy, gamma0, gammah,
                    hns0, phi, Rm, Rs, precision, plot_nr)
            self.amps_calculator.progress.connect(self.update_progress_bar)
            self.amps_calculator.result.connect(self.on_calculation_result)
            self.amps_calculator.start()
        elif backend == "pytte":
            self.t0 = time.time()
            self.statusUpdate.emit(("Calculating on CPU", 0))
            plot_item.curves = copy.copy((xaxis,
                                         np.zeros(len(theta),
                                                  dtype=np.complex128),
                                         np.zeros(len(theta),
                                                  dtype=np.complex128)))
            plot_item.curProgress = 0
            geotag = 0 if geometry.startswith('B') else np.pi*0.5
            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
                            thickness=Quantity(float(thickness), 'mm'),
                            debye_waller=1, xrt_crystal=crystalInstance,
                            Rx=Quantity(float(Rm), 'm'),
                            Ry=Quantity(float(Rs), 'm'),
                            asymmetry=Quantity(phi+geotag, 'rad'))
            if units == 'eV':
                tts = TTscan(constant=Quantity(theta0, 'rad'),
                             scan=Quantity(xaxis, 'eV'),
                             polarization='sigma')
            else:
                tts = TTscan(constant=Quantity(energy, 'eV'),
                             scan=Quantity(xaxis, 'rad'),
                             polarization='sigma')
            isRefl = geometry.endswith("flected")
            amps_calculator = TakagiTaupin(ttx, tts)
            integrationParams = amps_calculator.prepare_arrays()

            progress_queue = multiprocessing.Manager().Queue()
            self.poolsDict[plot_nr] = multiprocessing.Pool(
                    processes=self.proc_nr)
            self.tasks = []
            a0i = 8 if units == 'eV' else 6
            for step in range(npoints):
                args = integrationParams[:a0i]
                args += [vec_arg[step] for vec_arg in integrationParams[a0i:]]
                args += [isRefl, step, plot_nr]

                task = self.poolsDict[plot_nr].apply_async(
                        run_calculation, args=(args, progress_queue))
                self.tasks.append(task)

            self.timer = QTimer()
            self.timer.timeout.connect(
                partial(self.check_progress, progress_queue))
            self.timer.start(200)  # Adjust the interval as needed
        else:
            ampS, ampP = crystalInstance.get_amplitude(
                    xenergy, gamma0, gammah, hns0)
            self.statusUpdate.emit(("Ready", 100))
            self.on_calculation_result(
                    (xaxis, ampS, ampP, plot_nr))

    def check_progress(self, progress_queue):
        progress = None
        while not progress_queue.empty():
            progress = 1
            xp, aS, aP, plot_nr = progress_queue.get()
            plot_item = self.model.item(plot_nr)
            plot_item.curProgress += 1
            theta, curS, curP = plot_item.curves
            curS[xp] = aS
            curP[xp] = aP

        if progress is not None:
            self.on_calculation_result((theta, curS, curP, plot_nr))
            progressPercentage = 100*plot_item.curProgress/float(len(theta))
            self.statusUpdate.emit(("Calculating on CPU",
                                    progressPercentage))
            if plot_item.curProgress >= len(theta):
                self.statusUpdate.emit((
                        "Calculation completed in {:.3f}s".format(
                                time.time()-self.t0), 100))
                self.timer.stop()
                self.poolsDict[plot_nr].close()
#        if not any(task.ready() is False for task in self.tasks):
#            self.statusUpdate.emit(("Calculation completed in {:.3f}s".format(
#                    time.time()-self.t0), 100))

    def parse_limits(self, limstr):
        return np.array([float(pp) for pp in limstr.split(',')])

    def on_calculation_result(self, res_tuple):
        theta, curS, curP, plot_nr = res_tuple
        curS2 = abs(curS)**2
        curP2 = abs(curP)**2

        plot_item = self.model.item(plot_nr)
        plot_item.curves = copy.copy((theta, curS, curP))
        unitsStr = self.get_units(plot_item)
        convFactor = self.allUnits[self.get_units(plot_item)]
        curveTypes = self.get_curve_types(plot_item)
        lines = self.plot_lines[plot_item.plot_index]

        fwhms = []
        for curveType, line, label in zip(curveTypes, lines, self.allCurves):
            line.set_xdata(theta/convFactor)
            if label == 'σ':
                ydata = curS2
            elif label == 'π':
                ydata = curP2
            elif label == 'σ*σ':
                ydata = np.convolve(curS2, curS2, 'same') / curS2.sum()
            elif label == 'π*π':
                ydata = np.convolve(curP2, curP2, 'same') / curP2.sum()
            elif label == 'Δφ':
                ydata = np.angle(curS * curP.conj())

            if label != 'Δφ':
                fwhm = self.get_fwhm(theta, ydata)
            else:
                fwhm = None
            fwhms.append(fwhm)
            if fwhm is not None:
                unit = self.allUnitsStr[unitsStr]
                sp = '' if unit == '°' else ' '
                t = '' if fwhm is None else\
                    ": {0:#.3g}{1}{2}".format(fwhm/convFactor, sp, unit)
                line.set_label("{0} {1}{2}".format(plot_item.text(), label, t))

            line.set_ydata(ydata)
            line.set_visible(curveType)
        plot_item.fwhms = fwhms
        self.axes.set_xlim(min(theta)/convFactor, max(theta)/convFactor)

        self.rescale_axes()
        self.update_thetaB(plot_item)

        self.add_legend()
        self.canvas.draw()

    def on_item_clicked(self, index):
        while index.parent().isValid():
            index = index.parent()  # of plot_item
        if index.column() == 1:  # the empty cell near the plot title
            return
        plot_item = self.model.itemFromIndex(index)
        self.update_thetaB(plot_item)
        self.canvas.draw()

    def update_all_units(self, new_units):
        for index in range(self.model.rowCount()):
            plot_item = self.model.item(index)
            units_item = self.get_units_item(plot_item)
            units_combo = self.tree_view.indexWidget(units_item.index())

            if units_combo.currentText() != new_units:
                units_combo.setCurrentText(new_units)

    def update_legend(self, item):
        plot_index = item.plot_index
        for line, label, fwhm in zip(
                self.plot_lines[plot_index], self.allCurves.keys(),
                item.fwhms):
            convFactor = self.allUnits[self.get_units(item)]
            unitsStr = self.get_units(item)
            unit = self.allUnitsStr[unitsStr]
            sp = '' if unit == '°' else ' '
            tt = '' if fwhm is None else\
                ": {0:#.3g}{1}{2}".format(fwhm/convFactor, sp, unit)
            line.set_label("{0} {1}{2}".format(item.text(), label, tt))
        self.add_legend()

    def update_thetaB(self, item):
        ind = self.findIndexFromText("Separator Plot")
        modelIndex = self.model.indexFromItem(item.child(ind, 1))
        self.tree_view.indexWidget(modelIndex).setText(
            '&nbsp;θ<sub>B</sub> = {0:.3f}°'.format(np.degrees(item.thetaB)))

    def update_progress_bar(self, dataTuple):
        if int(dataTuple[1]) < 100:
            updStr = r"{0}, {1}% done".format(dataTuple[0], int(dataTuple[1]))
        else:
            updStr = r"{}".format(dataTuple[0])
        self.progressBar.setValue(int(dataTuple[1]))
        self.progressBar.setFormat(updStr)


class AmpCalculator(QThread):
    progress = Signal(tuple)
    result = Signal(tuple)

    def __init__(self, crystal, xaxis, energy, gamma0, gammah, hns0,
                 alpha, Rm, Rs, precision, plot_nr):
        super(AmpCalculator, self).__init__()
        self.crystalInstance = crystal
        self.xaxis = xaxis
        self.energy = energy
        self.gamma0 = gamma0
        self.gammah = gammah
        self.hns0 = hns0
        self.alpha = alpha
        self.Rm = Rm
        self.Rs = Rs
        self.precision = precision
        self.plot_nr = plot_nr

    def run(self):
        matCL = mcl.XRT_CL(r'materials.cl',
                           precisionOpenCL=self.precision,
                           targetOpenCL=targetOpenCL)
        ampS, ampP = self.crystalInstance.get_amplitude_pytte(
                self.energy, self.gamma0, self.gammah, self.hns0,
                ucl=matCL, alphaAsym=self.alpha, autoLimits=False,
                Ry=float(self.Rm)*1000.,
                Rx=float(self.Rs)*1000.,
                signal=self.progress)
#        self.result.emit((self.dtheta, abs(ampS)**2, abs(ampP)**2,
#                          self.plot_nr))
        self.result.emit((self.xaxis, ampS, ampP,
                          self.plot_nr))


class StateButtons(QFrame):
    statesActive = Signal(list)

    def __init__(self, parent, names, active=None):
        """
        *names*: a list of any objects that will be displayed as str(object),

        *active*: a subset of names that will be displayed as checked,

        The signal *statesActive* is emitted on pressing a button. It sends a
        list of selected names, as a subset of *names*.
        """

        super().__init__(parent)
        self.names = names
        self.buttons = []
        layout = QHBoxLayout()
        styleSheet = """
        QPushButton {
            border-style: outset;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightsalmon;}
        QPushButton:checked {
            border-style: inset;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightgreen;}
        QPushButton:hover {
            border-style: solid;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightblue;}
        """
        for name in names:
            strName = str(name)
            but = QPushButton(strName)
            but.setCheckable(True)

            bbox = but.fontMetrics().boundingRect(strName)
            but.setFixedSize(bbox.width()+12, bbox.height()+4)
            # but.setToolTip("go to the key frame")
            but.clicked.connect(self.buttonClicked)
            but.setStyleSheet(styleSheet)

            self.buttons.append(but)
            layout.addWidget(but)
        self.setLayout(layout)

        self.setActive(active)

    def getActive(self):
        return [name for (button, name) in
                zip(self.buttons, self.names) if button.isChecked()]

    def setActive(self, active):
        if not isinstance(active, (list, tuple)):
            return
        for button, name in zip(self.buttons, self.names):
            button.setChecked(name in active)

    def buttonClicked(self, checked):
        self.statesActive.emit(self.getActive())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = PlotWidget()
    main_widget.setWindowTitle("xrt Crystal reflectivity calculator")
    main_widget.show()
    sys.exit(app.exec_())
