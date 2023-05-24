# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:26:54 2023

@author: Roman Chernikov, Konstantin Klementiev, GPT-4
"""

import sys; sys.path.append(r'..\..\..')
import os
import re
import uuid
import time
import multiprocessing
import numpy as np
import copy
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
    QPushButton, QMenu, QComboBox, QFileDialog,\
    QSplitter, QTreeView, QMessageBox, QProgressBar, QCheckBox
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QBrush,\
    QPixmap, QColor

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
from xrt.backends.raycing import crystalclasses as rxtl
from xrt.backends.raycing.physconsts import CH

try:
    import pyopencl as cl
    import xrt.backends.raycing.myopencl as mcl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

HKLRE = r'\((\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})\)|(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})'


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

    def __init__(self):
        super().__init__()

        try:
            self.setWindowIcon(QIcon(os.path.join(
                    r'..\..\..\xrt\gui\xrtQook\_icons', 'xbcc.png')))
        except:
            # icon not found. who cares?
            pass
        self.statusUpdate.connect(self.update_progress_bar)
        self.layout = QHBoxLayout()
        self.mainSplitter = QSplitter(Qt.Horizontal, self)

        self.proc_nr = max(multiprocessing.cpu_count()-2, 1)

        # Create a QVBoxLayout for the plot and the toolbar
        plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout()

        self.poolsDict = {}

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.ax2 = self.axes.twinx()
        self.axes.set_ylabel('Amplitude', color='k')
        self.axes.tick_params(axis='y', labelcolor='k')
        self.ax2.set_ylabel('Phase', color='b')
        self.ax2.tick_params(axis='y', labelcolor='b')
        self.figure.tight_layout()

        self.plot_lines = {}

        # Add the Matplotlib toolbar to the QVBoxLayout
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)

        # Add the canvas to the QVBoxLayout
        self.plot_layout.addWidget(self.canvas)

        plot_widget.setLayout(self.plot_layout)
        self.mainSplitter.addWidget(plot_widget)

        tree_widget = QWidget(self)
        self.tree_layout = QVBoxLayout()
        self.model = QStandardItemModel()
#        self.model.setHorizontalHeaderLabels(["Name", "Value"])
        self.model.setHorizontalHeaderLabels(["", ""])
        self.model.itemChanged.connect(self.on_tree_item_changed)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self.model)

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

        self.allCrystals = []
        for ck in CRYSTALS.keys():
            if ck in rxtl.__all__:
                self.allCrystals.append(ck)

        self.allColors = []
        self.allIcons = {}
        for color, colorCode in mcolors.TABLEAU_COLORS.items():
            colorName = color.split(":")[-1]
            self.allColors.append(colorName)
            self.allIcons[colorName] = self.create_colored_icon(colorCode)

        self.allGeometries = ['Bragg reflected', 'Bragg transmitted',
                              'Laue reflected', 'Laue transmitted']
        self.allUnits = {'urad': 1e-6,
                         'mrad': 1e-3,
                         'deg': np.pi/180.,
                         'arcsec': np.pi/180./3600.,
                         'eV': 1}
        self.allUnitsStr = {'urad': r' $\mu$rad',
                            'mrad': r' mrad',
                            'deg': r' $\degree$',
                            'arcsec': r' arcsec',
                            'eV': r'eV'}

        self.allBackends = ['auto', 'pyTTE']

        self.xlabel_base_angle = r'$\theta-\theta_B$, '
        self.xlabel_base_e = r'$E - E_B$, '
        self.axes.set_xlabel(self.xlabel_base_angle+r'$\mu$rad')

        if isOpenCL:
            self.allBackends.append('xrtCL FP32')
            self.matCL = mcl.XRT_CL(r'materials.cl',
                                    precisionOpenCL='float32')
            for ctx in self.matCL.cl_ctx:
                for device in ctx.devices:
                    if device.double_fp_config == 63:
                        self.isFP64 = True
            if self.isFP64:
                self.allBackends.append('xrtCL FP64')

        self.add_plot()
        self.resize(1200, 700)
        self.mainSplitter.setSizes([700, 500])
        self.tree_view.resizeColumnToContents(0)

    def create_colored_icon(self, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def add_legend(self):
        if self.axes.get_legend() is not None:
            self.axes.get_legend().remove()
        lgs = []
        lns = []
        for lt in self.plot_lines.values():
            for l in lt:
                if l.get_visible():
                    lns.append(l)
                    lgs.append(l.get_label())
        self.axes.legend(lns, lgs)

    def add_plot(self):
        plot_uuid = uuid.uuid4()
        line_s = Line2D([], [])
        line_p = Line2D([], [], linestyle='--')
        line_phase = Line2D([], [], linestyle=':')
        self.axes.add_line(line_s)
        self.axes.add_line(line_p)
        self.ax2.add_line(line_phase)
        self.plot_lines[plot_uuid] = (line_s, line_p, line_phase)
        previousPlot = None
        plot_item = QStandardItem()
        plot_item.setFlags(plot_item.flags() | Qt.ItemIsEditable)
        plot_item.plot_index = plot_uuid
        plot_item.skipRecalculation = False
        plot_item.prevUnits = "angle"

        cbk_item = QStandardItem()
        self.model.appendRow([plot_item, cbk_item])
        plot_number = plot_item.row()

        initParams = [("Crystal", "Si", self.allCrystals),  # 0
                      ("Geometry", "Bragg reflected",  # 1
                       self.allGeometries),
                      ("hkl", "1, 1, 1", None),  # 2
                      ("Thickness, mm", "1.", None),  # 3
                      ("Asymmetry Angle, deg", "0.", None),  # 4
                      ("Bending Radius, m", "inf", None),  # 5
                      ("Separator", "", None),  # 6
                      ("Energy, eV", "9000", None),  # 7
                      ("Scan Range", "-100, 100", None),  # 8
                      ("Scan Points", "500", None),  # 9
                      ("Scan Units", "urad",
                       list(self.allUnits.keys())),  # 10
                      ("DCM Rocking Curve", "none",
                       ["none", "auto"]),  # 11
                      ("Calculation Backend", "auto",
                       self.allBackends),  # 12
                      ("Separator", "", None),  # 13
                      ("Curve Color", "blue", self.allColors),  # 14
                      ("Curve Type", "sigma pi", ["sigma pi", "sigma", "pi",
                                                  "phase", "hide all"])  # 15
                      ]

        if plot_number > 0:
            previousPlot = self.model.item(plot_number-1)

        for ii, (iname, ival, icb) in enumerate(initParams):
            newValue = ival
            if previousPlot is not None:
                if ii not in [5, 6, 12, 13, 14]:
                    newValue = previousPlot.child(ii, 1).text()
                elif iname == "Curve Color":
                    prevValue = previousPlot.child(ii, 1).text()
                    prevIndex = self.allColors.index(prevValue)
                    if prevIndex + 1 == len(self.allColors):
                        newValue = self.allColors[0]
                    else:
                        newValue = self.allColors[prevIndex+1]

            if iname == "Separator":
                sep_item = QStandardItem()
                sep_item.setFlags(Qt.ItemIsEnabled)
                sep_item.setBackground(QBrush(Qt.lightGray))
                sep_item2 = QStandardItem()
                sep_item2.setFlags(Qt.ItemIsEnabled)
                sep_item2.setBackground(QBrush(Qt.lightGray))
                plot_item.appendRow([sep_item, sep_item2])
            else:
                item_name = QStandardItem(iname)
                item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
                item_value = QStandardItem(str(newValue))
                item_value.setFlags(item_value.flags() | Qt.ItemIsEditable)
                plot_item.appendRow([item_name, item_value])

            if icb is not None:
                cb = QComboBox()
                if iname == "Curve Color":
                    model = QStandardItemModel()
                    cb.setModel(model)
                    for color in icb:
                        item = QStandardItem(color)
                        item.setIcon(self.allIcons[color])
                        model.appendRow(item)
                    plot_item.setIcon(self.allIcons[str(newValue)])
                else:
                    cb.addItems(icb)
                cb.setCurrentText(newValue)
                self.tree_view.setIndexWidget(item_value.index(), cb)
                cb.currentTextChanged.connect(
                        lambda text, item=item_value: item.setText(text))

            if iname == "Crystal":
                plot_name = newValue + "["
            elif iname == "hkl":
                for hkl in parse_hkl(newValue):
                    plot_name += str(hkl)
                plot_name += "] flat"
            elif iname == "Scan Units":
                lims = self.parse_limits(self.get_range_item(plot_item).text())
                convFactor = self.allUnits[newValue]
                self.get_range_item(plot_item).limRads = lims*convFactor

            if iname == "Curve Color":
                line_s.set_color("tab:"+newValue)
                line_p.set_color("tab:"+newValue)
                line_phase.set_color("tab:"+newValue)

        plot_item.setText(plot_name)
        line_s.set_label(plot_name+" $\sigma$")
        line_p.set_label(plot_name+" $\pi$")
        line_phase.set_label(plot_name+" $\phi_\sigma - \phi_\pi$")
        self.add_legend()

        plot_index = self.model.indexFromItem(plot_item)
        self.tree_view.expand(plot_index)

        self.calculate_amps_in_thread(plot_item)

    def get_fwhm(self, xaxis, curve):
        topHalf = np.where(curve >= 0.5*np.max(curve))[0]
        fwhm = np.abs(xaxis[topHalf[0]] - xaxis[topHalf[-1]])
        return fwhm

    def get_energy(self, item):
        try:
            return float(item.child(7, 1).text())
        except ValueError:
            return 9000.

    def get_scan_range(self, item):
        return item.child(8, 1).limRads  # item.child(8, 1).text()

    def get_range_item(self, item):
        return item.child(8, 1)

    def get_scan_points(self, item):
        try:
            return int(float(item.child(9, 1).text()))
        except ValueError:
            return 500

    def get_units(self, item):
        return item.child(10, 1).text()

    def get_units_item(self, item):
        return item.child(10, 1)

    def get_convolution(self, item):
        return item.child(11, 1).text()  #.endswith("true")

    def get_backend(self, item):
        return item.child(12, 1).text()

    def get_color(self, item):
        return item.child(14, 1).text()

    def get_scan_type(self, item):
        return item.child(15, 1).text()

    def on_tree_item_changed(self, item):
        if item.index().column() == 0:
            plot_index = item.plot_index
            if plot_index is not None:
                lines = self.plot_lines[plot_index]
                lines[0].set_label(item.text()+" $\sigma$")
                lines[1].set_label(item.text()+" $\pi$")
                lines[2].set_label(item.text()+" $\phi_\sigma - \phi_\pi$")
                self.add_legend()
                self.canvas.draw()
        else:
            parent = item.parent()
            if parent:
                plot_index = parent.plot_index
                line_s, line_p, line_phase = self.plot_lines[plot_index]
                param_name = parent.child(item.index().row(), 0).text()
                param_value = item.text()
                convFactor = self.allUnits[self.get_units(parent)]

                xaxis, curS, curP = copy.copy(parent.curves)

                if param_name not in ["Scan Range", "Scan Units",
                                      "Curve Color", "DCM Rocking Curve",
                                      "Curve Type"]:
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
                    self.axes.set_xlabel(
                            xlabel_base+self.allUnitsStr[param_value])
                    self.get_range_item(parent).setText("{0}, {1}".format(
                            xLimMin, xLimMax))

                    xaxis = np.linspace(xLimMin, xLimMax,
                                        self.get_scan_points(parent))
                    line_s.set_xdata(xaxis)
                    line_p.set_xdata(xaxis)
                    line_phase.set_xdata(xaxis)
                    self.axes.set_xlim(xLimMin, xLimMax)
                    self.rescale_axes()

                    self.update_all_units(param_value)
                    self.update_title(parent)
                    self.canvas.draw()

                elif param_name.endswith("Color"):
                    line_s.set_color("tab:"+param_value)
                    line_p.set_color("tab:"+param_value)
                    line_phase.set_color("tab:"+param_value)
                    parent.setIcon(self.allIcons[param_value])
                    self.update_title_color(parent)
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
        delete_action.triggered.connect(lambda: self.delete_plot(item))
        context_menu.exec_(self.tree_view.viewport().mapToGlobal(point))

    def delete_plot(self, item):
        plot_index = item.plot_index
        lines = self.plot_lines[plot_index]

        self.axes.lines.remove(lines[0])
        self.axes.lines.remove(lines[1])
        self.ax2.lines.remove(lines[2])
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
            QMessageBox.warning(
                    self, "Warning", "No plot selected for export.")
            return

        selected_index = selected_indexes[0]
        while selected_index.parent().isValid():
            selected_index = selected_index.parent()
        root_item = self.model.itemFromIndex(selected_index)
        theta, curvS, curvP = root_item.curves
        absCurvS = abs(curvS)**2
        absCurvP = abs(curvP)**2

        crystal = root_item.child(0, 1).text()
        geometry = root_item.child(1, 1).text()
        hkl = root_item.child(2, 1).text()
        thck = float(root_item.child(3, 1).text())
        asymmetry = float(root_item.child(4, 1).text())
        radius = root_item.child(5, 1).text()
        radiusStr = radius if radius == "inf" else radius + "m"
        energy = self.get_energy(root_item)
        thetaB = np.degrees(float(root_item.thetaB))

        units = self.get_units(root_item)
        convFactor = self.allUnits[units]
        xaxis = "dE" if units == 'eV' else "dtheta"

        fileName = re.sub(r'[^a-zA-Z0-9_\-.]+', '_', root_item.text())

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save File", fileName,
                "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, "w") as f:
                f.write(f"# Crystal: {crystal}[{hkl}]\tThickness: {thck:.8g}mm\n")
                f.write(f"# Asymmetry: {asymmetry:.8g}deg\tRbend: {radiusStr}\n")
                f.write(f"# Energy: {energy}eV\tTheta_B: {thetaB:.8g}deg\n")
                f.write(f"# Geometry: {geometry}\n\n")
                f.write(f"# {xaxis}({units})\tsigma\tpi\n")
                for i in range(len(theta)):
                    f.write(f"{theta[i]/convFactor:.8g}\t{absCurvS[i]:.8e}\t{absCurvP[i]:.8e}\n")

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
                               precisionOpenCL=precision)
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
        crystal = plot_item.child(0, 1).text()
        geometry = plot_item.child(1, 1).text()
        hkl = plot_item.child(2, 1).text()
        thickness = plot_item.child(3, 1).text()
        asymmetry = plot_item.child(4, 1).text()
        radius = plot_item.child(5, 1).text()

        energy = self.get_energy(plot_item)

        npoints = self.get_scan_points(plot_item)
        units = self.get_units(plot_item)

        backendStr = self.get_backend(plot_item)
        plot_nr = plot_item.row()

        useTT = False
        backend = "xrt"
        precision = "float64"

        if float(radius) == 0:
            radius = "inf"

        if backendStr == "auto":
            if radius == "inf":
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
                    hns0, phi, radius, precision, plot_nr)
            self.amps_calculator.progress.connect(self.update_progress_bar)
            self.amps_calculator.result.connect(self.on_calculation_result)
            self.amps_calculator.start()
        elif backend == "pytte":
            self.t0 = time.time()
            self.statusUpdate.emit(("Calculating on CPU", 0))
            plot_item.curves = np.copy((xaxis,
                                        np.zeros(len(theta),
                                                 dtype=np.complex128),
                                        np.zeros(len(theta),
                                                 dtype=np.complex128)))
            plot_item.curProgress = 0
            geotag = 0 if geometry.startswith('B') else np.pi*0.5
            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
                            thickness=Quantity(float(thickness), 'mm'),
                            debye_waller=1, xrt_crystal=crystalInstance,
                            Rx=Quantity(float(radius), 'm'),
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
                    lambda: self.check_progress(progress_queue))
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

        plot_item = self.model.item(plot_nr)
        plot_item.curves = copy.copy((theta, curS, curP))

        isConvolution = self.get_convolution(plot_item)
        convFactor = self.allUnits[self.get_units(plot_item)]

        line_s, line_p, line_phase = self.plot_lines[plot_item.plot_index]

        line_s.set_xdata(theta/convFactor)
        line_p.set_xdata(theta/convFactor)
        line_phase.set_xdata(theta/convFactor)

        pltCurvPh = np.angle(curS * curP.conj())
        absCurS = abs(curS)**2
        absCurP = abs(curP)**2

        if isConvolution == "auto":
            pltCurvS = np.convolve(absCurS, absCurS, 'same') / curS.sum()
            pltCurvP = np.convolve(absCurP, absCurP, 'same') / curP.sum()
        else:
            pltCurvS = absCurS
            pltCurvP = absCurP

        plot_item.fwhm = self.get_fwhm(theta, pltCurvS)

        line_s.set_ydata(pltCurvS)
        line_p.set_ydata(pltCurvP)
        line_phase.set_ydata(pltCurvPh)

        scanType = self.get_scan_type(plot_item)
        if scanType == "sigma pi":
            line_s.set_visible(True)
            line_p.set_visible(True)
            line_phase.set_visible(False)
        elif scanType == "sigma":
            line_s.set_visible(True)
            line_p.set_visible(False)
            line_phase.set_visible(False)
        elif scanType == "pi":
            line_s.set_visible(False)
            line_p.set_visible(True)
            line_phase.set_visible(False)
        elif scanType == "phase":
            line_s.set_visible(False)
            line_p.set_visible(False)
            line_phase.set_visible(True)
        else:
            line_s.set_visible(False)
            line_p.set_visible(False)
            line_phase.set_visible(False)

        self.axes.set_xlim(min(theta)/convFactor,
                           max(theta)/convFactor)

        self.rescale_axes()
        self.update_title(plot_item)
        self.update_title_color(plot_item)

        self.add_legend()
        self.canvas.draw()

    def on_item_clicked(self, index):
        while index.parent().isValid():
            index = index.parent()

        plot_item = self.model.itemFromIndex(index)
        self.update_title(plot_item)
        self.update_title_color(plot_item)
        self.canvas.draw()

    def update_all_units(self, new_units):
        for index in range(self.model.rowCount()):
            plot_item = self.model.item(index)
            units_item = self.get_units_item(plot_item)
            units_combo = self.tree_view.indexWidget(units_item.index())

            if units_combo.currentText() != new_units:
                units_combo.setCurrentText(new_units)

    def update_title(self, item):
        unitsStr = self.get_units(item)
        convFactor = self.allUnits[unitsStr]
        if hasattr(item, "fwhm"):
            self.axes.set_title(
                r"$\theta_B$ = {0:.3f}{1}, FWHM$_\sigma$ = {2:.3f} {3}".format(
                        np.degrees(item.thetaB),
                        self.allUnitsStr["deg"],
                        item.fwhm/convFactor,
                        self.allUnitsStr[unitsStr]))
        self.canvas.draw()

    def update_title_color(self, item):
        cur_color = self.get_color(item)
        title2 = self.ax2.set_title("-----", loc='left', pad=2, weight='bold')
        title2.set_color("tab:"+cur_color)
        self.canvas.draw()

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
                 alpha, radius, precision, plot_nr):
        super(AmpCalculator, self).__init__()
        self.crystalInstance = crystal
        self.xaxis = xaxis
        self.energy = energy
        self.gamma0 = gamma0
        self.gammah = gammah
        self.hns0 = hns0
        self.alpha = alpha
        self.radius = radius
        self.precision = precision
        self.plot_nr = plot_nr

    def run(self):
        matCL = mcl.XRT_CL(r'materials.cl',
                           precisionOpenCL=self.precision)
        ampS, ampP = self.crystalInstance.get_amplitude_pytte(
                self.energy, self.gamma0, self.gammah, self.hns0,
                ucl=matCL, alphaAsym=self.alpha,
                Ry=float(self.radius)*1000., signal=self.progress)
#        self.result.emit((self.dtheta, abs(ampS)**2, abs(ampP)**2,
#                          self.plot_nr))
        self.result.emit((self.xaxis, ampS, ampP,
                          self.plot_nr))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = PlotWidget()
    main_widget.setWindowTitle("xrt Crystal reflectivity calculator")
    main_widget.show()
    sys.exit(app.exec_())
