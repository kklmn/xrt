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
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
    QPushButton, QMenu, QComboBox, QFileDialog,\
    QSplitter, QTreeView, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QBrush
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from xrt.backends.raycing.pyTTE_x.elastic_tensors import CRYSTALS
from xrt.backends.raycing.pyTTE_x import TTcrystal, TTscan, Quantity
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy import TakagiTaupin
from xrt.backends.raycing import crystalclasses as rxtl

try:
    import pyopencl as cl
    import xrt.backends.raycing.myopencl as mcl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

HKLRE = r'\((\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})\)|(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})'
SCAN_LENGTH = 1000


def parse_hkl(hklstr):
    matches = re.findall(HKLRE, hklstr)
    hklval = []
    for match in matches:
        for group in match:
            if group != '':
                hklval.append(int(group))
    return hklval

class PlotWidget(QWidget):
#    statusUpdate = Signal(tuple)

    def __init__(self):
        super().__init__()

        try:
            self.setWindowIcon(QIcon(os.path.join(
                    r'..\..\..\xrt\gui\xrtQook\_icons', 'xbcc.png')))
        except:
            # icon not found. who cares?
            pass
#        self.statusUpdate.connect(self.updateProgressBar)
        self.layout = QHBoxLayout()
        self.mainSplitter = QSplitter(Qt.Horizontal, self)

        # Create a QVBoxLayout for the plot and the toolbar
        plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
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
        self.model.setHorizontalHeaderLabels(["Name", "Value"])
        self.model.itemChanged.connect(self.on_tree_item_changed)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self.model)

        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
                self.show_context_menu)

        self.add_plot_button = QPushButton("Add curve")
        self.add_plot_button.clicked.connect(self.add_plot)

        self.export_button = QPushButton("Export curve")
        self.export_button.clicked.connect(self.export_curve)

        # Create a QHBoxLayout for the buttons and add them to the main layout
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

#        if isOpenCL:
#            self.matCL = mcl.XRT_CL(r'materials.cl',
#                                    # precisionOpenCL='float32',
#                                    precisionOpenCL='float64'
#                                    )
#        else:
#            self.matCL = None

        self.allCrystals = []
        for ck in CRYSTALS.keys():
            if not ck.startswith("prototype"):
                self.allCrystals.append(ck)

        self.allColors = []
        for color in mcolors.TABLEAU_COLORS.keys():
            self.allColors.append(color.split(":")[-1])

        self.allGeometries = ['Bragg reflected', 'Bragg transmitted',
                              'Laue reflected', 'Laue transmitted']
        self.allUnits = {'urad': 1e-6,
                         'mrad': 1e-3,
                         'deg': np.pi/180.,
                         'arcsec': np.pi/180./3600.}
        self.allUnitsStr = {'urad': r' $\mu$rad',
                            'mrad': r' mrad',
                            'deg': r' $\degree$',
                            'arcsec': r' arcsec'}

        self.allBackends = ['auto', 'pyTTE']
        if isOpenCL:
            self.allBackends.append('xrtCL FP32')
            self.allBackends.append('xrtCL FP64')

        self.add_plot()
        self.resize(1200, 700)
        self.mainSplitter.setSizes([700, 500])
        self.tree_view.resizeColumnToContents(0)

    def add_plot(self):
        theta, curS, curP = self.calculate_amplitudes(
                            "Si",  # Crystal
                            "Bragg reflected",  # Geometry
                            "1, 1, 1",  # hkl
                            "1.",  # Thickness
                            "0.",  # Asymmetry
                            "inf",  # Bending radius
                            "9000",  # Energy
                            np.array([-100, 100])*1e-6,
                            "auto")
        line_s = Line2D(theta*1e6, curS)
        line_p = Line2D(theta*1e6, curP, linestyle='--')
        self.axes.add_line(line_s)
        self.axes.add_line(line_p)
        self.xlabel_base = r'$\theta-\theta_B$, '
        self.axes.set_xlabel(self.xlabel_base+r'$\mu$rad')
        plot_uuid = uuid.uuid4()
        self.plot_lines[plot_uuid] = (line_s, line_p)
        plot_name = "Si111 flat"
        line_s.set_label(plot_name+" $\sigma$")
        line_p.set_label(plot_name+" $\pi$")
        line_s.set_color('blue')
        line_p.set_color('blue')
        lgs = []
        for lt in self.plot_lines.values():
            for l in lt:
                lgs.append(l.get_label())

        plot_item = QStandardItem(plot_name)
        plot_item.setFlags(plot_item.flags() | Qt.ItemIsEditable)
        plot_item.curves = np.copy((theta, curS, curP))
        plot_item.plot_index = plot_uuid
        self.model.appendRow(plot_item)

        for iname, ival, icb in [("Crystal", "Si", self.allCrystals),  # 0
                                 ("Geometry", "Bragg reflected",  # 1
                                  self.allGeometries),
                                 ("hkl", "1, 1, 1", None),  # 2
                                 ("Thickness, mm", "1.", None),  # 3
                                 ("Asymmetry Angle, deg", "0.", None),  # 4
                                 ("Bending Radius, m", "inf", None),  # 5
                                 ("Separator", "", None),  # 6
                                 ("Energy, eV", "9000", None),  # 7
                                 ("Scan Range", "-100, 100", None),  # 8
                                 ("Scan Units", "urad",
                                  list(self.allUnits.keys())),  # 9
                                 ("Detuning Curve", "false",
                                  ["true", "false"]),  # 10
                                 ("Calculation Backend", "auto",
                                  self.allBackends),  # 11
                                 ("Separator", "", None),  # 12
                                 ("Curve Color", "blue", self.allColors)  # 13
                                 ]:

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
                item_value = QStandardItem(str(ival))
                item_value.setFlags(item_value.flags() | Qt.ItemIsEditable)
                plot_item.appendRow([item_name, item_value])

            if iname == "Scan Range":
                item_value.limRads = np.array([-100, 100])*1e-6

            if icb is not None:
                cb = QComboBox()
                cb.addItems(icb)
                cb.setCurrentText(ival)
                self.tree_view.setIndexWidget(item_value.index(), cb)
                cb.currentTextChanged.connect(
                        lambda text, item=item_value: item.setText(text))

        plot_index = self.model.indexFromItem(plot_item)
        self.tree_view.expand(plot_index)

        self.axes.legend(lgs)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw()

    def update_all_units(self, new_units):
        for index in range(self.model.rowCount()):
            plot_item = self.model.item(index)
            units_item = plot_item.child(9, 1)
            units_combo = self.tree_view.indexWidget(units_item.index())

            if units_combo.currentText() != new_units:
                units_combo.setCurrentText(new_units)

    def on_tree_item_changed(self, item):
        if item.index().column() == 0:
            plot_index = item.plot_index
            if plot_index is not None:
                lines = self.plot_lines[plot_index]
                lines[0].set_label(item.text()+" $\sigma$")
                lines[1].set_label(item.text()+" $\pi$")
                lgs = []
                for lt in self.plot_lines.values():
                    for l in lt:
                        lgs.append(l.get_label())
                self.axes.legend(lgs)
                self.canvas.draw()
        else:
            parent = item.parent()
            if parent:
                plot_index = parent.plot_index
                line_s, line_p = self.plot_lines[plot_index]
                param_name = parent.child(item.index().row(), 0).text()
                param_value = item.text()
                convFactor = self.allUnits[parent.child(9, 1).text()]

                allParams = [parent.child(i, 1).text() for i in range(6)]
                allParams.append(parent.child(7, 1).text())  # Energy
                allParams.append(
                        self.parse_limits(
                                parent.child(8, 1).text())*convFactor)
                allParams.append(parent.child(11, 1).text())  # Backend

                theta, curS, curP = np.copy(parent.curves)
                isConvolution = parent.child(10, 1).text().endswith("true")

                if param_name not in ["Scan Range", "Scan Units",
                                      "Curve Color", "Detuning Curve"]:
#                    theta, curS, curP = self.calculate_amplitudes(*allParams)
#                    parent.curves = np.copy((theta, curS, curP))
                    allParams.append(parent.row())
                    self.calculate_amps_in_thread(*allParams)
                elif param_name.endswith("Range"):
                    convFactor = self.allUnits[parent.child(9, 1).text()]
                    newLims = self.parse_limits(param_value)*convFactor
                    oldLims = parent.child(8, 1).limRads
                    if np.sum(np.abs(newLims-oldLims)) > 1e-14:
                        parent.child(8, 1).limRads = newLims
                        allParams[7] = newLims
                        allParams.append(parent.row())
                        self.calculate_amps_in_thread(*allParams)
#                        theta, curS, curP =\
#                            self.calculate_amplitudes(*allParams)
#                        parent.curves = np.copy((theta, curS, curP))
                    else:  # Only the units changed
                        theta = np.linspace(min(newLims)/convFactor,
                                            max(newLims)/convFactor,
                                            SCAN_LENGTH)
                        line_s.set_xdata(theta)
                        line_p.set_xdata(theta)
#                        self.axes.set_xlim(min(theta), max(theta))
                        self.axes.relim()
                        self.axes.autoscale_view()
                        self.canvas.draw()
                        return
                elif param_name.endswith("Units"):
                    convFactor = self.allUnits[param_value]
                    self.axes.set_xlabel(
                            self.xlabel_base+self.allUnitsStr[param_value])
                    newLims = parent.child(8, 1).limRads/convFactor
                    parent.child(8, 1).setText("{0}, {1}".format(newLims[0],
                                                                 newLims[1]))
                    self.update_all_units(param_value)
                elif param_name.endswith("Color"):
                    line_s.set_color(param_value)
                    line_p.set_color(param_value)
                    lgs = []
                    for lt in self.plot_lines.values():
                        for l in lt:
                            lgs.append(l.get_label())
                    self.axes.legend(lgs)
                    self.canvas.draw()
                    return

                if param_name not in {"Curve Color", "Scan Units"}:
                    line_s.set_xdata(theta/convFactor)
                    line_p.set_xdata(theta/convFactor)
                    if isConvolution:
                        pltCurvS = np.convolve(curS, curS, 'same') / curS.sum()
                        pltCurvP = np.convolve(curP, curP, 'same') / curP.sum()
                    else:
                        pltCurvS = curS
                        pltCurvP = curP
                    line_s.set_ydata(pltCurvS)
                    line_p.set_ydata(pltCurvP)
#                    self.axes.set_xlim(min(theta)/convFactor,
#                                       max(theta)/convFactor)
                    self.axes.relim()
                    self.axes.autoscale_view()
                    self.canvas.draw()

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

        for line in lines:
            self.axes.lines.remove(line)
        self.plot_lines.pop(plot_index)

        row = item.row()
        self.model.removeRow(row)

        lgs = []
        for lt in self.plot_lines.values():
            for l in lt:
                lgs.append(l.get_label())
        self.axes.legend(lgs)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw()

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
        units = root_item.child(9, 1).text()
        convFactor = self.allUnits[units]

        fileName = root_item.text()

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save File", fileName,
                "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, "w") as f:
                f.write(f"#dtheta,{units}\tsigma\tpi\n")
                for i in range(len(theta)):
                    f.write(f"{theta[i]/convFactor:.8e}\t{curvS[i]:.8e}\t{curvP[i]:.8e}\n")

    def calculate_amplitudes(self, crystal, geometry, hkl, thickness,
                             asymmetry,  radius, energy, limits, backendStr):
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

        E = float(energy)
        theta0 = crystalInstance.get_Bragg_angle(E)
        alpha = np.radians(float(asymmetry))
        if limits is None:
            theta = np.linspace(-100, 100, SCAN_LENGTH)*1e-6 + theta0
        else:
            theta = np.linspace(limits[0], limits[-1], SCAN_LENGTH) + theta0

        if geometry.startswith("B"):
            gamma0 = -np.sin(theta+alpha)
            gammah = np.sin(theta-alpha)
        else:
            gamma0 = -np.cos(theta+alpha)
            gammah = -np.cos(theta-alpha)
        hns0 = np.sin(alpha)*np.cos(theta+alpha) -\
            np.cos(alpha)*np.sin(theta+alpha)

        if backend == "xrtCL":
            self.matCL = mcl.XRT_CL(r'materials.cl',
                                    precisionOpenCL=precision)
            ampS, ampP = crystalInstance.get_amplitude_pytte(
                    E, gamma0, gammah, hns0, ucl=self.matCL, alphaAsym=alpha,
                    Ry=float(radius)*1000.)
        elif backend == "pytte":
            geotag = 0 if geometry.startswith('B') else np.pi*0.5
            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
                            thickness=Quantity(float(thickness), 'mm'),
                            debye_waller=1, xrt_crystal=crystalInstance,
                            Rx=Quantity(float(radius), 'm'),
                            asymmetry=Quantity(alpha+geotag, 'rad'))
            tts = TTscan(constant=Quantity(E, 'eV'),
                         scan=Quantity(theta-theta0, 'rad'),
                         polarization='sigma')
            ttp = TTscan(constant=Quantity(E, 'eV'),
                         scan=Quantity(theta-theta0, 'rad'),
                         polarization='pi')

            scan_tt_s = TakagiTaupin(ttx, tts)
            scan_tt_p = TakagiTaupin(ttx, ttp)
            scan_vector, Rs, Ts, ampS = scan_tt_s.run()
            scan_vector, Rp, Tp, ampP = scan_tt_p.run()
        else:
            ampS, ampP = crystalInstance.get_amplitude(E, gamma0, gammah, hns0)

        return theta - theta0, abs(ampS)**2, abs(ampP)**2

    def calculate_amps_in_thread(self, crystal, geometry, hkl, thickness,
                                 asymmetry,  radius, energy, limits,
                                 backendStr, plot_nr):
        self.amps_calculator = AmpCalculator(crystal, geometry, hkl, thickness,
                                        asymmetry,  radius, energy, limits,
                                        backendStr, plot_nr)
        self.amps_calculator.progress.connect(self.updateProgressBar)
        self.amps_calculator.result.connect(self.on_calculation_result)
        self.amps_calculator.finished.connect(self.on_calculation_finished)
        self.amps_calculator.start()

    def parse_limits(self, limstr):
        return np.array([float(pp) for pp in limstr.split(',')])

    def on_calculation_result(self, res_tuple):
        theta, curS, curP, plot_nr = res_tuple

        plot_item = self.model.item(plot_nr)
        plot_item.curves = np.copy((theta, curS, curP))

        isConvolution = plot_item.child(10, 1).text().endswith("true")
        convFactor = self.allUnits[plot_item.child(9, 1).text()]

        line_s, line_p = self.plot_lines[plot_item.plot_index]

        line_s.set_xdata(theta/convFactor)
        line_p.set_xdata(theta/convFactor)
        if isConvolution:
            pltCurvS = np.convolve(curS, curS, 'same') / curS.sum()
            pltCurvP = np.convolve(curP, curP, 'same') / curP.sum()
        else:
            pltCurvS = curS
            pltCurvP = curP
        line_s.set_ydata(pltCurvS)
        line_p.set_ydata(pltCurvP)
#                    self.axes.set_xlim(min(theta)/convFactor,
#                                       max(theta)/convFactor)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw()

    def on_calculation_finished(self):
#        print("All done")
        pass

    def updateProgressBar(self, dataTuple):
        if int(dataTuple[1]) < 100:
            updStr = r"{0}, {1}% done".format(dataTuple[0], int(dataTuple[1]))
        else:
            updStr = r"{}".format(dataTuple[0])
        self.progressBar.setValue(int(dataTuple[1]))
        self.progressBar.setFormat(updStr)


class AmpCalculator(QThread):
    progress = Signal(tuple)
    result = Signal(tuple)

    def __init__(self, crystal, geometry, hkl, thickness,
                 asymmetry,  radius, energy, limits, backendStr, plot_nr):
        super(AmpCalculator, self).__init__()
        self.crystal = crystal
        self.geometry = geometry
        self.hkl = hkl
        self.thickness = thickness
        self.asymmetry = asymmetry
        self.radius = radius
        self.energy = energy
        self.limits = limits
        self.backendStr = backendStr
        self.plot_nr = plot_nr

    def run(self):
        self.progress.emit(("Starting", 0))
        useTT = False
        backend = "xrt"
        precision = "float64"

        if self.backendStr == "auto":
            if self.radius == "inf":
                backend = "xrt"
            elif self.geometry == "Bragg transmitted":
                backend = "pytte"
            elif isOpenCL:
                useTT = True
                backend = "xrtCL"
                precision = "float64"
                if self.geometry.startswith("B"):
                    precision = "float32"
            else:
                backend = "pytte"
        elif self.backendStr == "pyTTE":
            backend = "pytte"
        elif self.backendStr == "xrtCL FP32":
            self.progress.emit(("Calculating on GPU", 0))
            backend = "xrtCL"
            precision = "float32"
            useTT = True
            if self.geometry.startswith("Laue"):
                self.progress.emit(
                        ("Calculations in Laue geometry require FP64", 0))
        elif self.backendStr == "xrtCL FP64":
            self.progress.emit(("Calculating on GPU", 0))
            backend = "xrtCL"
            precision = "float64"
            useTT = True
        else:
            return

        if backend == "xrtCL" and self.geometry == "Bragg transmitted":
            self.progress.emit(
                    ("Bragg transmitted not supported in OpenCL", 100))
            return

        hklList = parse_hkl(self.hkl)
        crystalClass = getattr(rxtl, self.crystal)
        crystalInstance = crystalClass(hkl=hklList, t=float(self.thickness),
                                       geom=self.geometry,
                                       useTT=useTT)

        E = float(self.energy)
        theta0 = crystalInstance.get_Bragg_angle(E)
        alpha = np.radians(float(self.asymmetry))
        if self.limits is None:
            theta = np.linspace(-100, 100, SCAN_LENGTH)*1e-6 + theta0
        else:
            theta = np.linspace(
                    self.limits[0], self.limits[-1], SCAN_LENGTH) + theta0

        if self.geometry.startswith("B"):
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
                    E, gamma0, gammah, hns0, ucl=matCL, alphaAsym=alpha,
                    Ry=float(self.radius)*1000., signal=self.progress)
        elif backend == "pytte":
            t0 = time.time()
            self.progress.emit(("Calculating on CPU", 0))
            geotag = 0 if self.geometry.startswith('B') else np.pi*0.5
            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
                            thickness=Quantity(float(self.thickness), 'mm'),
                            debye_waller=1, xrt_crystal=crystalInstance,
                            Rx=Quantity(float(self.radius), 'm'),
                            asymmetry=Quantity(alpha+geotag, 'rad'))
            tts = TTscan(constant=Quantity(E, 'eV'),
                         scan=Quantity(theta-theta0, 'rad'),
                         polarization='sigma')
            ttp = TTscan(constant=Quantity(E, 'eV'),
                         scan=Quantity(theta-theta0, 'rad'),
                         polarization='pi')

            scan_tt_s = TakagiTaupin(ttx, tts)
            scan_tt_p = TakagiTaupin(ttx, ttp)
            scan_vector, Rs, Ts, ampS = scan_tt_s.run()
            self.progress.emit(("Calculating on CPU", 50))
            scan_vector, Rp, Tp, ampP = scan_tt_p.run()
            self.progress.emit(("Calculation completed in {:.3f}s".format(
                    time.time()-t0), 100))
            if self.geometry.endswith("mitted"):
                ampS = np.sqrt(Ts)
                ampP = np.sqrt(Tp)
        else:
            ampS, ampP = crystalInstance.get_amplitude(E, gamma0, gammah, hns0)
            self.progress.emit(("Ready", 100))
        self.result.emit((theta-theta0, abs(ampS)**2, abs(ampP)**2,
                         self.plot_nr))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = PlotWidget()
    main_widget.setWindowTitle("xrt Crystal reflectivity calculator")
    main_widget.show()
    sys.exit(app.exec_())
