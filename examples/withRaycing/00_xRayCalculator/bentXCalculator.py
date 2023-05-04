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
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
    QPushButton, QMenu, QComboBox, QFileDialog,\
    QSplitter, QTreeView, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSlot, QTimer
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
from xrt.backends.raycing.pyTTE_x.pyTTE_rkpy_qt import TakagiTaupin, CalculateAmplitudes #, integrate_single_scan_step
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


def run_calculation(params, progress_queue):
    calculation = CalculateAmplitudes(*params)
    result = calculation.run()
    progress_queue.put((result))


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

        proc_nr = max(multiprocessing.cpu_count()-2, 1)
        self.pool = multiprocessing.Pool(processes=proc_nr)

        # Create a QVBoxLayout for the plot and the toolbar
        plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
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
        plot_uuid = uuid.uuid4()
        line_s = Line2D([], [])
        line_p = Line2D([], [], linestyle='--')
#        line_s = Line2D(theta*1e6, curS)
#        line_p = Line2D(theta*1e6, curP, linestyle='--')
        self.axes.add_line(line_s)
        self.axes.add_line(line_p)
        self.xlabel_base = r'$\theta-\theta_B$, '
        self.axes.set_xlabel(self.xlabel_base+r'$\mu$rad')
        self.plot_lines[plot_uuid] = (line_s, line_p)
        previousPlot = None
        plot_item = QStandardItem()
        plot_item.setFlags(plot_item.flags() | Qt.ItemIsEditable)
        plot_item.plot_index = plot_uuid
        self.model.appendRow(plot_item)
        plot_number = plot_item.row()

        calcParams = []

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
                      ("Scan Units", "urad",
                       list(self.allUnits.keys())),  # 9
                      ("DCM Rocking Curve", "false",
                       ["true", "false"]),  # 10
                      ("Calculation Backend", "auto",
                       self.allBackends),  # 11
                      ("Separator", "", None),  # 12
                      ("Curve Color", "blue", self.allColors)  # 13
                      ]

        if plot_number > 0:
            previousPlot = self.model.item(plot_number-1)

        for ii, (iname, ival, icb) in enumerate(initParams):
            newValue = ival
            if previousPlot is not None:
                if ii not in [5, 6, 11, 12, 13]:
                    newValue = previousPlot.child(ii, 1).text()
                elif ii == 13:
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

            if ii == 9:
                convFactor = self.allUnits[newValue]
                newLims = self.parse_limits(
                        plot_item.child(8, 1).text())*convFactor
                plot_item.child(8, 1).limRads = newLims
                calcParams.append(newLims)

            if icb is not None:
                cb = QComboBox()
                cb.addItems(icb)
                cb.setCurrentText(newValue)
                self.tree_view.setIndexWidget(item_value.index(), cb)
                cb.currentTextChanged.connect(
                        lambda text, item=item_value: item.setText(text))

            if ii not in [6, 8, 9, 10, 12, 13]:
                calcParams.append(newValue)

            if ii == 0:
                plot_name = newValue
            elif ii == 2:
                for hkl in parse_hkl(newValue):
                    plot_name += str(hkl)
                plot_name += "flat"
            if ii == 13:
                line_s.set_color(newValue)
                line_p.set_color(newValue)

        plot_item.setText(plot_name)
        line_s.set_label(plot_name+" $\sigma$")
        line_p.set_label(plot_name+" $\pi$")
        lgs = []
        for lt in self.plot_lines.values():
            for l in lt:
                lgs.append(l.get_label())
        self.axes.legend(lgs)

        plot_index = self.model.indexFromItem(plot_item)
        self.tree_view.expand(plot_index)

        calcParams.append(plot_number)

        self.calculate_amps_in_thread(*calcParams)
#            "Si",  # Crystal
#            "Bragg reflected",  # Geometry
#            "1, 1, 1",  # hkl
#            "1.",  # Thickness
#            "0.",  # Asymmetry
#            "inf",  # Bending radius
#            "9000",  # Energy
#            np.array([-100, 100])*1e-6,
#            "auto",
#            plot_number)

#        self.axes.relim()
#        self.axes.autoscale_view()
#        self.canvas.draw()

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
                                      "Curve Color", "DCM Rocking Curve"]:
                    allParams.append(parent.row())  # plot number
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
                    else:  # Only the units changed
                        theta = np.linspace(min(newLims)/convFactor,
                                            max(newLims)/convFactor,
                                            SCAN_LENGTH)
                        line_s.set_xdata(theta)
                        line_p.set_xdata(theta)
                        self.axes.set_xlim(min(theta), max(theta))
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

#                if param_name not in {"Curve Color", "Scan Units"}:
#                    line_s.set_xdata(theta/convFactor)
#                    line_p.set_xdata(theta/convFactor)
#                    if isConvolution:
#                        pltCurvS = np.convolve(curS, curS, 'same') / curS.sum()
#                        pltCurvP = np.convolve(curP, curP, 'same') / curP.sum()
#                    else:
#                        pltCurvS = curS
#                        pltCurvP = curP
#                    line_s.set_ydata(pltCurvS)
#                    line_p.set_ydata(pltCurvP)
##                    self.axes.set_xlim(min(theta)/convFactor,
##                                       max(theta)/convFactor)
#                    self.axes.relim()
#                    self.axes.autoscale_view()
#                    self.canvas.draw()

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
            matCL = mcl.XRT_CL(r'materials.cl',
                               precisionOpenCL=precision)
            ampS, ampP = crystalInstance.get_amplitude_pytte(
                    E, gamma0, gammah, hns0, ucl=matCL, alphaAsym=alpha,
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

        useTT = False
        backend = "xrt"
        precision = "float64"

        if backendStr == "auto":
            if radius == "inf":
                backend = "xrt"
            elif geometry == "Bragg transmitted":
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
            self.statusUpdate.emit(("Calculating on GPU", 0))
            backend = "xrtCL"
            precision = "float32"
            useTT = True
#            if geometry.startswith("Laue"):
#                self.progress.emit(
#                        ("Calculations in Laue geometry require FP64", 0))
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

        E = float(energy)
        theta0 = crystalInstance.get_Bragg_angle(E)
        alpha = np.radians(float(asymmetry))
        if limits is None:
            theta = np.linspace(-100, 100, SCAN_LENGTH)*1e-6 + theta0
        else:
            theta = np.linspace(
                    limits[0], limits[-1], SCAN_LENGTH) + theta0

        if geometry.startswith("B"):
            gamma0 = -np.sin(theta+alpha)
            gammah = np.sin(theta-alpha)
        else:
            gamma0 = -np.cos(theta+alpha)
            gammah = -np.cos(theta-alpha)
        hns0 = np.sin(alpha)*np.cos(theta+alpha) -\
            np.cos(alpha)*np.sin(theta+alpha)

        if backend == "xrtCL" and geometry == "Bragg transmitted":
            self.statusUpdate.emit(
                    ("Bragg transmitted not supported in OpenCL", 100))
            return

        if backend == "xrtCL":
            self.amps_calculator = AmpCalculator(
                    crystalInstance, theta-theta0, E, gamma0, gammah, hns0,
                    alpha, radius, precision, plot_nr)
            self.amps_calculator.progress.connect(self.update_progress_bar)
            self.amps_calculator.result.connect(self.on_calculation_result)
#            self.amps_calculator.finished.connect(self.on_calculation_finished)
            self.amps_calculator.start()
        elif backend == "pytte":
            self.t0 = time.time()
            self.statusUpdate.emit(("Calculating on CPU", 0))
            plot_item = self.model.item(plot_nr)
            plot_item.curves = np.copy((theta-theta0, np.zeros_like(theta),
                                        np.zeros_like(theta)))
            plot_item.curProgress = 0
            geotag = 0 if geometry.startswith('B') else np.pi*0.5
            ttx = TTcrystal(crystal='Si', hkl=crystalInstance.hkl,
                            thickness=Quantity(float(thickness), 'mm'),
                            debye_waller=1, xrt_crystal=crystalInstance,
                            Rx=Quantity(float(radius), 'm'),
                            asymmetry=Quantity(alpha+geotag, 'rad'))
            tts = TTscan(constant=Quantity(E, 'eV'),
                         scan=Quantity(theta-theta0, 'rad'),
                         polarization='sigma')
            isRefl = geometry.endswith("flected")
            amps_calculator = TakagiTaupin(ttx, tts)
            integrationParams = amps_calculator.prepare_arrays()

            progress_queue = multiprocessing.Manager().Queue()
            self.tasks = []

            for step in range(SCAN_LENGTH):
                args = integrationParams[:6]
                args += [vec_arg[step] for vec_arg in integrationParams[6:]]
                args += [isRefl, step, plot_nr]
#                print(len(args), args)

                task = self.pool.apply_async(run_calculation,
                                             args=(args, progress_queue))
                self.tasks.append(task)

            self.timer = QTimer()
            self.timer.timeout.connect(
                    lambda: self.check_progress(progress_queue))
            self.timer.start(200)  # Adjust the interval as needed
        else:
            ampS, ampP = crystalInstance.get_amplitude(E, gamma0, gammah, hns0)
            self.statusUpdate.emit(("Ready", 100))
            self.on_calculation_result(
                    (theta-theta0, abs(ampS)**2, abs(ampP)**2, plot_nr))

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

        if not any(task.ready() is False for task in self.tasks):
            self.statusUpdate.emit(("Calculation completed in {:.3f}s".format(
                    time.time()-self.t0), 100))
            self.timer.stop()
#            self.pool.close()

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
        self.axes.set_xlim(min(theta)/convFactor,
                           max(theta)/convFactor)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw()

    def on_calculation_finished(self):
#        print("All done")
        pass

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

    def __init__(self, crystal, dtheta, energy, gamma0, gammah, hns0,
                 alpha, radius, precision, plot_nr):
        super(AmpCalculator, self).__init__()
        self.crystalInstance = crystal
        self.dtheta = dtheta
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
        self.result.emit((self.dtheta, abs(ampS)**2, abs(ampP)**2,
                          self.plot_nr))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = PlotWidget()
    main_widget.setWindowTitle("xrt Crystal reflectivity calculator")
    main_widget.show()
    sys.exit(app.exec_())
