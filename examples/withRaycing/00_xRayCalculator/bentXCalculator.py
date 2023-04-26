# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:26:54 2023

@author: Roman Chernikov, Konstantin Klementiev, GPT-4
"""

import sys; sys.path.append(r'..\..\..')
import os
import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
    QTreeWidget, QTreeWidgetItem, QPushButton, QMenu, QComboBox, QFileDialog,\
    QSplitter
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from xrt.backends.raycing.pyTTE_x.elastic_tensors import CRYSTALS
from xrt.backends.raycing.pyTTE_x import TakagiTaupin, TTcrystal, TTscan, Quantity
from xrt.backends.raycing import materials as rmat
from xrt.backends.raycing import crystalclasses as rxtl

try:
    import pyopencl as cl
    import xrt.backends.raycing.myopencl as mcl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

hklre = r'\((\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})\)|(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})'


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
#        self.mainSplitter = QSplitter()
#        self.layout.addWidget(self.mainSplitter)
        self.setLayout(self.layout)

        # Create a QVBoxLayout for the plot and the toolbar
        self.plot_layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.plot_lines = []

        # Add the Matplotlib toolbar to the QVBoxLayout
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)

        # Add the canvas to the QVBoxLayout
        self.plot_layout.addWidget(self.canvas)

        # Add the QVBoxLayout to the main QHBoxLayout
        self.layout.addLayout(self.plot_layout)
#        self.mainSplitter.addWidget(self.plot_layout)

        self.tree_layout = QVBoxLayout()
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(2)
        self.tree_widget.setHeaderLabels(['Name', 'Value'])
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.itemChanged.connect(self.on_tree_item_changed)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
                self.show_context_menu)

        self.add_plot_button = QPushButton("Add new plot")
        self.add_plot_button.clicked.connect(self.add_plot)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_curve)

        # Create a QHBoxLayout for the buttons and add them to the main layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.add_plot_button)
        self.buttons_layout.addWidget(self.export_button)
        self.tree_layout.addLayout(self.buttons_layout)
        self.tree_layout.addWidget(self.tree_widget)

        self.layout.addLayout(self.tree_layout)
#        self.mainSplitter.addWidget(self.tree_layout)

        if isOpenCL:
            self.matCL = mcl.XRT_CL(r'materials.cl',
#                                   precisionOpenCL='float32',
                                    precisionOpenCL='float64'
                                    )
        else:
            self.matCL = None

        self.allCrystals = []
        for ck in CRYSTALS.keys():
            if not ck.startswith("prototype"):
                self.allCrystals.append(ck)

        self.allGeometries = ['Bragg reflected', 'Bragg transmitted',
                              'Laue reflected', 'Laue transmitted']
        self.add_plot()

    def add_plot(self):
        theta, curS, curP = self.calculate_amplitudes(
                            "Si",  # Crystal
                            "Bragg reflected",  # Geometry
                            "1, 1, 1",  # hkl
                            "1.",  # Thickness
                            "0.",  # Asymmetry
                            "inf",  # Bending radius
                            "9000")  # Energy
        line_s = Line2D(theta, curS)
        line_p = Line2D(theta, curP, linestyle='--')
        self.plot_lines.append((line_s, line_p))
        self.axes.add_line(line_s)
        self.axes.add_line(line_p)
        plot_number = len(self.plot_lines)
        plot_name = "Si"
        line_s.set_label(plot_name+" $\sigma$")
        line_p.set_label(plot_name+" $\pi$")
        line_s.set_color('red')
        line_p.set_color('red')
        lgs = []
        for lt in self.plot_lines:
            for l in lt:
                lgs.append(l.get_label())

        plot_item = QTreeWidgetItem(self.tree_widget, [plot_name, ""])
        plot_item.setData(1, Qt.UserRole, plot_number - 1)
        plot_item.setFlags(plot_item.flags() | Qt.ItemIsEditable)

        p01CrystalItem = QTreeWidgetItem(plot_item, ["Crystal", "Si"])
        p01CrystalCB = QComboBox()
        p01CrystalCB.addItems(self.allCrystals)
        p01CrystalCB.setCurrentText("Si")
        self.tree_widget.setItemWidget(p01CrystalItem, 1, p01CrystalCB)
        p01CrystalCB.currentTextChanged.connect(
                lambda text, item=p01CrystalItem: item.setText(1, text))

        p02GeometryItem = QTreeWidgetItem(plot_item, ["Geometry", "Bragg reflected"])
        p02GeometryCB = QComboBox()
        p02GeometryCB.addItems(self.allGeometries)
        p02GeometryCB.setCurrentIndex(0)
        self.tree_widget.setItemWidget(p02GeometryItem, 1, p02GeometryCB)
        p02GeometryCB.currentTextChanged.connect(
                lambda text, item=p02GeometryItem: item.setText(1, text))

        p03hklItem = QTreeWidgetItem(plot_item, ["hkl", "1, 1, 1"])
        p03hklItem.setFlags(p03hklItem.flags() | Qt.ItemIsEditable)

        p04ThicknessItem = QTreeWidgetItem(plot_item, ["Thickness, mm", "1."])
        p04ThicknessItem.setFlags(p04ThicknessItem.flags() | Qt.ItemIsEditable)

        p05Asymmetry = QTreeWidgetItem(plot_item,
                                       ["Asymmetry Angle, deg", "0."])
        p05Asymmetry.setFlags(p05Asymmetry.flags() | Qt.ItemIsEditable)

        p06Radius = QTreeWidgetItem(plot_item, ["Bending Radius, m", "inf"])
        p06Radius.setFlags(p06Radius.flags() | Qt.ItemIsEditable)

        p07Energy = QTreeWidgetItem(plot_item, ["Energy, eV", "9000"])
        p07Energy.setFlags(p07Energy.flags() | Qt.ItemIsEditable)

        color_item = QTreeWidgetItem(plot_item, ["Color", "red"])
        color_combobox = QComboBox()
        color_combobox.addItems(["red", "green", "blue"])
        color_combobox.setCurrentText("red")
        self.tree_widget.setItemWidget(color_item, 1, color_combobox)
        color_combobox.currentTextChanged.connect(
                lambda text, item=color_item: item.setText(1, text))

        self.axes.legend(lgs)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw()

    def on_tree_item_changed(self, item, column):
        if column == 0:
            plot_index = item.data(1, Qt.UserRole)
            if plot_index is not None:
                lines = self.plot_lines[plot_index]
                lines[0].set_label(item.text(0)+" $\sigma$")
                lines[1].set_label(item.text(0)+" $\pi$")
                lgs = []
                for lt in self.plot_lines:
                    for l in lt:
                        lgs.append(l.get_label())
                self.axes.legend(lgs)
                self.canvas.draw()
        else:
            parent = item.parent()
            if parent:
                plot_index = parent.data(1, Qt.UserRole)
                line_s, line_p = self.plot_lines[plot_index]  # Get both sine and cosine lines
                param_name = item.text(0)
                param_value = item.text(1)

                if param_name == "Crystal":
                    theta, curS, curP = self.calculate_amplitudes(
                            param_value,  # Crystal
                            parent.child(1).text(1),  # Geometry
                            parent.child(2).text(1),  # hkl
                            parent.child(3).text(1),  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name == "Geometry":
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            param_value,  # Geometry
                            parent.child(2).text(1),  # hkl
                            parent.child(3).text(1),  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name == "hkl":
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            parent.child(1).text(1),  # Geometry
                            param_value,  # hkl
                            parent.child(3).text(1),  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name.startswith("Thickness"):
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            parent.child(1).text(1),  # Geometry
                            parent.child(2).text(1),  # hkl
                            param_value,  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name.startswith("Asym"):
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            parent.child(1).text(1),  # Geometry
                            parent.child(2).text(1),  # hkl
                            parent.child(3).text(1),  # Thickness
                            param_value,  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name.startswith("Bending"):
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            parent.child(1).text(1),  # Geometry
                            parent.child(2).text(1),  # hkl
                            parent.child(3).text(1),  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            param_value,  # Bending radius
                            parent.child(6).text(1)  # Energy
                            )
                elif param_name.startswith("Energy"):
                    theta, curS, curP = self.calculate_amplitudes(
                            parent.child(0).text(1),  # Crystal
                            parent.child(1).text(1),  # Geometry
                            parent.child(2).text(1),  # hkl
                            parent.child(3).text(1),  # Thickness
                            parent.child(4).text(1),  # Asymmetry
                            parent.child(5).text(1),  # Bending radius
                            param_value  # Energy
                            )
                elif param_name == "Color":
                    line_s.set_color(param_value)
                    line_p.set_color(param_value)
                    lgs = []
                    for lt in self.plot_lines:
                        for l in lt:
                            lgs.append(l.get_label())
                    self.axes.legend(lgs)
                    self.canvas.draw()
                    return

                if param_name not in {"Color",}:
                    line_s.set_xdata(theta)
                    line_p.set_xdata(theta)
                    line_s.set_ydata(curS)
                    line_p.set_ydata(curP)  # Update the cosine line as well
                    self.axes.relim()
                    self.axes.autoscale_view()

                    self.canvas.draw()

    def show_context_menu(self, position):
        item = self.tree_widget.itemAt(position)
        if item and item.parent() is None:
            menu = QMenu(self.tree_widget)
            delete_action = menu.addAction("Delete plot")
            delete_action.triggered.connect(lambda: self.delete_plot(item))
            menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

    def delete_plot(self, item):
        plot_index = item.data(1, Qt.UserRole)
        line = self.plot_lines[plot_index]

        self.axes.lines.remove(line)
        self.plot_lines.pop(plot_index)
        self.tree_widget.takeTopLevelItem(self.tree_widget.indexOfTopLevelItem(item))

        for i, line in enumerate(self.plot_lines[plot_index:]):
            line.set_label(f"name{(plot_index + i + 1):02d}")
            self.tree_widget.topLevelItem(plot_index + i).setText(0, line.get_label())

        self.axes.legend([l.get_label() for l in self.plot_lines])
        self.canvas.draw()

    def export_curve(self):
        selected_item = self.tree_widget.currentItem()
        if selected_item is None or selected_item.parent() is not None:
            return

        plot_index = selected_item.data(1, Qt.UserRole)
        if plot_index is not None:
            line = self.plot_lines[plot_index]

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Curve", "", "Text Files (*.txt);;All Files (*)", options=options)

            if file_name:
                x, y = line.get_data()
                data = np.column_stack((x, y))
                np.savetxt(file_name, data, fmt='%.6f', delimiter='\t', header="X\tY", comments='')

    def calculate_sine(self, amplitude, frequency, phase, x=None):
        if x is None:
            x = np.linspace(0, 2 * np.pi, 100)
        y_sin = amplitude * np.sin(frequency * x + phase)
        y_cos = amplitude * np.cos(frequency * x + phase)
        return x, y_sin, y_cos

    def calculate_amplitudes(self, crystal, geometry, hkl, thickness,
                             asymmetry,  radius, energy, x=None):
        useTT = False if radius == 'inf' else True
        hklList = self.parse_hkl(hkl)
        crystalClass = getattr(rxtl, crystal)
        crystalInstance = crystalClass(hkl=hklList, t=float(thickness),
                                       geom=geometry,
                                       useTT=useTT)

        E = float(energy)
        theta0 = crystalInstance.get_Bragg_angle(E)
        alpha = np.radians(float(asymmetry))
        theta = np.linspace(-100, 100, 1000)*1e-6 + theta0

#        s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
#        sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
#        if geometry.startswith('Bragg'):
#            n = (0, 0, 1)  # outward surface normal
#        else:
#            n = (0, -1, 0)  # outward surface normal
#        hn = (0, np.sin(alpha), np.cos(alpha))  # outward Bragg normal
#        hns0x = sum(i*j for i, j in zip(hn, s0))
#        gamma0x = sum(i*j for i, j in zip(n, s0))
#        gammahx = sum(i*j for i, j in zip(n, sh))

        if geometry.startswith("B"):
            gamma0 = -np.sin(theta+alpha)
            gammah = np.sin(theta-alpha)
        else:
            gamma0 = -np.cos(theta+alpha)
            gammah = -np.cos(theta-alpha)
        hns0 = np.sin(alpha)*np.cos(theta+alpha)-np.cos(alpha)*np.sin(theta+alpha)

#        print("gamma0", gamma0, gamma0x)
#        print("gammah", gammah, gammahx)
#        print("hns0", hns0, hns0x)

        if useTT:
            ampS, ampP = crystalInstance.get_amplitude_pytte(E, gamma0, gammah, hns0,
                                                       ucl=self.matCL, alphaAsym=alpha,
                                                       Ry=float(radius)*1000.)
        else:
            ampS, ampP = crystalInstance.get_amplitude(E, gamma0, gammah, hns0)

        return (theta - theta0)*1e6, abs(ampS)**2, abs(ampP)**2

    def parse_hkl(self, hklstr):
        matches = re.findall(hklre, hklstr)
        hklval = []
        for match in matches:
            for group in match:
                if group != '':
                    hklval.append(int(group))
        return hklval


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = PlotWidget()
    main_widget.setWindowTitle("xrt Crystal reflectivity calculator")
    main_widget.show()
    sys.exit(app.exec_())