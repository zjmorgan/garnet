import sys
import os
os.environ['QT_API'] = 'pyqt5'

from qtpy.QtWidgets import (QWidget,
                            QFrame,
                            QGridLayout,
                            QHBoxLayout,
                            QVBoxLayout,
                            QPushButton,
                            QCheckBox,
                            QComboBox,
                            QLineEdit)

from qtpy.QtGui import QDoubleValidator

import numpy as np
import pyvista as pv

import scipy.linalg

from pyvistaqt import QtInteractor#, QMainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow

from garnet.models.reciprocal_space_viewer import ReciprocalSpaceViewer as RSV
from mantid.simpleapi import LoadNexus

class ReciprocalSpaceViewer(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.UB = None

        self.proj_box = QCheckBox('Parallel Projection', self)
        self.proj_box.clicked.connect(self.change_proj)

        self.reset_button = QPushButton('Reset View', self)
        self.reset_button.clicked.connect(self.reset_view)

        self.view_combo = QComboBox(self)
        self.view_combo.addItem('[uvw]')
        self.view_combo.addItem('[hkl]')

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.axis1_line = QLineEdit()
        self.axis2_line = QLineEdit()
        self.axis3_line = QLineEdit()

        self.axis1_line.setValidator(validator)
        self.axis2_line.setValidator(validator)
        self.axis3_line.setValidator(validator)

        self.manual_button = QPushButton('View Plane', self)
        self.manual_button.clicked.connect(self.view_manual)

        self.px_button = QPushButton('+Qx', self)
        self.px_button.clicked.connect(self.view_yz)

        self.py_button = QPushButton('+Qy', self)
        self.py_button.clicked.connect(self.view_zx)

        self.pz_button = QPushButton('+Qz', self)
        self.pz_button.clicked.connect(self.view_xy)

        self.mx_button = QPushButton('-Qx', self)
        self.mx_button.clicked.connect(self.view_zy)

        self.my_button = QPushButton('-Qy', self)
        self.my_button.clicked.connect(self.view_xz)

        self.mz_button = QPushButton('-Qz', self)
        self.mz_button.clicked.connect(self.view_yx)

        self.a_star_button = QPushButton('a*', self)
        self.a_star_button.clicked.connect(self.view_bc_star)

        self.b_star_button = QPushButton('b*', self)
        self.b_star_button.clicked.connect(self.view_ca_star)

        self.c_star_button = QPushButton('c*', self)
        self.c_star_button.clicked.connect(self.view_ab_star)

        self.a_button = QPushButton('a', self)
        self.a_button.clicked.connect(self.view_bc)

        self.b_button = QPushButton('b', self)
        self.b_button.clicked.connect(self.view_ca)

        self.c_button = QPushButton('c', self)
        self.c_button.clicked.connect(self.view_ab)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)

        layout = QVBoxLayout()
        camera_layout = QHBoxLayout()
        plot_layout = QHBoxLayout()
        view_layout = QGridLayout()

        camera_layout.addWidget(self.proj_box)
        camera_layout.addWidget(self.reset_button)
        camera_layout.addStretch(1)
        camera_layout.addWidget(self.axis1_line)
        camera_layout.addWidget(self.axis2_line)
        camera_layout.addWidget(self.axis3_line)
        camera_layout.addWidget(self.view_combo)
        camera_layout.addWidget(self.manual_button)

        plot_layout.addWidget(self.plotter.interactor)

        view_layout.addWidget(self.px_button, 0, 0)
        view_layout.addWidget(self.py_button, 0, 1)
        view_layout.addWidget(self.pz_button, 0, 2)
        view_layout.addWidget(self.a_star_button, 0, 3)
        view_layout.addWidget(self.b_star_button, 0, 4)
        view_layout.addWidget(self.c_star_button, 0, 5)

        view_layout.addWidget(self.mx_button, 1, 0)
        view_layout.addWidget(self.my_button, 1, 1)
        view_layout.addWidget(self.mz_button, 1, 2)
        view_layout.addWidget(self.a_button, 1, 3)
        view_layout.addWidget(self.b_button, 1, 4)
        view_layout.addWidget(self.c_button, 1, 5)

        layout.addLayout(camera_layout)
        layout.addLayout(plot_layout)
        layout.addLayout(view_layout)

        self.setLayout(layout)

    def add_peaks(self, peak_dict):

        mesh = pv.PolyData(peak_dict['coordinates'])
        mesh['colors'] = peak_dict['intensities']

        self.plotter.add_mesh(mesh,
                              scalars='colors',
                              point_size=5,
                              log_scale=True,
                              render_points_as_spheres=True,
                              scalar_bar_args={'title': 'Intensity'})

        # geom = pv.Sphere(radius=1, theta_resolution=5, phi_resolution=5)

        # for j, key in enumerate(peak_dict.keys()):

        #     I, T, pk_no = peak_dict[key]

        #     if I > 0:

        #         glyph = mesh.glyph(scale=1, geom=geom)

        #         glyph.transform(T, inplace=True)

        #         glyph.__name = 'peak-{}'.format(key)

        #         scalars = np.full(geom.n_cells, np.log10(I))

        #         self.plotter.add_mesh(glyph,
        #                               name='peak-{}'.format(j),
        #                               scalars=scalars,
        #                               smooth_shading=True)

        self.plotter.show_axes()

        self.plotter.add_camera_orientation_widget()
        self.plotter.enable_depth_peeling()
        #self.plotter.enable_mesh_picking(callback=callback, style='surface', left_clicking=True, show=True, show_message=False, smooth_shading=True)

        self.change_proj()

    def change_proj(self):

        if self.proj_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def reset_view(self):

        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def set_UB(self, UB):

        self.UB = UB

        Gstar = np.dot(UB.T, UB)
        B = scipy.linalg.cholesky(Gstar, lower=False)
        U = np.dot(UB, np.linalg.inv(B))

        t = B.copy()
        t /= np.max(t, axis=1)

        T = np.dot(U,t)

        b = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                b.SetElement(i,j,T[i,j])

        actor = self.plotter.add_axes(xlabel='a*', ylabel='b*', zlabel='c*')
        actor.SetUserMatrix(b)

    def view_xy(self):
        self.plotter.view_xy()

    def view_yz(self):
        self.plotter.view_yz()

    def view_zx(self):
        self.plotter.view_zx()

    def view_yx(self):
        self.plotter.view_yx()

    def view_zy(self):
        self.plotter.view_zy()

    def view_xz(self):
        self.plotter.view_xz()

    def view_ab_star(self):
        if self.UB is not None:
            self.plotter.view_vector(np.dot(self.UB,[0,0,1]),
                                     np.dot(self.UB,[1,0,0]))

    def view_bc_star(self):
        if self.UB is not None:
            self.plotter.view_vector(np.dot(self.UB,[1,0,0]),
                                     np.dot(self.UB,[0,1,0]))

    def view_ca_star(self):
        if self.UB is not None:
            self.plotter.view_vector(np.dot(self.UB,[0,1,0]),
                                     np.dot(self.UB,[0,0,1]))

    def view_ab(self):
        if self.UB is not None:
            self.plotter.view_vector(np.cross(np.dot(self.UB,[1,0,0]),
                                              np.dot(self.UB,[0,1,0])),
                                     np.cross(np.dot(self.UB,[0,1,0]),
                                              np.dot(self.UB,[0,0,1])))

    def view_bc(self):
        if self.UB is not None:
            self.plotter.view_vector(np.cross(np.dot(self.UB,[0,1,0]),
                                              np.dot(self.UB,[0,0,1])),
                                     np.cross(np.dot(self.UB,[0,0,1]),
                                              np.dot(self.UB,[1,0,0])))

    def view_ca(self):
        if self.UB is not None:
            self.plotter.view_vector(np.cross(np.dot(self.UB,[0,0,1]),
                                              np.dot(self.UB,[1,0,0])),
                                     np.cross(np.dot(self.UB,[1,0,0]),
                                              np.dot(self.UB,[0,1,0])))

    def view_manual(self):

        axes_type = self.view_combo.currentText()

        axes = [self.axis1_line, self.axis2_line, self.axis3_line]
        valid_axes = all([axis.hasAcceptableInput() for axis in axes])

        if valid_axes and self.UB is not None:
            axis1 = float(self.axis1_line.text())
            axis2 = float(self.axis2_line.text())
            axis3 = float(self.axis3_line.text())

            ind = np.array([axis1,axis2,axis3])

            if axes_type == '[hkl]':
                matrix = self.UB
            else:
                matrix = np.cross(np.dot(self.UB, np.roll(np.eye(3),2,1)).T,
                                  np.dot(self.UB, np.roll(np.eye(3),1,1)).T).T

            vec = np.dot(matrix, ind)

            self.plotter.view_vector(vec)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Testing Look')

        peaks = LoadNexus('/SNS/TOPAZ/shared/zgf/garnet_peaks.nxs')
        rsv = RSV('peaks').get_peak_info()
        UB = RSV('peaks').get_UB()

        widget = ReciprocalSpaceViewer()
        widget.add_peaks(rsv)
        widget.set_UB(UB)

        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())