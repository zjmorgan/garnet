import sys
import numpy as np

from qtpy.QtWidgets import (QWidget,
                            QTableWidget,
                            QTableWidgetItem,
                            QHeaderView,
                            QLineEdit,
                            QLabel,
                            QComboBox,
                            QPushButton,
                            QCheckBox,
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout,
                            QFrame,
                            QTabWidget)

from PyQt5.QtWidgets import QApplication, QMainWindow

from qtpy.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor

class PeaksUB(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        ub_layout = self.__init_ub()
        viewer_layout = self.__init_viewer()
        peaks_layout = self.__init_peaks()

        vert_sep_left = QFrame()
        vert_sep_right = QFrame()

        vert_sep_left.setFrameShape(QFrame.VLine)
        vert_sep_right.setFrameShape(QFrame.VLine)

        layout = QHBoxLayout()

        layout.addLayout(ub_layout)
        layout.addWidget(vert_sep_left)
        layout.addLayout(viewer_layout)
        layout.addWidget(vert_sep_right)
        layout.addLayout(peaks_layout)

        self.setLayout(layout)

    def __init_ub(self):

        ub_layout = QVBoxLayout()

        self.a_line = QLineEdit()
        self.b_line = QLineEdit()
        self.c_line = QLineEdit()

        self.alpha_line = QLineEdit()
        self.beta_line = QLineEdit()
        self.gamma_line = QLineEdit()

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.1, 1000, 4, notation=notation)

        self.a_line.setValidator(validator)
        self.b_line.setValidator(validator)
        self.c_line.setValidator(validator)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(10, 170, 4, notation=notation)

        self.alpha_line.setValidator(validator)
        self.beta_line.setValidator(validator)
        self.gamma_line.setValidator(validator)

        a_label = QLabel('a')
        b_label = QLabel('b')
        c_label = QLabel('c')

        alpha_label = QLabel('α')
        beta_label = QLabel('β')
        gamma_label = QLabel('γ')

        angstrom_label = QLabel('Å')
        degree_label = QLabel('°')

        parameters_layout = QGridLayout()

        parameters_layout.addWidget(a_label, 0, 0)
        parameters_layout.addWidget(self.a_line, 0, 1)
        parameters_layout.addWidget(b_label, 0, 2)
        parameters_layout.addWidget(self.b_line, 0, 3)
        parameters_layout.addWidget(c_label, 0, 4)
        parameters_layout.addWidget(self.c_line, 0, 5)
        parameters_layout.addWidget(angstrom_label, 0, 6)
        parameters_layout.addWidget(alpha_label, 1, 0)
        parameters_layout.addWidget(self.alpha_line, 1, 1)
        parameters_layout.addWidget(beta_label, 1, 2)
        parameters_layout.addWidget(self.beta_line, 1, 3)
        parameters_layout.addWidget(gamma_label, 1, 4)
        parameters_layout.addWidget(self.gamma_line, 1, 5)
        parameters_layout.addWidget(degree_label, 1, 6)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-5, 5, 4, notation=notation)

        self.dh1_line = QLineEdit()
        self.dk1_line = QLineEdit()
        self.dl1_line = QLineEdit()

        self.dh2_line = QLineEdit()
        self.dk2_line = QLineEdit()
        self.dl2_line = QLineEdit()

        self.dh3_line = QLineEdit()
        self.dk3_line = QLineEdit()
        self.dl3_line = QLineEdit()

        self.dh1_line.setValidator(validator)
        self.dk1_line.setValidator(validator)
        self.dl1_line.setValidator(validator)

        self.dh2_line.setValidator(validator)
        self.dk2_line.setValidator(validator)
        self.dl2_line.setValidator(validator)

        self.dh3_line.setValidator(validator)
        self.dk3_line.setValidator(validator)
        self.dl3_line.setValidator(validator)

        self.max_order_line = QLineEdit('0')

        mod_vec1_label = QLabel('1:')
        mod_vec2_label = QLabel('2:')
        mod_vec3_label = QLabel('3:')

        dh_label = QLabel('Δh')
        dk_label = QLabel('Δk')
        dl_label = QLabel('Δl')

        max_order_label = QLabel('Max Order')

        self.cross_box = QCheckBox('Cross Terms', self)
        self.cross_box.setChecked(False)

        satellite_layout = QGridLayout()

        satellite_layout.addWidget(dh_label, 0, 1, Qt.AlignCenter)
        satellite_layout.addWidget(dk_label, 0, 2, Qt.AlignCenter)
        satellite_layout.addWidget(dl_label, 0, 3, Qt.AlignCenter)
        satellite_layout.addWidget(max_order_label, 0, 4, Qt.AlignCenter)
        satellite_layout.addWidget(mod_vec1_label, 1, 0)
        satellite_layout.addWidget(self.dh1_line, 1, 1)
        satellite_layout.addWidget(self.dk1_line, 1, 2)
        satellite_layout.addWidget(self.dl1_line, 1, 3)
        satellite_layout.addWidget(self.max_order_line, 1, 4)
        satellite_layout.addWidget(mod_vec2_label, 2, 0)
        satellite_layout.addWidget(self.dh2_line, 2, 1)
        satellite_layout.addWidget(self.dk2_line, 2, 2)
        satellite_layout.addWidget(self.dl2_line, 2, 3)
        satellite_layout.addWidget(self.cross_box, 2, 4)
        satellite_layout.addWidget(mod_vec3_label, 3, 0)
        satellite_layout.addWidget(self.dh3_line, 3, 1)
        satellite_layout.addWidget(self.dk3_line, 3, 2)
        satellite_layout.addWidget(self.dl3_line, 3, 3)

        x_label = QLabel('x:')
        y_label = QLabel('y:')
        z_label = QLabel('z:')

        a_star_label = QLabel('a*')
        b_star_label = QLabel('b*')
        c_star_label = QLabel('c*')

        self.uh_line = QLineEdit()
        self.uk_line = QLineEdit()
        self.ul_line = QLineEdit()

        self.vh_line = QLineEdit()
        self.vk_line = QLineEdit()
        self.vl_line = QLineEdit()

        self.wh_line = QLineEdit()
        self.wk_line = QLineEdit()
        self.wl_line = QLineEdit()

        self.uh_line.setEnabled(False)
        self.uk_line.setEnabled(False)
        self.ul_line.setEnabled(False)

        self.vh_line.setEnabled(False)
        self.vk_line.setEnabled(False)
        self.vl_line.setEnabled(False)

        self.wh_line.setEnabled(False)
        self.wk_line.setEnabled(False)
        self.wl_line.setEnabled(False)

        orientation_layout = QGridLayout()

        orientation_layout.addWidget(a_star_label, 0, 1, Qt.AlignCenter)
        orientation_layout.addWidget(b_star_label, 0, 2, Qt.AlignCenter)
        orientation_layout.addWidget(c_star_label, 0, 3, Qt.AlignCenter)
        orientation_layout.addWidget(x_label, 1, 0)
        orientation_layout.addWidget(self.uh_line, 1, 1)
        orientation_layout.addWidget(self.uk_line, 1, 2)
        orientation_layout.addWidget(self.ul_line, 1, 3)
        orientation_layout.addWidget(y_label, 2, 0)
        orientation_layout.addWidget(self.vh_line, 2, 1)
        orientation_layout.addWidget(self.vk_line, 2, 2)
        orientation_layout.addWidget(self.vl_line, 2, 3)
        orientation_layout.addWidget(z_label, 3, 0)
        orientation_layout.addWidget(self.wh_line, 3, 1)
        orientation_layout.addWidget(self.wk_line, 3, 2)
        orientation_layout.addWidget(self.wl_line, 3, 3)

        self.save_peaks_button = QPushButton('Save Peaks', self)
        self.load_peaks_button = QPushButton('Load Peaks', self)

        self.save_ub_button = QPushButton('Save UB', self)
        self.load_ub_button = QPushButton('Load UB', self)

        peaks_io_layout = QHBoxLayout()

        peaks_io_layout.addStretch(1)
        peaks_io_layout.addWidget(self.save_peaks_button)
        peaks_io_layout.addWidget(self.load_peaks_button)

        ub_io_layout = QHBoxLayout()

        ub_io_layout.addStretch(1)
        ub_io_layout.addWidget(self.save_ub_button)
        ub_io_layout.addWidget(self.load_ub_button)

        convert_tab = self.__init_convert_tab()
        peaks_tab = self.__init_peaks_tab()
        ub_tab = self.__init_ub_tab()

        ub_layout.addWidget(convert_tab)
        ub_layout.addLayout(parameters_layout)
        ub_layout.addWidget(peaks_tab)
        ub_layout.addLayout(peaks_io_layout)
        ub_layout.addLayout(satellite_layout)
        ub_layout.addWidget(ub_tab)
        ub_layout.addLayout(ub_io_layout)
        ub_layout.addLayout(orientation_layout)

        return ub_layout

    def __init_convert_tab(self):

        convert_tab = QTabWidget()

        convert_to_q_tab = QWidget()
        convert_to_q_tab_layout = QVBoxLayout()

        convert_to_q_params_layout = QGridLayout()

        self.experiment_combo = QComboBox(self)

        skip_run_label = QLabel('Skip Runs:')
        filter_time_label = QLabel('Filter Time:')
        time_unit_label = QLabel('s')

        validator = QIntValidator(1, 1000, self)

        self.skip_run_line = QLineEdit('1')
        self.skip_run_line.setValidator(validator)
    
        self.filter_time_line = QLineEdit('60')
        self.filter_time_line.setValidator(validator)

        convert_to_q_params_layout.addWidget(skip_run_label, 0, 0)
        convert_to_q_params_layout.addWidget(self.skip_run_line, 0, 1)
        convert_to_q_params_layout.addWidget(filter_time_label, 1, 0)
        convert_to_q_params_layout.addWidget(self.filter_time_line, 1, 1)
        convert_to_q_params_layout.addWidget(time_unit_label, 1, 2)

        self.convert_to_q_button = QPushButton('Convert', self)

        convert_to_q_action_layout = QHBoxLayout()
        convert_to_q_action_layout.addWidget(self.convert_to_q_button)
        convert_to_q_action_layout.addStretch(1)

        convert_to_q_tab_layout.addWidget(self.experiment_combo)
        convert_to_q_tab_layout.addLayout(convert_to_q_params_layout)
        convert_to_q_tab_layout.addStretch(1)
        convert_to_q_tab_layout.addLayout(convert_to_q_action_layout)

        convert_to_q_tab.setLayout(convert_to_q_tab_layout)

        convert_to_hkl_tab = QWidget()

        convert_to_hkl_tab = QWidget()
        convert_to_hkl_tab_layout = QVBoxLayout()

        convert_to_hkl_params_layout = QGridLayout()

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.U1_line = QLineEdit('1')
        self.U2_line = QLineEdit('0')
        self.U3_line = QLineEdit('0')

        self.V1_line = QLineEdit('0')
        self.V2_line = QLineEdit('1')
        self.V3_line = QLineEdit('0')

        self.W1_line = QLineEdit('0')
        self.W2_line = QLineEdit('0')
        self.W3_line = QLineEdit('1')

        self.U1_line.setValidator(validator)
        self.U2_line.setValidator(validator)
        self.U3_line.setValidator(validator)

        self.V1_line.setValidator(validator)
        self.V2_line.setValidator(validator)
        self.V3_line.setValidator(validator)

        self.W1_line.setValidator(validator)
        self.W2_line.setValidator(validator)
        self.W3_line.setValidator(validator)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-50, 50, 5, notation=notation)

        self.min1_line = QLineEdit('-6')
        self.min2_line = QLineEdit('-6')
        self.min3_line = QLineEdit('-6')

        self.max1_line = QLineEdit('6')
        self.max2_line = QLineEdit('6')
        self.max3_line = QLineEdit('6')

        self.min1_line.setValidator(validator)
        self.min2_line.setValidator(validator)
        self.min3_line.setValidator(validator)

        self.max1_line.setValidator(validator)
        self.max2_line.setValidator(validator)
        self.max3_line.setValidator(validator)

        validator = QIntValidator(1, 1001, self)

        self.bins1_line = QLineEdit('301')
        self.bins2_line = QLineEdit('301')
        self.bins3_line = QLineEdit('301')

        self.bins1_line.setValidator(validator)
        self.bins2_line.setValidator(validator)
        self.bins3_line.setValidator(validator)

        ax1_label = QLabel('1:')
        ax2_label = QLabel('2:')
        ax3_label = QLabel('3:')

        h_label = QLabel('h')
        k_label = QLabel('k')
        l_label = QLabel('l')

        min_label = QLabel('Min')
        max_label = QLabel('Max')
        bins_label = QLabel('Bins')

        convert_to_hkl_params_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)

        convert_to_hkl_params_layout.addWidget(min_label, 0, 4, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(max_label, 0, 5, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(bins_label, 0, 6, Qt.AlignCenter)

        convert_to_hkl_params_layout.addWidget(ax1_label, 1, 0, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(ax2_label, 2, 0, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(ax3_label, 3, 0, Qt.AlignCenter)

        convert_to_hkl_params_layout.addWidget(self.U1_line, 1, 1)
        convert_to_hkl_params_layout.addWidget(self.V1_line, 2, 1)
        convert_to_hkl_params_layout.addWidget(self.W1_line, 3, 1)

        convert_to_hkl_params_layout.addWidget(self.U2_line, 1, 2)
        convert_to_hkl_params_layout.addWidget(self.V2_line, 2, 2)
        convert_to_hkl_params_layout.addWidget(self.W2_line, 3, 2)

        convert_to_hkl_params_layout.addWidget(self.U3_line, 1, 3)
        convert_to_hkl_params_layout.addWidget(self.V3_line, 2, 3)
        convert_to_hkl_params_layout.addWidget(self.W3_line, 3, 3)

        convert_to_hkl_params_layout.addWidget(self.min1_line, 1, 4)
        convert_to_hkl_params_layout.addWidget(self.min2_line, 2, 4)
        convert_to_hkl_params_layout.addWidget(self.min3_line, 3, 4)

        convert_to_hkl_params_layout.addWidget(self.max1_line, 1, 5)
        convert_to_hkl_params_layout.addWidget(self.max2_line, 2, 5)
        convert_to_hkl_params_layout.addWidget(self.max3_line, 3, 5)

        convert_to_hkl_params_layout.addWidget(self.bins1_line, 1, 6)
        convert_to_hkl_params_layout.addWidget(self.bins2_line, 2, 6)
        convert_to_hkl_params_layout.addWidget(self.bins3_line, 3, 6)

        self.convert_to_hkl_button = QPushButton('Convert', self)

        convert_to_hkl_action_layout = QHBoxLayout()
        convert_to_hkl_action_layout.addWidget(self.convert_to_hkl_button)
        convert_to_hkl_action_layout.addStretch(1)

        convert_to_hkl_tab_layout.addLayout(convert_to_hkl_params_layout)
        convert_to_hkl_tab_layout.addStretch(1)
        convert_to_hkl_tab_layout.addLayout(convert_to_hkl_action_layout)

        convert_to_hkl_tab.setLayout(convert_to_hkl_tab_layout)

        convert_tab.addTab(convert_to_q_tab, 'Convert To Q')
        convert_tab.addTab(convert_to_hkl_tab, 'Convert To HKL')

        return convert_tab

    def __init_peaks_tab(self):

        peaks_tab = QTabWidget()

        find_tab = QWidget()
        find_tab_layout = QVBoxLayout()

        max_peaks_label = QLabel('Max Peaks:')
        min_distance_label = QLabel('Min Distance:')
        density_threshold_label = QLabel('Min Density:')
        find_edge_label = QLabel('Edge Pixels:') 
        distance_unit_label = QLabel('Å⁻¹')

        validator = QIntValidator(10, 1000, self)

        self.max_peaks_line = QLineEdit('100')
        self.max_peaks_line.setValidator(validator)

        self.min_distance_line = QLineEdit('100')

        validator = QIntValidator(1, 10000, self)

        self.density_threshold_line = QLineEdit()
        self.density_threshold_line.setValidator(validator)

        validator = QIntValidator(0, 64, self)

        self.find_edge_line = QLineEdit('0')
        self.find_edge_line.setValidator(validator)

        find_params_layout = QGridLayout()

        find_params_layout.addWidget(max_peaks_label, 0, 0)
        find_params_layout.addWidget(self.max_peaks_line, 0, 1)
        find_params_layout.addWidget(density_threshold_label, 1, 0)
        find_params_layout.addWidget(self.density_threshold_line, 1, 1)
        find_params_layout.addWidget(min_distance_label, 2, 0)
        find_params_layout.addWidget(self.min_distance_line, 2, 1)
        find_params_layout.addWidget(distance_unit_label, 2, 2)
        find_params_layout.addWidget(find_edge_label, 3, 0)
        find_params_layout.addWidget(self.find_edge_line, 3, 1)

        self.find_button = QPushButton('Find', self)

        find_action_layout = QHBoxLayout()
        find_action_layout.addWidget(self.find_button)
        find_action_layout.addStretch(1)

        find_tab_layout.addLayout(find_params_layout)
        find_tab_layout.addStretch(1)
        find_tab_layout.addLayout(find_action_layout)

        find_tab.setLayout(find_tab_layout)

        index_tab = QWidget()
        index_tab_layout = QVBoxLayout()

        index_tolerance_label = QLabel('Tolerance:')

        self.index_sat_box = QCheckBox('Satellite', self)
        self.index_sat_box.setChecked(False)

        self.index_tolerance_line = QLineEdit()
        self.index_sat_tolerance_line = QLineEdit()

        index_params_layout = QGridLayout()

        index_params_layout.addWidget(index_tolerance_label, 0, 0)
        index_params_layout.addWidget(self.index_tolerance_line, 0, 1)
        index_params_layout.addWidget(self.index_sat_tolerance_line, 0, 2)
        index_params_layout.addWidget(self.index_sat_box, 1, 2)

        self.round_box = QCheckBox('Round hkl', self)
        self.round_box.setChecked(True)

        self.index_button = QPushButton('Index', self)

        index_action_layout = QHBoxLayout()
        index_action_layout.addWidget(self.index_button)
        index_action_layout.addWidget(self.round_box)
        index_action_layout.addStretch(1)

        index_tab_layout.addLayout(index_params_layout)
        index_tab_layout.addStretch(1)
        index_tab_layout.addLayout(index_action_layout)

        index_tab.setLayout(index_tab_layout)

        centering_label = QLabel('Centering:')

        self.centering_combo = QComboBox(self)
        self.centering_combo.addItem('P')
        self.centering_combo.addItem('I')
        self.centering_combo.addItem('F')
        self.centering_combo.addItem('Robv')
        self.centering_combo.addItem('Rrev')
        self.centering_combo.addItem('A')
        self.centering_combo.addItem('B')
        self.centering_combo.addItem('C')
        self.centering_combo.addItem('H')

        min_d_unit_label = QLabel('Å')
        max_d_unit_label = QLabel('Å')

        min_d_label = QLabel('Min d-spacing:')
        max_d_label = QLabel('Max d-spacing:')
        predict_edge_label = QLabel('Edge Pixels:') 

        self.predict_sat_box = QCheckBox('Satellite', self)
        self.predict_sat_box.setChecked(False)

        self.min_d_line = QLineEdit()
        self.max_d_line = QLineEdit()
        self.min_sat_d_line = QLineEdit()
        self.max_sat_d_line = QLineEdit()
        self.predict_edge_line = QLineEdit('0')

        predict_tab = QWidget()
        predict_tab_layout = QVBoxLayout()

        predict_params_layout = QGridLayout()

        predict_params_layout.addWidget(centering_label, 0, 0)
        predict_params_layout.addWidget(self.centering_combo, 0, 1)
        predict_params_layout.addWidget(min_d_label, 1, 0)
        predict_params_layout.addWidget(self.min_d_line, 1, 1)
        predict_params_layout.addWidget(self.min_sat_d_line, 1, 2)
        predict_params_layout.addWidget(min_d_unit_label, 1, 3)
        predict_params_layout.addWidget(max_d_label, 2, 0)
        predict_params_layout.addWidget(self.max_d_line, 2, 1)
        predict_params_layout.addWidget(self.max_sat_d_line, 2, 2)
        predict_params_layout.addWidget(max_d_unit_label, 2, 3)
        predict_params_layout.addWidget(predict_edge_label, 3, 0)
        predict_params_layout.addWidget(self.predict_edge_line, 3, 1)
        predict_params_layout.addWidget(self.predict_sat_box, 3, 2)

        self.predict_button = QPushButton('Predict', self)

        predict_action_layout = QHBoxLayout()
        predict_action_layout.addWidget(self.predict_button)
        predict_action_layout.addStretch(1)

        predict_tab_layout.addLayout(predict_params_layout)
        predict_tab_layout.addStretch(1)
        predict_tab_layout.addLayout(predict_action_layout)

        predict_tab.setLayout(predict_tab_layout)

        self.centroid_box = QCheckBox('Centroid', self)
        self.centroid_box.setChecked(True)

        self.adaptive_box = QCheckBox('Adaptive Envelope', self)
        self.adaptive_box.setChecked(True)

        radius_label = QLabel('Radius:')
        inner_label = QLabel('Inner Factor:')
        outer_label = QLabel('Outer Factor:')
        radius_unit_label = QLabel('Å⁻¹')

        self.radius_line = QLineEdit('0.25')
        self.inner_line = QLineEdit('1.5')
        self.outer_line = QLineEdit('2')

        integrate_tab = QWidget()
        integrate_tab_layout = QVBoxLayout()

        integrate_params_layout = QGridLayout()

        integrate_params_layout.addWidget(radius_label, 0, 0)
        integrate_params_layout.addWidget(self.radius_line, 0, 1)
        integrate_params_layout.addWidget(radius_unit_label, 0, 2)
        integrate_params_layout.addWidget(inner_label, 2, 0)
        integrate_params_layout.addWidget(self.inner_line, 2, 1)
        integrate_params_layout.addWidget(outer_label, 3, 0)
        integrate_params_layout.addWidget(self.outer_line, 3, 1)

        self.integrate_button = QPushButton('Integrate', self)

        integrate_action_layout = QHBoxLayout()
        integrate_action_layout.addWidget(self.integrate_button)
        integrate_action_layout.addWidget(self.centroid_box)
        integrate_action_layout.addWidget(self.adaptive_box)
        integrate_action_layout.addStretch(1)

        integrate_tab_layout.addLayout(integrate_params_layout)
        integrate_tab_layout.addStretch(1)
        integrate_tab_layout.addLayout(integrate_action_layout)

        integrate_tab.setLayout(integrate_tab_layout)

        self.filter_combo = QComboBox(self)
        self.filter_combo.addItem('I/σ')
        self.filter_combo.addItem('d')
        self.filter_combo.addItem('λ')
        self.filter_combo.addItem('Q')
        self.filter_combo.addItem('Run #')

        self.comparison_combo = QComboBox(self)
        self.comparison_combo.addItem('>')
        self.comparison_combo.addItem('<')
        self.comparison_combo.addItem('>=')
        self.comparison_combo.addItem('<=')
        self.comparison_combo.addItem('=')
        self.comparison_combo.addItem('!=')

        self.filter_line = QLineEdit('10')

        filter_tab = QWidget()
        filter_tab_layout = QVBoxLayout()

        filter_params_layout = QGridLayout()

        filter_params_layout.addWidget(self.filter_combo, 0, 0)
        filter_params_layout.addWidget(self.comparison_combo, 0, 1)
        filter_params_layout.addWidget(self.filter_line, 0, 2)

        self.filter_button = QPushButton('Filter', self)

        filter_action_layout = QHBoxLayout()
        filter_action_layout.addWidget(self.filter_button)
        filter_action_layout.addStretch(1)

        filter_tab_layout.addLayout(filter_params_layout)
        filter_tab_layout.addStretch(1)
        filter_tab_layout.addLayout(filter_action_layout)

        filter_tab.setLayout(filter_tab_layout)

        peaks_tab.addTab(find_tab, 'Find Peaks')
        peaks_tab.addTab(index_tab, 'Index Peaks')
        peaks_tab.addTab(predict_tab, 'Predict Peaks')
        peaks_tab.addTab(integrate_tab, 'Integrate Peaks')
        peaks_tab.addTab(filter_tab, 'Filter Peaks')

        return peaks_tab    

    def __init_ub_tab(self):

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 1, 5, notation=notation)        

        ub_tab = QTabWidget()

        calculate_tolerance_label = QLabel('Tolerance:')

        self.calculate_tolerance_line = QLineEdit('0.1')
        self.calculate_tolerance_line.setValidator(validator)
        
        max_scalar_error_label = QLabel('Max Scalar Error:')

        self.max_scalar_error_line = QLineEdit('0.2')
        self.max_scalar_error_line.setValidator(validator)

        calculate_tab = QWidget()
        calculate_tab_layout = QVBoxLayout()

        calculate_params_layout = QGridLayout()

        calculate_params_layout.addWidget(calculate_tolerance_label, 0, 0)
        calculate_params_layout.addWidget(self.calculate_tolerance_line, 0, 1)
        calculate_params_layout.addWidget(max_scalar_error_label, 0, 2)
        calculate_params_layout.addWidget(self.max_scalar_error_line, 0, 3)

        self.conventional_button = QPushButton('Conventional', self)
        self.niggli_button = QPushButton('Niggli', self)
        self.select_button = QPushButton('Select', self)

        self.form_combo = QComboBox(self)
        form_label = QLabel('Form:') 

        calculate_action_layout = QHBoxLayout()
        calculate_action_layout.addWidget(self.conventional_button)
        calculate_action_layout.addStretch(1)
        calculate_action_layout.addWidget(self.niggli_button)
        calculate_action_layout.addWidget(form_label)
        calculate_action_layout.addWidget(self.form_combo)
        calculate_action_layout.addWidget(self.select_button)

        stretch = QHeaderView.Stretch

        self.cell_table = QTableWidget()

        self.cell_table.setRowCount(0)
        self.cell_table.setColumnCount(9)

        header = ['Error', 'Bravais', 'a', 'b', 'c', 'α', 'β', 'γ', 'V']

        self.cell_table.horizontalHeader().setSectionResizeMode(stretch)
        self.cell_table.setHorizontalHeaderLabels(header)
        self.cell_table.setEditTriggers(QTableWidget.NoEditTriggers)

        calculate_tab_layout.addLayout(calculate_params_layout)
        calculate_tab_layout.addWidget(self.cell_table)
        calculate_tab_layout.addStretch(1)
        calculate_tab_layout.addLayout(calculate_action_layout)

        calculate_tab.setLayout(calculate_tab_layout)

        transform_tolerance_label = QLabel('Tolerance:')

        self.transform_tolerance_line = QLineEdit('0.1')
        self.transform_tolerance_line.setValidator(validator)

        transform_label = QLabel('Lattice:')

        self.lattice_combo = QComboBox(self)
        self.lattice_combo.addItem('Triclinic')
        self.lattice_combo.addItem('Monoclinic')
        self.lattice_combo.addItem('Orthorhombic')
        self.lattice_combo.addItem('Tetragonal')
        self.lattice_combo.addItem('Rhombohedral')
        self.lattice_combo.addItem('Hexagonal')
        self.lattice_combo.addItem('Cubic')

        self.symmetry_combo = QComboBox(self)
        self.symmetry_combo.addItem('x,y,z')
        self.symmetry_combo.addItem('-x,-y,-z')

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.T11_line = QLineEdit('1')
        self.T12_line = QLineEdit('0')
        self.T13_line = QLineEdit('0')

        self.T21_line = QLineEdit('0')
        self.T22_line = QLineEdit('1')
        self.T23_line = QLineEdit('0')

        self.T31_line = QLineEdit('0')
        self.T32_line = QLineEdit('0')
        self.T33_line = QLineEdit('1')

        self.T11_line.setValidator(validator)
        self.T12_line.setValidator(validator)
        self.T13_line.setValidator(validator)

        self.T21_line.setValidator(validator)
        self.T22_line.setValidator(validator)
        self.T23_line.setValidator(validator)

        self.T31_line.setValidator(validator)
        self.T32_line.setValidator(validator)
        self.T33_line.setValidator(validator)

        hp_label = QLabel('h′:')
        kp_label = QLabel('k′:')
        lp_label = QLabel('l′:')

        h_label = QLabel('h')
        k_label = QLabel('k')
        l_label = QLabel('l')

        transform_tab = QWidget()
        transform_tab_layout = QVBoxLayout()

        transform_params_layout = QGridLayout()

        transform_params_layout.addWidget(transform_tolerance_label, 0, 0)
        transform_params_layout.addWidget(self.transform_tolerance_line, 0, 1)
        transform_params_layout.addWidget(transform_label, 1, 0)
        transform_params_layout.addWidget(self.lattice_combo, 1, 1)
        transform_params_layout.addWidget(self.symmetry_combo, 1, 2)

        transform_matrix_layout = QGridLayout()

        transform_matrix_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        transform_matrix_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        transform_matrix_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)
        transform_matrix_layout.addWidget(hp_label, 1, 0)
        transform_matrix_layout.addWidget(self.T11_line, 1, 1)
        transform_matrix_layout.addWidget(self.T12_line, 1, 2)
        transform_matrix_layout.addWidget(self.T13_line, 1, 3)
        transform_matrix_layout.addWidget(kp_label, 2, 0)
        transform_matrix_layout.addWidget(self.T21_line, 2, 1)
        transform_matrix_layout.addWidget(self.T22_line, 2, 2)
        transform_matrix_layout.addWidget(self.T23_line, 2, 3)
        transform_matrix_layout.addWidget(lp_label, 3, 0)
        transform_matrix_layout.addWidget(self.T31_line, 3, 1)
        transform_matrix_layout.addWidget(self.T32_line, 3, 2)
        transform_matrix_layout.addWidget(self.T33_line, 3, 3)

        self.transform_button = QPushButton('Transform', self)

        transform_action_layout = QHBoxLayout()
        transform_action_layout.addWidget(self.transform_button)
        transform_action_layout.addStretch(1)

        transform_tab_layout.addLayout(transform_params_layout)
        transform_tab_layout.addLayout(transform_matrix_layout)
        transform_tab_layout.addStretch(1)
        transform_tab_layout.addLayout(transform_action_layout)

        transform_tab.setLayout(transform_tab_layout)

        refine_tolerance_label = QLabel('Tolerance:')

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 1, 5, notation=notation)  

        self.refine_tolerance_line = QLineEdit()
        self.refine_tolerance_line.setValidator(validator)

        optimize_label = QLabel('Lattice:')

        self.optimize_combo = QComboBox(self)
        self.optimize_combo.addItem('Unconstrained')
        self.optimize_combo.addItem('Constrained')
        self.optimize_combo.addItem('Triclinic')
        self.optimize_combo.addItem('Monoclinic')
        self.optimize_combo.addItem('Orthorhombic')
        self.optimize_combo.addItem('Tetragonal')
        self.optimize_combo.addItem('Rhombohedral')
        self.optimize_combo.addItem('Hexagonal')
        self.optimize_combo.addItem('Cubic')

        refine_tab = QWidget()
        refine_tab_layout = QVBoxLayout()

        refine_params_layout = QGridLayout()

        refine_params_layout.addWidget(refine_tolerance_label, 0, 0)
        refine_params_layout.addWidget(self.refine_tolerance_line, 0, 1)
        refine_params_layout.addWidget(optimize_label, 1, 0)
        refine_params_layout.addWidget(self.optimize_combo, 1, 1)

        self.refine_button = QPushButton('Refine', self)

        refine_action_layout = QHBoxLayout()
        refine_action_layout.addWidget(self.refine_button)
        refine_action_layout.addStretch(1)

        refine_tab_layout.addLayout(refine_params_layout)
        refine_tab_layout.addStretch(1)
        refine_tab_layout.addLayout(refine_action_layout)

        refine_tab.setLayout(refine_tab_layout)

        ub_tab.addTab(calculate_tab, 'Calculate UB') 
        ub_tab.addTab(transform_tab, 'Transform UB') 
        ub_tab.addTab(refine_tab, 'Refine UB') 

        return ub_tab

    def __init_peaks(self):

        peaks_layout = QVBoxLayout()

        calculator_layout = QGridLayout()

        h_label = QLabel('h', self)
        k_label = QLabel('k', self)
        l_label = QLabel('l', self)

        peak_1_label = QLabel('1:', self)
        peak_2_label = QLabel('2:', self)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h1_line = QLineEdit()
        self.k1_line = QLineEdit()
        self.l1_line = QLineEdit()

        self.h2_line = QLineEdit()
        self.k2_line = QLineEdit()
        self.l2_line = QLineEdit()

        self.h1_line.setValidator(validator)
        self.k1_line.setValidator(validator)
        self.l1_line.setValidator(validator)

        self.h2_line.setValidator(validator)
        self.k2_line.setValidator(validator)
        self.l2_line.setValidator(validator)

        d_label = QLabel('d', self)
        d1_unit_label = QLabel('Å', self)
        d2_unit_label = QLabel('Å', self)

        phi_label = QLabel('φ', self)
        phi_unit_label = QLabel('°', self)

        self.d1_line = QLineEdit()
        self.d2_line = QLineEdit()
        self.phi_line = QLineEdit()

        self.d1_line.setEnabled(False)
        self.d2_line.setEnabled(False)
        self.phi_line.setEnabled(False)

        self.calculate = QPushButton('Calculate', self)

        calculator_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        calculator_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        calculator_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)
        calculator_layout.addWidget(d_label, 0, 4, Qt.AlignCenter)
        calculator_layout.addWidget(phi_label, 0, 6, Qt.AlignCenter)

        calculator_layout.addWidget(peak_1_label, 1, 0)
        calculator_layout.addWidget(self.h1_line, 1, 1)
        calculator_layout.addWidget(self.k1_line, 1, 2)
        calculator_layout.addWidget(self.l1_line, 1, 3)
        calculator_layout.addWidget(self.d1_line, 1, 4)
        calculator_layout.addWidget(d1_unit_label, 1, 5)
        calculator_layout.addWidget(self.phi_line, 1, 6)
        calculator_layout.addWidget(phi_unit_label, 1, 7)

        calculator_layout.addWidget(peak_2_label, 2, 0)
        calculator_layout.addWidget(self.h2_line, 2, 1)
        calculator_layout.addWidget(self.k2_line, 2, 2)
        calculator_layout.addWidget(self.l2_line, 2, 3)
        calculator_layout.addWidget(self.d2_line, 2, 4)
        calculator_layout.addWidget(d2_unit_label, 2, 5)
        calculator_layout.addWidget(self.calculate, 2, 6)

        stretch = QHeaderView.Stretch

        self.peaks_table = QTableWidget()

        self.peaks_table.setRowCount(0)
        self.peaks_table.setColumnCount(7)

        header = ['h', 'k', 'l', 'd', 'λ', 'I', 'I/σ']

        self.peaks_table.horizontalHeader().setSectionResizeMode(stretch)
        self.peaks_table.setHorizontalHeaderLabels(header)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)

        extended_info = QGridLayout()

        d_label = QLabel('d', self)
        lambda_label = QLabel('λ', self)
        length_unit_label = QLabel('Å', self)

        run_label = QLabel('Run #', self)
        bank_label = QLabel('Bank #', self)
        row_label = QLabel('Row #', self)
        col_label = QLabel('Col #', self)

        self.d_line = QLineEdit()
        self.lambda_line = QLineEdit()
        self.run_line = QLineEdit()
        self.bank_line = QLineEdit()
        self.row_line = QLineEdit()
        self.col_line = QLineEdit()

        self.d_line.setEnabled(False)
        self.lambda_line.setEnabled(False)
        self.run_line.setEnabled(False)
        self.bank_line.setEnabled(False)
        self.row_line.setEnabled(False)
        self.col_line.setEnabled(False)

        extended_info.addWidget(self.sigma_line, 0, 3)
        extended_info.addWidget(d_label, 0, 4)
        extended_info.addWidget(self.d_line, 0, 5)
        extended_info.addWidget(lambda_label, 0, 6)
        extended_info.addWidget(self.lambda_line, 0, 7)
        extended_info.addWidget(length_unit_label, 0, 8)

        extended_info.addWidget(run_label, 1, 0)
        extended_info.addWidget(self.run_line, 1, 1)
        extended_info.addWidget(bank_label, 1, 2)
        extended_info.addWidget(self.bank_line, 1, 3)
        extended_info.addWidget(row_label, 1, 4)
        extended_info.addWidget(self.row_line, 1, 5)
        extended_info.addWidget(col_label, 1, 6)
        extended_info.addWidget(self.col_line, 1, 7)

        peaks_layout.addLayout(calculator_layout)
        peaks_layout.addWidget(self.peaks_table)
        peaks_layout.addLayout(extended_info)

        return peaks_layout

    def __init_viewer(self):

        self.proj_box = QCheckBox('Parallel Projection', self)

        self.reset_button = QPushButton('Reset View', self)

        self.view_combo = QComboBox(self)
        self.view_combo.addItem('[hkl]')
        self.view_combo.addItem('[uvw]')

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.axis1_line = QLineEdit()
        self.axis2_line = QLineEdit()
        self.axis3_line = QLineEdit()

        self.axis1_line.setValidator(validator)
        self.axis2_line.setValidator(validator)
        self.axis3_line.setValidator(validator)

        self.axis1_label = QLabel('h', self)
        self.axis2_label = QLabel('k', self)
        self.axis3_label = QLabel('l', self)

        self.manual_button = QPushButton('View Axis', self)

        self.a_star_button = QPushButton('a*', self)
        self.b_star_button = QPushButton('b*', self)
        self.c_star_button = QPushButton('c*', self)

        self.a_button = QPushButton('a', self)
        self.b_button = QPushButton('b', self)
        self.c_button = QPushButton('c', self)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)

        viewer_layout = QVBoxLayout()
        camera_layout = QGridLayout()

        camera_layout.addWidget(self.proj_box, 0, 0)
        camera_layout.addWidget(self.reset_button, 1, 0)
        camera_layout.addWidget(self.a_star_button, 0, 1)
        camera_layout.addWidget(self.b_star_button, 0, 2)
        camera_layout.addWidget(self.c_star_button, 0, 3)
        camera_layout.addWidget(self.a_button, 1, 1)
        camera_layout.addWidget(self.b_button, 1, 2)
        camera_layout.addWidget(self.c_button, 1, 3)
        camera_layout.addWidget(self.axis1_label, 0, 4, Qt.AlignCenter)
        camera_layout.addWidget(self.axis2_label, 0, 5, Qt.AlignCenter)
        camera_layout.addWidget(self.axis3_label, 0, 6, Qt.AlignCenter)
        camera_layout.addWidget(self.axis1_line, 1, 4)
        camera_layout.addWidget(self.axis2_line, 1, 5)
        camera_layout.addWidget(self.axis3_line, 1, 6)
        camera_layout.addWidget(self.view_combo, 0, 7)
        camera_layout.addWidget(self.manual_button, 1, 7)

        peak_info = QGridLayout()
        
        left_label = QLabel('(', self)
        left_comma_label = QLabel(',', self)
        right_comma_label = QLabel(',', self)
        right_label = QLabel(')', self)

        int_h_label = QLabel('h', self)
        int_k_label = QLabel('k', self)
        int_l_label = QLabel('l', self)

        int_m_label = QLabel('m', self)
        int_n_label = QLabel('n', self)
        int_p_label = QLabel('p', self)

        self.h_line = QLineEdit()
        self.k_line = QLineEdit()
        self.l_line = QLineEdit()

        self.int_h_line = QLineEdit()
        self.int_k_line = QLineEdit()
        self.int_l_line = QLineEdit()

        self.int_m_line = QLineEdit()
        self.int_n_line = QLineEdit()
        self.int_p_line = QLineEdit()

        self.intensity_line = QLineEdit()
        self.sigma_line = QLineEdit()

        self.intensity_line.setEnabled(False)
        self.sigma_line.setEnabled(False)

        intensity_label = QLabel('I ', self)
        pm_label = QLabel('±', self)

        peak_info.addWidget(intensity_label, 0, 0)
        peak_info.addWidget(self.intensity_line, 0, 1)
        peak_info.addWidget(pm_label, 0, 2)
        peak_info.addWidget(self.sigma_line, 0, 3)

        peak_info.addWidget(int_h_label, 0, 7, Qt.AlignCenter)
        peak_info.addWidget(int_k_label, 0, 8, Qt.AlignCenter)
        peak_info.addWidget(int_l_label, 0, 9, Qt.AlignCenter)
        peak_info.addWidget(int_m_label, 0, 10, Qt.AlignCenter)
        peak_info.addWidget(int_n_label, 0, 11, Qt.AlignCenter)
        peak_info.addWidget(int_p_label, 0, 12, Qt.AlignCenter)
        peak_info.addWidget(left_label, 1, 0)
        peak_info.addWidget(self.h_line, 1, 1)
        peak_info.addWidget(left_comma_label, 1, 2)
        peak_info.addWidget(self.k_line, 1, 3)
        peak_info.addWidget(right_comma_label, 1, 4)
        peak_info.addWidget(self.l_line, 1, 5)
        peak_info.addWidget(right_label, 1, 6)
        peak_info.addWidget(self.int_h_line, 1, 7)
        peak_info.addWidget(self.int_k_line, 1, 8)
        peak_info.addWidget(self.int_l_line, 1, 9)
        peak_info.addWidget(self.int_m_line, 1, 10)
        peak_info.addWidget(self.int_n_line, 1, 11)
        peak_info.addWidget(self.int_p_line, 1, 12)

        viewer_layout.addLayout(camera_layout)
        viewer_layout.addWidget(self.plotter.interactor)
        viewer_layout.addLayout(peak_info)

        return viewer_layout

    def set_transform(self, T):

        if T is not None:

            a = pv._vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    a.SetElement(i,j,T[i,j])

            actor = self.plotter.add_axes(xlabel='a',
                                          ylabel='b',
                                          zlabel='c')
            actor.SetUserMatrix(a)

    def view_vector(self, vecs):

        if len(vecs) == 2:
            vec = np.cross(vecs[0],vecs[1])
            self.plotter.view_vector(vecs[0],vec)
        else:
            self.plotter.view_vector(vecs)

    def update_axis_labels(self):

        axes_type = self.view_combo.currentText()

        if axes_type == '[hkl]':
            self.axis1_label.setText('h')
            self.axis2_label.setText('k')
            self.axis3_label.setText('l')
        else:
            self.axis1_label.setText('u')
            self.axis2_label.setText('v')
            self.axis3_label.setText('w')

    def get_manual_indices(self):

        axes_type = self.view_combo.currentText()

        axes = [self.axis1_line, self.axis2_line, self.axis3_line]
        valid_axes = all([axis.hasAcceptableInput() for axis in axes])

        if valid_axes:

            axis1 = float(self.axis1_line.text())
            axis2 = float(self.axis2_line.text())
            axis3 = float(self.axis3_line.text())

            ind = np.array([axis1,axis2,axis3])

            return axes_type, ind

    def change_proj(self):

        if self.proj_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def reset_view(self):

        self.plotter.reset_camera()
        self.plotter.view_isometric()

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Testing Look')

        widget = PeaksUB()
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())