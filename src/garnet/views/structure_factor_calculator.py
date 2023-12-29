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
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout,
                            QFrame)

from PyQt5.QtWidgets import QApplication, QMainWindow

from qtpy.QtGui import QDoubleValidator, QIntValidator

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

class StructureFactorCalculatorView(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        layout = QHBoxLayout()

        structure = QVBoxLayout()

        crystal = QHBoxLayout()
        parameters = QGridLayout()

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

        parameters.addWidget(a_label, 0, 0)
        parameters.addWidget(self.a_line, 0, 1)
        parameters.addWidget(b_label, 0, 2)
        parameters.addWidget(self.b_line, 0, 3)
        parameters.addWidget(c_label, 0, 4)
        parameters.addWidget(self.c_line, 0, 5)
        parameters.addWidget(angstrom_label, 0, 6)
        parameters.addWidget(alpha_label, 1, 0)
        parameters.addWidget(self.alpha_line, 1, 1)
        parameters.addWidget(beta_label, 1, 2)
        parameters.addWidget(self.beta_line, 1, 3)
        parameters.addWidget(gamma_label, 1, 4)
        parameters.addWidget(self.gamma_line, 1, 5)
        parameters.addWidget(degree_label, 1, 6)

        self.crystal_system_combo = QComboBox(self)
        self.crystal_system_combo.addItem('Triclinic')
        self.crystal_system_combo.addItem('Monoclinic')
        self.crystal_system_combo.addItem('Orthorhombic')
        self.crystal_system_combo.addItem('Tetragonal')
        self.crystal_system_combo.addItem('Trigonal')
        self.crystal_system_combo.addItem('Hexagonal')
        self.crystal_system_combo.addItem('Cubic')

        self.space_group_combo = QComboBox(self)
        self.setting_combo = QComboBox(self)

        self.load_CIF_button = QPushButton('Load CIF', self)

        crystal.addWidget(self.crystal_system_combo)
        crystal.addWidget(self.space_group_combo)
        crystal.addWidget(self.setting_combo)
        crystal.addWidget(self.load_CIF_button)

        structure.addLayout(crystal)
        structure.addLayout(parameters)

        stretch = QHeaderView.Stretch

        self.atm_table = QTableWidget()

        self.atm_table.setRowCount(0)
        self.atm_table.setColumnCount(6)

        self.atm_table.horizontalHeader().setStretchLastSection(True)
        self.atm_table.horizontalHeader().setSectionResizeMode(stretch)
        self.atm_table.setHorizontalHeaderLabels(['atm','x','y','z','occ','U'])

        structure.addWidget(self.atm_table)

        factors = QVBoxLayout()

        calculate = QHBoxLayout()

        dmin_label = QLabel('d(min)')
        angstrom_label = QLabel('Å')

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.1, 1000, 4, notation=notation)

        self.dmin_line = QLineEdit()
        self.dmin_line.setValidator(validator)

        self.calculate = QPushButton('Calculate', self)

        calculate.addWidget(dmin_label)
        calculate.addWidget(self.dmin_line)
        calculate.addWidget(angstrom_label)
        calculate.addStretch(1)
        calculate.addWidget(self.calculate)

        stretch = QHeaderView.Stretch

        self.f2_table = QTableWidget()

        self.f2_table.setRowCount(0)
        self.f2_table.setColumnCount(5)

        self.f2_table.horizontalHeader().setStretchLastSection(True)
        self.f2_table.horizontalHeader().setSectionResizeMode(stretch)
        self.f2_table.setHorizontalHeaderLabels(['h','k','l','d','F²'])

        indivdual = QGridLayout()

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h_line = QLineEdit()
        self.k_line = QLineEdit()
        self.l_line = QLineEdit()

        self.h_line.setValidator(validator)
        self.k_line.setValidator(validator)
        self.l_line.setValidator(validator)

        self.d_label = QLabel('d = '+' '*12+' Å', self)

        self.f2_label = QLabel('F² = '+' '*12+ ' ', self)

        self.equivalents = QLabel(' '*48, self)

        self.indivdual = QPushButton('Calculate', self)

        indivdual.addWidget(self.h_line, 0, 0)
        indivdual.addWidget(self.k_line, 0, 1)
        indivdual.addWidget(self.l_line, 0, 2)
        indivdual.addWidget(self.d_label, 0, 3)
        indivdual.addWidget(self.f2_label, 0, 4)

        indivdual.addWidget(self.equivalents, 1, 0, 1, 3)
        indivdual.addWidget(self.indivdual, 1, 4, 1, 1)

        factors.addLayout(calculate)
        factors.addWidget(self.f2_table)
        factors.addLayout(indivdual)

        vert_sep = QFrame()
        vert_sep.setFrameShape(QFrame.VLine)

        layout.addLayout(structure)
        layout.addWidget(vert_sep)
        layout.addLayout(factors)

        self.setLayout(layout)

    def get_crystal_system(self):

        return self.crystal_system_combo.currentText()

    def update_space_groups(self, nos):

        self.space_group_combo.clear()
        for no in nos:
            self.space_group_combo.addItem(no)

    def get_space_group(self):

        return self.space_group_combo.currentText()

    def update_settings(self, settings):

        self.setting_combo.clear()
        for setting in settings:
            self.setting_combo.addItem(setting)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('F^2')

        widget = StructureFactorCalculatorView()
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())