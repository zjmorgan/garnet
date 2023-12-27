import sys
import numpy as np

from qtpy.QtWidgets import (QWidget,
                            QTableWidget,
                            QTableWidgetItem,
                            QHeaderView,
                            QLineEdit,
                            QLabel,
                            QPushButton,
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout)

from PyQt5.QtWidgets import QApplication, QMainWindow

from qtpy.QtGui import QDoubleValidator, QIntValidator

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

class StructureFactorCalculatorView(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        layout = QVBoxLayout()

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

        layout.addLayout(calculate)
        layout.addWidget(self.f2_table)
        layout.addLayout(indivdual)

        self.setLayout(layout)

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