import sys

from qtpy.QtWidgets import (QWidget,
                            QLineEdit,
                            QLabel,
                            QPushButton,
                            QComboBox,
                            QTableWidget,
                            QTableWidgetItem,
                            QHeaderView,
                            QFrame,
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from qtpy.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QApplication, QMainWindow

class IntrumentData(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        experiment_layout = QHBoxLayout()

        self.experiment_label = QLabel('Experiment Name', self)
        self.experiment_combo = QComboBox(self)

        experiment_layout.addWidget(self.experiment_label)
        experiment_layout.addWidget(self.experiment_combo)

        self.table = QTableWidget()

        header = ['Projection','Name','Min','Max','Width','Bins']

        self.table.setRowCount(3)
        self.table.setColumnCount(len(header))
        self.table.setHorizontalHeaderLabels(header)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.symmetry_label = QLabel('Symmetry Operations', self)
        self.symmetry_button = QPushButton('Copy From Sample', self)
        self.symmetry_line = QLineEdit(self)

        self.symmetry_combo = QComboBox(self)
        self.symmetry_combo.addItem('Point Group')
        self.symmetry_combo.addItem('Space Group')
        self.symmetry_combo.addItem('Manual')

        self.group_combo = QComboBox(self)
        self.symmetry_combo.addItem('Point Group')

        symmetry_layout = QGridLayout()
        symmetry_layout.addWidget(self.symmetry_label, 0, 0)
        symmetry_layout.addWidget(self.symmetry_combo, 0, 1)
        symmetry_layout.addWidget(self.group_combo, 0, 2)
        symmetry_layout.addWidget(self.symmetry_button, 0, 3)
        symmetry_layout.addWidget(self.symmetry_line, 1, 0, 1, 4)

        parameters_layout = QVBoxLayout()

        parameters_layout.addLayout(experiment_layout)
        parameters_layout.addWidget(self.table)
        parameters_layout.addLayout(symmetry_layout)

        sample_layout = QHBoxLayout()

        self.sample_label = QLabel('Sample Name', self)
        self.sample_combo = QComboBox(self)

        sample_layout.addWidget(self.sample_label)
        sample_layout.addWidget(self.sample_combo)

        layout = QHBoxLayout()

        vert_sep = QFrame()
        vert_sep.setFrameShape(QFrame.VLine)

        layout.addLayout(parameters_layout)
        layout.addWidget(vert_sep)
        layout.addLayout(sample_layout)

        self.setLayout(layout)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Reciprocal Space Reconstruction')

        widget = IntrumentData()
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())