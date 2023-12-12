import sys

from qtpy.QtWidgets import (QWidget,
                            QLineEdit,
                            QLabel,
                            QPushButton,
                            QGridLayout)

from qtpy.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QApplication, QMainWindow

class PeakCalculator(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        layout = QGridLayout()

        self.h_label = QLabel('h', self)
        self.k_label = QLabel('k', self)
        self.l_label = QLabel('l', self)
 
        self.peak_1_label = QLabel('Peak 1', self)   
        self.peak_2_label = QLabel('Peak 2', self)   

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

        self.d1_label = QLabel('d = '+' '*12+' Å', self)   
        self.d2_label = QLabel('d = '+' '*12+' Å', self)   

        self.phi12_label = QLabel('φ = '+' '*12+ ' °', self)   

        self.calculate = QPushButton(text='Calculate')

        layout.addWidget(self.calculate, 0, 0, Qt.AlignRight)
        layout.addWidget(self.h_label, 0, 1, Qt.AlignCenter)
        layout.addWidget(self.k_label, 0, 2, Qt.AlignCenter)
        layout.addWidget(self.l_label, 0, 3, Qt.AlignCenter)
        layout.addWidget(self.phi12_label, 0, 4)

        layout.addWidget(self.peak_1_label, 1, 0, Qt.AlignRight)    
        layout.addWidget(self.h1_line, 1, 1)
        layout.addWidget(self.k1_line, 1, 2)
        layout.addWidget(self.l1_line, 1, 3)
        layout.addWidget(self.d1_label, 1, 4)
    
        layout.addWidget(self.peak_2_label, 2, 0, Qt.AlignRight)    
        layout.addWidget(self.h2_line, 2, 1)
        layout.addWidget(self.k2_line, 2, 2)
        layout.addWidget(self.l2_line, 2, 3)
        layout.addWidget(self.d2_label, 2, 4)

        self.setLayout(layout)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Testing Look')

        widget = PeakCalculator()
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())