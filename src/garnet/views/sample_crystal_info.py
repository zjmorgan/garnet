import sys
import numpy as np

from qtpy.QtWidgets import (QWidget,
                            QLabel,
                            QGridLayout)

from PyQt5.QtWidgets import QApplication, QMainWindow

class SampleCrystalInfo(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        layout = QGridLayout()

        self.a_label = QLabel('a = '+' '*12+' Å', self)
        self.b_label = QLabel('b = '+' '*12+' Å', self)
        self.c_label = QLabel('c = '+' '*12+' Å', self)

        self.alpha_label = QLabel('α = '+' '*12+ ' °', self)
        self.beta_label = QLabel('β = '+' '*12+ ' °', self)
        self.gamma_label = QLabel('γ = '+' '*12+ ' °', self)

        self.beam_label = QLabel('beam:', self)
        self.horz_label = QLabel('horz:', self)
        self.vert_label = QLabel('vert:', self)

        layout.addWidget(self.a_label, 0, 0)
        layout.addWidget(self.b_label, 0, 1)
        layout.addWidget(self.c_label, 0, 2)

        layout.addWidget(self.alpha_label, 1, 0)
        layout.addWidget(self.beta_label, 1, 1)
        layout.addWidget(self.gamma_label, 1, 2)

        layout.addWidget(self.beam_label, 2, 0)
        layout.addWidget(self.horz_label, 2, 1)
        layout.addWidget(self.vert_label, 2, 2)

        self.setLayout(layout)

    def set_lattice_constants(self, parameters, errors):

        a, b, c, alpha, beta, gamma = parameters
        siga, sigb, sigc, sigalpha, sigbeta, siggamma = np.array(errors)*1e4

        const = '{:.4f}'
        free = '{:.4f}({:.0f})'

        a_text = (const if siga == 0 else free).format(a,siga)
        b_text = (const if sigb == 0 else free).format(b,sigb)
        c_text = (const if sigc == 0 else free).format(c,sigc)

        alpha_text = (const if sigalpha == 0 else free).format(alpha,sigalpha)
        beta_text = (const if sigbeta == 0 else free).format(beta,sigbeta)
        gamma_text = (const if siggamma == 0 else free).format(gamma,siggamma)

        self.a_label.setText('a = '+a_text+' Å')
        self.b_label.setText('b = '+b_text+' Å')
        self.c_label.setText('c = '+c_text+' Å')

        self.alpha_label.setText('α = '+alpha_text+' °')
        self.beta_label.setText('β = '+beta_text+' °')
        self.gamma_label.setText('γ = '+gamma_text+' °')

    def set_crystal_directions(self, beam, horz, vert):

        beam = np.array(beam)/np.max(np.abs(beam))
        horz = np.array(horz)/np.max(np.abs(horz))
        vert = np.array(vert)/np.max(np.abs(vert))

        vec = '[{:.2f},{:.2f},{:.2f}]'

        self.beam_label.setText('beam: '+vec.format(*beam))
        self.vert_label.setText('horz: '+vec.format(*vert))
        self.horz_label.setText('vert: '+vec.format(*horz))

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('UB info')

        widget = SampleCrystalInfo()
        widget.set_lattice_constants([7.2,7.2,8.9,90,90,120],
                                     [0.0001,0.0001,0.0002,0,0,0])
        widget.set_crystal_directions([1.2,0,0],[0,1,0],[0,0,0.8])
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())