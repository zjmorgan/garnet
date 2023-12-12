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

        self.space_label = QLabel('SG:', self)   
        self.point_label = QLabel('PG:', self)   
        self.laue_label = QLabel('LG:', self)   
        self.beam_label = QLabel('Beam:', self)   
        self.horz_label = QLabel('Horz:', self)   
        self.vert_label = QLabel('Vert:', self)   

        layout.addWidget(self.a_label, 0, 0)
        layout.addWidget(self.b_label, 1, 0)
        layout.addWidget(self.c_label, 2, 0)
        layout.addWidget(self.space_label, 3, 0)
        layout.addWidget(self.point_label, 4, 0)
        layout.addWidget(self.laue_label, 5, 0)
        layout.addWidget(self.alpha_label, 0, 1)
        layout.addWidget(self.beta_label, 1, 1)
        layout.addWidget(self.gamma_label, 2, 1)
        layout.addWidget(self.beam_label, 3, 1)
        layout.addWidget(self.horz_label, 4, 1)
        layout.addWidget(self.vert_label, 5, 1)

        self.setLayout(layout)

    def set_lattice_constants(self, parameters, errors):

        a, b, c, alpha, beta, gamma = parameters
        siga, sigb, sigc, sigalpha, sigbeta, siggamma = np.array(errors)*1e4

        const = '{:8.4f}'
        free = '{:8.4f}({:.0f})'

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

    def set_groups(self, sg, pg, lg):

        self.space_label.setText('SG: '+sg.replace(' ', ''))
        self.point_label.setText('PG: '+pg.replace(' ', ''))
        self.laue_label.setText('LG: '+lg.replace(' ', ''))

    def set_crystal_directions(self, beam, horz, vert):

        beam = np.array(beam)/np.max(np.abs(beam))
        horz = np.array(horz)/np.max(np.abs(horz))
        vert = np.array(vert)/np.max(np.abs(vert))

        self.beam_label.setText('Beam: [{:.2f},{:.2f},{:.2f}]'.format(*beam))
        self.vert_label.setText('Vert: [{:.2f},{:.2f},{:.2f}]'.format(*vert))
        self.horz_label.setText('Horz: [{:.2f},{:.2f},{:.2f}]'.format(*horz))

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('UB info')

        widget = SampleCrystalInfo()
        widget.set_lattice_constants([7.2,7.2,8.9,90,90,120],
                                     [0.0001,0.0001,0.0002,0,0,0])
        widget.set_groups('P 6/m m m', '6/mmm', '6/mmm')
        widget.set_crystal_directions([1.2,0,0],[0,1,0],[0,0,0.8])
        self.setCentralWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())