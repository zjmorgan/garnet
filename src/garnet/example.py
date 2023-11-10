import sys
import os
os.environ['QT_API'] = 'pyqt5'

import numpy as np

from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

from garnet.models.coverage_optimizer import ExperimentPlanner

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import  NavigationToolbar2QT
from matplotlib.figure import Figure

class Worker(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
    
        inst_name = 'TOPAZ'

        axes = ['{},0,1,0,1', '135,0,0,1,1', '{},0,1,0,1']

        limits = [(-180,180), None, (-180,180)]

        UB = np.array([[-0.11589006, -0.09516246,  0.10667678],
                       [ 0.03385979,  0.1151471 ,  0.13950266],
                       [-0.13888608,  0.1074783 , -0.05500369]])

        wl_limits = [0.3, 3.5]

        d_min = 0.5

        point_group = 'm-3m'
        refl_cond = 'Body centred'

        self.garnet = ExperimentPlanner(inst_name, axes, limits, UB,
                                   wl_limits, point_group, refl_cond, d_min)
    
    def run(self):

        cov_int = self.garnet.initialize_settings(6, 6, 'garnet', n_proc=6)
        cov_opt = self.garnet.optimize_settings(20)

class MainWindow(QWidget):

    def __init__(self):

        super(MainWindow, self).__init__()

        self.setWindowTitle('Testing Parallel')

        self.parallel_button = QPushButton('Run Parallel', self)
        self.parallel_button.clicked.connect(self.start_process)

        layout = QVBoxLayout()

        layout.addWidget(self.parallel_button)
        
        canvas = FigureCanvas(Figure())
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas)

        self.ax = canvas.figure.subplots()
        self.line, = self.ax.plot([0], [0], '-ko', clip_on=False)
        self.ax.set_xlabel('Iteration [#]')
        self.ax.set_ylabel('Coverage [%]')
        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,100)
        self.ax.minorticks_on()
        self.ax.xaxis.get_major_locator().set_params(integer=True)

        self.setLayout(layout)
        
        self.p = None

    def process_finished(self):
        cov = self.p.garnet.coverage
        cov = [0]+cov
        self.line.set_data(np.arange(len(cov)), cov)
        self.ax.set_xlim(0,len(cov)-1)
        self.line.figure.canvas.draw()
        
        self.p = None

    def start_process(self):
        if self.p is None: 
            self.p = Worker()
            self.p.finished.connect(self.process_finished) 
            self.p.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
