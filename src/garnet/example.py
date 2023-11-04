import sys
import os
os.environ['QT_API'] = 'pyqt5'

import time

from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

from garnet.models.parallel_task_utilities import ParallelTasks

def func(vals, i):
    time.sleep(4)

    print(i,vals)

class Worker(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
    
    def run(self):

        vals = [1,2,3,4,5,6]

        pt = ParallelTasks(func, [])
        pt.run_tasks(vals, 2)

class MainWindow(QWidget):

    def __init__(self):

        super(MainWindow, self).__init__()

        self.setWindowTitle('Testing Parallel')

        self.parallel_button = QPushButton('Run Parallel', self)
        self.parallel_button.clicked.connect(self.start_process)

        layout = QHBoxLayout()

        layout.addWidget(self.parallel_button)

        self.setLayout(layout)
        
        self.p = None

    def start_process(self):
        if self.p is None: 
            print('starting process')
            self.p = Worker()
            self.p.finished.connect(self.process_finished) 
            self.p.start()

    def process_finished(self):
        self.p = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
