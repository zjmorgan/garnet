import sys
from qtpy.QtWidgets import QApplication, QMainWindow

from mantid.kernel import Logger
from mantidqt.gui_helper import set_matplotlib_backend

set_matplotlib_backend()

from garnet._version import __version__  
from garnet.views.main_window import MainWindow

logger = Logger('garnet')

class Garnet(QMainWindow):

    __instance = None

    def __new__(cls):
        if Garnet.__instance is None:
            Garnet.__instance = QMainWindow.__new__(cls)  
        return Garnet.__instance

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.information(f'Garnet version: {__version__}')

        self.setWindowTitle(f'garnet - {__version__}')
        self.main_window = MainWindow(self)
        self.setCentralWidget(self.main_window)

def gui():
    app = QApplication(sys.argv)
    window = Garnet()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    gui()