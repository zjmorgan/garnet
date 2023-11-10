from PyQt5.QtCore import QThread

class WorkerThread(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.function = None
        self.args = []
        
    def setup_function(self, function, args):

        self.function = function
        self.args = args

    def run(self):

        self.function(*self.args)