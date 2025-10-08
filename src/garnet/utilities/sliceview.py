#!/usr/bin/env python3
import sys
import os

from qtpy.QtWidgets import QApplication, QMainWindow

from mantid.simpleapi import LoadMD, mtd
from mantidqt.widgets.sliceviewer.presenters.presenter import SliceViewer
from mantidqt.plotting.functions import plot_md_ws_from_names


class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        name = os.path.splitext(os.path.basename(filename))[0]
        self.setWindowTitle(name)
        LoadMD(Filename=filename, OutputWorkspace=name)
        try:
            viewer = SliceViewer(mtd[name])
            self.setCentralWidget(viewer.view)
        except:
            plot_md_ws_from_names([name], True, False)
            sys.exit(app.exec_())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
