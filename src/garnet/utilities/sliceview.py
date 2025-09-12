#!/usr/bin/env python3
import sys
import os

from qtpy.QtWidgets import QApplication, QMainWindow

from mantid.simpleapi import LoadMD
from mantidqt.widgets.sliceviewer.presenters.presenter import SliceViewer

theme = False
try:
    import qdarktheme

    qdarktheme.enable_hi_dpi()
    theme = True
except ImportError:
    print("Default theme")


class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        name = os.path.splitext(os.path.basename(filename))[0]
        self.setWindowTitle(name)
        ws = LoadMD(Filename=filename)
        viewer = SliceViewer(ws)
        self.setCentralWidget(viewer.view)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if theme:
        qdarktheme.setup_theme("light")
    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
