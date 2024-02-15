import os
os.environ['QT_API'] = 'pyqt5'

from qtpy.QtWidgets import (QHBoxLayout,
                            QVBoxLayout,
                            QWidget,
                            QTabWidget,
                            QPushButton)

import pyvista
pyvista.set_plot_theme('document')

from mantidqt.widgets.algorithmprogress import AlgorithmProgressWidget

from garnet.models.reciprocal_space_viewer import ReciprocalSpaceViewerModel
from garnet.views.reciprocal_space_viewer import ReciprocalSpaceViewerView
from garnet.presenters.reciprocal_space_viewer import ReciprocalSpaceViewer

from garnet.models.reciprocal_space_slicer import ReciprocalSpaceSlicerModel
from garnet.views.reciprocal_space_slicer import ReciprocalSpaceSlicerView
from garnet.presenters.reciprocal_space_slicer import ReciprocalSpaceSlicer

# from garnet.models.satellite_peak_indexer import SatellitePeakIndexerModel
# from garnet.views.satellite_peak_indexer import SatellitePeakIndexerView
# from garnet.presenters.satellite_peak_indexer import SatellitePeakIndexer

# from garnet.models.coverage_optimizer import CoverageOptimizerModel
# from garnet.views.coverage_optimizer import CoverageOptimizerView
# from garnet.presenters.coverage_optimizer import CoverageOptimizer

from garnet.models.sample_tools import SampleModel
from garnet.views.sample_tools import SampleView
from garnet.presenters.sample_tools import Sample

from garnet.models.crystal_structure_tools import CrystalStructureModel
from garnet.views.crystal_structure_tools import CrystalStructureView
from garnet.presenters.crystal_structure_tools import CrystalStructure

class MainWindow(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.tabs = QTabWidget()

        # rsv_view = ReciprocalSpaceViewerView(self)
        # rsv_model = ReciprocalSpaceViewerModel()
        # self.rsv = ReciprocalSpaceViewer(rsv_view, rsv_model)
        # self.tabs.addTab(rsv_view, 'ReciprocalSpaceViewer')

        # spi_view = SatellitePeakIndexerView(self)
        # spi_model = SatellitePeakIndexerModel()
        # self.spi = SatellitePeakIndexer(spi_view, spi_model)
        # self.tabs.addTab(spi_view, 'SatellitePeakIndexer')

        # co_view = CoverageOptimizerView(self)
        # co_model = CoverageOptimizerModel()
        # self.co = CoverageOptimizer(co_view, co_model)
        # self.tabs.addTab(co_view, 'CoverageOptimizer')

        # rss_view = ReciprocalSpaceSlicerView(self)
        # rss_model = ReciprocalSpaceSlicerModel()
        # self.rss = ReciprocalSpaceSlicer(rss_view, rss_model)
        # self.tabs.addTab(rss_view, 'ReciprocalSpaceSlicer')

        cs_view = CrystalStructureView(self)
        cs_model = CrystalStructureModel()
        self.cs = CrystalStructure(cs_view, cs_model)
        self.tabs.addTab(cs_view, 'CrystalStructure')

        s_view = SampleView(self)
        s_model = SampleModel()
        self.s = Sample(s_view, s_model)
        self.tabs.addTab(s_view, 'Sample')

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        apw = AlgorithmProgressWidget(self)
        apw.findChild(QPushButton).setText('Algorithm progress')

        hor_layout = QHBoxLayout()
        hor_layout.addWidget(apw)

        layout.addLayout(hor_layout)

        self.setLayout(layout)