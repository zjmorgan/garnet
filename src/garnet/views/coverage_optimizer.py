from qtpy.QtWidgets import (QWidget,
                            QPushButton,
                            QLineEdit,
                            QLabel,
                            QTableWidget,
                            QTableWidgetItem,
                            QHeaderView,
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout)

from qtpy.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

class CoverageOptimizerView(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.initialize_button = QPushButton('Initialize', self)
        self.optimize_button = QPushButton('Optimize', self)

        self.settings_label = QLabel('Settings [#]', self)
        self.generations_label = QLabel('Generations [#]', self)
        self.individuals_label = QLabel('Individuals [#]', self)
        self.elite_label = QLabel('Elite Rate [%]', self)
        self.mutation_label = QLabel('Mutation Rate [%]', self)

        self.settings_line = QLineEdit('10')
        self.generations_line = QLineEdit('5')
        self.individuals_line = QLineEdit('20')
        self.elite_line = QLineEdit('20')
        self.mutation_line = QLineEdit('20')

        validator = QIntValidator(1, 100)
        self.settings_line.setValidator(validator)
        self.generations_line.setValidator(validator)
        self.individuals_line.setValidator(validator)

        notation = QDoubleValidator.StandardNotation
        validator = QDoubleValidator(0, 100, 5, notation=notation)
        self.elite_line.setValidator(validator)
        self.mutation_line.setValidator(validator)

        settings_layout = QGridLayout()

        settings_layout.addWidget(self.initialize_button, 0, 0)
        settings_layout.addWidget(self.settings_label, 0, 1, Qt.AlignCenter)
        settings_layout.addWidget(self.generations_label, 0, 2, Qt.AlignCenter)
        settings_layout.addWidget(self.individuals_label, 0, 3, Qt.AlignCenter)
        settings_layout.addWidget(self.elite_label, 0, 4, Qt.AlignCenter)
        settings_layout.addWidget(self.mutation_label, 0, 5, Qt.AlignCenter)

        settings_layout.addWidget(self.optimize_button, 1, 0)
        settings_layout.addWidget(self.settings_line, 1, 1)
        settings_layout.addWidget(self.generations_line, 1, 2)
        settings_layout.addWidget(self.individuals_line, 1, 3)
        settings_layout.addWidget(self.elite_line, 1, 4)
        settings_layout.addWidget(self.mutation_line, 1, 5)

        self.canvas = FigureCanvas(Figure())

        optimize_layout = QVBoxLayout()
        optimize_layout.addLayout(settings_layout)
        optimize_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        optimize_layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self.line_best, = self.ax.plot([0], [0], '-ko',
                                       label='Best', clip_on=False)
        self.line_worst, = self.ax.plot([0], [0], '-rs',
                                        label='Worst', clip_on=False)
        self.ax.set_xlabel('Generation [#]')
        self.ax.set_ylabel('Coverage [%]')
        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,100)
        self.ax.legend()
        self.ax.minorticks_on()
        self.ax.xaxis.get_major_locator().set_params(integer=True)

        generation_layout = QGridLayout()

        self.table = QTableWidget()

        self.table.setRowCount(1)
        self.table.setColumnCount(1)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        result_layout = QVBoxLayout()
        result_layout.addLayout(generation_layout)
        result_layout.addWidget(self.table)

        layout = QHBoxLayout()

        layout.addLayout(optimize_layout)
        layout.addLayout(result_layout)

        self.setLayout(layout)

    def generate_table(self, axes, n_orient):

        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        self.table.setRowCount(n_orient)
        self.table.setColumnCount(len(axes))
        self.table.setHorizontalHeaderLabels(axes)

    def update_table(self, settings):

        for i, setting in enumerate(settings):        
            for j, angle in enumerate(setting):
                    self.table.setItem(i, j, QTableWidgetItem(angle))

    def update_plots(self, generation, best, worst):

        self.line_best.set_data(generation, best)
        self.line_worst.set_data(generation, worst)
        self.ax.set_xlim(0, max(generation))
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def get_settings(self):

        return int(self.settings_line.text())

    def get_generations(self):

        return int(self.generations_line.text())

    def get_individuals(self):

        return int(self.individuals_line.text())

    def get_elite_rate(self):

        return float(self.elite_line.text())

    def get_mutation_rate(self):

        return float(self.mutation_line.text())