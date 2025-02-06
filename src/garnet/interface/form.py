import sys

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
    QComboBox,
    QCheckBox,
)

from qtpy.QtGui import QDoubleValidator, QIntValidator

from garnet.config.instruments import beamlines


class FormInterface(QWidget):
    def __init__(self):
        super().__init__()

        plan = self.init_plan()

        layout = QVBoxLayout()
        layout.addLayout(plan)

        norm_tab = self.norm_plan()

        plan_widget = QTabWidget(self)
        plan_widget.addTab(norm_tab, "Normalization")

        layout.addWidget(plan_widget)

        self.setLayout(layout)

    def norm_plan(self):
        tab = QWidget()

        return tab

    def init_plan(self):
        layout = QVBoxLayout()

        experiment_params_layout = QHBoxLayout()
        run_params_layout = QHBoxLayout()
        wavelength_params_layout = QHBoxLayout()
        instrument_params_layout = QGridLayout()

        self.instrument_combo = QComboBox(self)
        self.instrument_combo.addItem("TOPAZ")
        self.instrument_combo.addItem("MANDI")
        self.instrument_combo.addItem("CORELLI")
        self.instrument_combo.addItem("SNAP")
        self.instrument_combo.addItem("WAND²")
        self.instrument_combo.addItem("DEMAND")

        self.grouping_combo = QComboBox(self)
        self.grouping_combo.addItem("1x1")

        ipts_label = QLabel("IPTS:")
        exp_label = QLabel("Experiment:")
        run_label = QLabel("Runs:")
        angstrom_label = QLabel("Å")

        validator = QIntValidator(1, 1000000000, self)

        self.runs_line = QLineEdit("")

        self.ipts_line = QLineEdit("")
        self.ipts_line.setValidator(validator)

        self.exp_line = QLineEdit("")
        self.exp_line.setValidator(validator)

        self.bkg_line = QLineEdit("")
        self.van_line = QLineEdit("")
        self.flux_line = QLineEdit("")
        self.eff_line = QLineEdit("")
        self.spec_line = QLineEdit("")
        self.cal_line = QLineEdit("")
        self.tube_line = QLineEdit("")

        self.wl_min_line = QLineEdit("0.3")
        self.wl_max_line = QLineEdit("3.5")

        wl_label = QLabel("λ:")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.2, 10, 5, notation=notation)

        self.wl_min_line.setValidator(validator)
        self.wl_max_line.setValidator(validator)

        validator = QIntValidator(1, 1000, self)

        self.load_button = QPushButton("Load Config", self)

        self.bkg_browse_button = QPushButton("Background", self)
        self.van_browse_button = QPushButton("Vanadium", self)
        self.flux_browse_button = QPushButton("Flux", self)
        self.eff_browse_button = QPushButton("Efficiency", self)
        self.spec_browse_button = QPushButton("Spectra", self)
        self.cal_browse_button = QPushButton("Detector", self)
        self.tube_browse_button = QPushButton("Tube", self)

        experiment_params_layout.addWidget(self.instrument_combo)
        experiment_params_layout.addWidget(ipts_label)
        experiment_params_layout.addWidget(self.ipts_line)
        experiment_params_layout.addWidget(exp_label)
        experiment_params_layout.addWidget(self.exp_line)
        run_params_layout.addWidget(self.load_button)

        run_params_layout.addWidget(run_label)
        run_params_layout.addWidget(self.runs_line)

        wavelength_params_layout.addWidget(self.grouping_combo)
        wavelength_params_layout.addWidget(wl_label)
        wavelength_params_layout.addWidget(self.wl_min_line)
        wavelength_params_layout.addWidget(self.wl_max_line)
        wavelength_params_layout.addWidget(angstrom_label)
        # wavelength_params_layout.addStretch(1)

        instrument_params_layout.addWidget(self.bkg_line, 0, 0)
        instrument_params_layout.addWidget(self.bkg_browse_button, 0, 1)
        instrument_params_layout.addWidget(self.van_line, 1, 0)
        instrument_params_layout.addWidget(self.van_browse_button, 1, 1)
        instrument_params_layout.addWidget(self.flux_line, 2, 0)
        instrument_params_layout.addWidget(self.flux_browse_button, 2, 1)
        instrument_params_layout.addWidget(self.eff_line, 3, 0)
        instrument_params_layout.addWidget(self.eff_browse_button, 3, 1)
        instrument_params_layout.addWidget(self.spec_line, 4, 0)
        instrument_params_layout.addWidget(self.spec_browse_button, 4, 1)
        instrument_params_layout.addWidget(self.cal_line, 5, 0)
        instrument_params_layout.addWidget(self.cal_browse_button, 5, 1)
        instrument_params_layout.addWidget(self.tube_line, 6, 0)
        instrument_params_layout.addWidget(self.tube_browse_button, 6, 1)

        layout.addLayout(experiment_params_layout)
        layout.addLayout(run_params_layout)
        layout.addLayout(wavelength_params_layout)
        layout.addLayout(instrument_params_layout)

        return layout

    # def initUI(self):
    #     layout = QVBoxLayout()

    #     self.fields = {}
    #     labels = [
    #         "Instrument", "IPTS", "Runs", "UBFile", "VanadiumFile",
    #         "EfficiencyFile", "FluxFile", "SpectraFile", "BackgroundFile",
    #         "DetectorCalibration", "TubeCalibration", "Grouping"
    #     ]

    #     for label in labels:
    #         row = QHBoxLayout()
    #         row.addWidget(QLabel(label + ":"))
    #         line_edit = QLineEdit()
    #         self.fields[label] = line_edit
    #         row.addWidget(line_edit)
    #         layout.addLayout(row)

    #     # Elastic Checkbox
    #     self.elastic_checkbox = QCheckBox("Elastic")
    #     layout.addWidget(self.elastic_checkbox)

    #     # Integration Fields
    #     self.integration_text = QTextEdit()
    #     self.integration_text.setPlaceholderText("Integration parameters in YAML format")
    #     layout.addWidget(QLabel("Integration:"))
    #     layout.addWidget(self.integration_text)

    #     # Normalization Fields
    #     self.normalization_text = QTextEdit()
    #     self.normalization_text.setPlaceholderText("Normalization parameters in YAML format")
    #     layout.addWidget(QLabel("Normalization:"))
    #     layout.addWidget(self.normalization_text)

    #     # Save Button
    #     save_button = QPushButton("Save File")
    #     save_button.clicked.connect(self.saveFile)
    #     layout.addWidget(save_button)

    #     self.setLayout(layout)
    #     self.setWindowTitle("Form Generator")

    # def saveFile(self):
    #     options = QFileDialog.Options()
    #     filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "YAML Files (*.yaml);;All Files (*)", options=options)
    #     if filename:
    #         data = {
    #             "Instrument": self.fields["Instrument"].text(),
    #             "IPTS": int(self.fields["IPTS"].text()),
    #             "Runs": self.fields["Runs"].text(),
    #             "UBFile": self.fields["UBFile"].text(),
    #             "VanadiumFile": self.fields["VanadiumFile"].text(),
    #             "EfficiencyFile": self.fields["EfficiencyFile"].text(),
    #             "FluxFile": self.fields["FluxFile"].text(),
    #             "SpectraFile": self.fields["SpectraFile"].text(),
    #             "BackgroundFile": self.fields["BackgroundFile"].text() or None,
    #             "DetectorCalibration": self.fields["DetectorCalibration"].text(),
    #             "TubeCalibration": self.fields["TubeCalibration"].text(),
    #             "Grouping": self.fields["Grouping"].text(),
    #             "Elastic": self.elastic_checkbox.isChecked(),
    #             "Integration": yaml.safe_load(self.integration_text.toPlainText()),
    #             "Normalization": yaml.safe_load(self.normalization_text.toPlainText())
    #         }

    #         with open(filename, "w") as file:
    #             yaml.dump(data, file, default_flow_style=False)

    #         print(f"File saved: {filename}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = FormInterface()
    form.show()
    sys.exit(app.exec_())
