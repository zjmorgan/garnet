import os
import sys

import yaml

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from mantid.simpleapi import (
    LoadNexus,
    LoadEmptyInstrument,
    LoadParameterFile,
    LoadIsawPeaks,
    FilterPeaks,
    IndexPeaks,
    CombinePeaksWorkspaces,
    CalculateUMatrix,
    ApplyInstrumentToPeaks,
    SCDCalibratePanels,
    MoveInstrumentComponent,
    CloneWorkspace,
    DeleteWorkspace,
    mtd,
)

from mantid.geometry import UnitCell


class Calibration:
    def __init__(self, config):
        defaults = {
            "Instrument": "TOPAZ",
            "InstrumentDefinition": None,
            "PeaksTable": None,
            "OutputFolder": "",
            "UnitCellLengths": [5.431, 5.431, 5.431],
            "UnitCellAngles": [90, 90, 90],
        }

        defaults.update(config)

        self.instrument = defaults.get("Instrument")
        self.instrument_definition = defaults.get("InstrumentDefinition")

        self.peaks = defaults.get("PeaksTable")

        self.output_folder = defaults.get("OutputFolder")

        self.calibration_folder = "/SNS/{}/shared/calibration"

        self.a, self.b, self.c = defaults.get("UnitCellLengths")
        self.alpha, self.beta, self.gamma = defaults.get("UnitCellAngles")

        self.interations = 10

    def load_peaks(self):
        ext = os.path.splitext(self.peaks)[1]
        if ext == ".nxs":
            LoadNexus(
                OutputWorkspace="peaks",
                Filename=self.peaks,
            )
        else:
            LoadIsawPeaks(
                OutputWorkspace="peaks",
                Filename=self.peaks,
            )

        FilterPeaks(
            InputWorkspace="peaks",
            OutputWorkspace="peaks_ws",
            FilterVariable="Signal/Noise",
            FilterValue=15,
            Operator=">",
        )

        for peak in mtd["peaks_ws"]:
            peak.setIntensity(0)
            peak.setSigmaIntensity(0)

        runs = np.unique(mtd["peaks_ws"].column("RunNumber")).tolist()

        for i, run in enumerate(runs):
            FilterPeaks(
                InputWorkspace="peaks_ws",
                OutputWorkspace="tmp",
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
            )

            CalculateUMatrix(
                PeaksWorkspace="tmp",
                a=self.a,
                b=self.b,
                c=self.c,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            U = mtd["tmp"].sample().getOrientedLattice().getU().copy()

            for peak in mtd["tmp"]:
                peak.setGoniometerMatrix(peak.getGoniometerMatrix() @ U)

            mtd["tmp"].sample().getOrientedLattice().setU(np.eye(3))

            if i == 0:
                CloneWorkspace(InputWorkspace="tmp", OutputWorkspace="peaks")
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace="tmp",
                    RHSWorkspace="peaks",
                    OutputWorkspace="peaks",
                )

            DeleteWorkspace(Workspace="tmp")

        mtd["peaks"].sample().getOrientedLattice().setU(np.eye(3))

        IndexPeaks(PeaksWorkspace="peaks", Tolerance=0.1)

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )

    def _get_output_folder(self):
        calibration_folder = self.calibration_folder.format(self.instrument)
        output_folder = os.path.join(calibration_folder, self.output_folder)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_ouput(self, ext=".xml"):
        return os.path.join(self._get_output_folder(), "calibration" + ext)

    def calibrate(self):
        SCDCalibratePanels(
            PeakWorkspace="peaks",
            RecalculateUB=True,
            Tolerance=0.2,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            OutputWorkspace="calibration_table",
            DetCalFilename=self._get_ouput(".DetCal"),
            CSVFilename=self._get_ouput(".csv"),
            XmlFilename=self._get_ouput(".xml"),
            CalibrateT0=False,
            SearchRadiusT0=10,
            CalibrateL1=True,
            SearchRadiusL1=0.5,
            CalibrateBanks=True,
            SearchRadiusTransBank=0.5,
            SearchRadiusRotXBank=15,
            SearchRadiusRotYBank=15,
            SearchRadiusRotZBank=15,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.01,
            TuneSamplePosition=False,
            CalibrateSize=self.instrument != "CORELLI",
            SearchRadiusSize=0.15,
            FixAspectRatio=False,
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

        inst = mtd["peaks"].getInstrument()
        sample_pos = inst.getComponentByName("sample-position").getPos()

        components = np.unique(mtd["peaks"].column("BankName")).tolist()
        components += ["sample-position"]

        for component in components:
            MoveInstrumentComponent(
                Workspace=self.instrument,
                ComponentName=component,
                X=-sample_pos[0],
                Y=-sample_pos[1],
                Z=-sample_pos[2],
                RelativePosition=True,
            )

        MoveInstrumentComponent(
            Workspace=self.instrument,
            ComponentName="moderator",
            X=0,
            Y=0,
            Z=-sample_pos[2],
            RelativePosition=True,
        )

        ApplyInstrumentToPeaks(
            InputWorkspace="peaks",
            InstrumentWorkspace=self.instrument,
            OutputWorkspace="peaks",
        )

    def generate_diagnostic(self, iteration):
        uc = UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )

        peak_dict = {}

        banks = mtd["peaks"].column("BankName")

        d_min = np.inf
        d_max = 0

        for i, peak in enumerate(mtd["peaks"]):
            d = peak.getDSpacing()
            d0 = uc.d(*peak.getIntHKL())

            key = banks[i]
            items = peak_dict.get(key)
            if items is None:
                items = [], []
            items[0].append(d0)
            items[1].append(d)

            peak_dict[key] = items

            if d > d_max:
                d_max = d
            if d < d_min:
                d_min = d

        with PdfPages(self._get_ouput("_{}.pdf".format(iteration))) as pdf:
            for key in peak_dict.keys():
                fig, ax = plt.subplots(1, 1, layout="constrained")

                x, y = peak_dict[key]

                x = np.array(x)
                y = np.array(y)

                # ax.plot([d_min, d_max], [0, 0], color='C0')
                ax.plot(x, y / x - 1, ".", color="C1")
                ax.axhline(0, linestyle="-", color="C0")

                ax.set_title(key)
                ax.minorticks_on()
                # ax.set_xlim(d_min, d_max)
                # ax.set_ylim(d_min, d_max)
                ax.set_xlabel(r"$d_0$ [$\AA$]")
                ax.set_ylabel(r"$d/d_0-1$")

                pdf.savefig(fig)
                plt.close()

    def run(self):
        self.load_instrument()
        self.load_peaks()
        for iteration in range(self.interations):
            self.generate_diagnostic(iteration)
            self.calibrate()
        self.generate_diagnostic(self.interations)


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Calibration(params)
    norm.run()
