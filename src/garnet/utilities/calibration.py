import os
import sys

import yaml

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from mantid.simpleapi import (
    SaveNexus,
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
            OutputWorkspace="peaks",
            FilterVariable="Signal/Noise",
            FilterValue=15,
            Operator=">",
        )

        self.goniometer_dict = {}

        for peak in mtd["peaks"]:
            peak.setIntensity(0)
            peak.setSigmaIntensity(0)
            run = peak.getRunNumber()
            R = peak.getGoniometerMatrix().copy()
            self.goniometer_dict[run] = R

    def initialize_peaks(self):
        runs = np.unique(mtd["peaks"].column("RunNumber")).tolist()

        for i, run in enumerate(runs):
            FilterPeaks(
                InputWorkspace="peaks",
                OutputWorkspace="tmp",
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
            )

            for peak in mtd["tmp"]:
                peak.setGoniometerMatrix(np.eye(3))

            CalculateUMatrix(
                PeaksWorkspace="tmp",
                a=self.a,
                b=self.b,
                c=self.c,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            RU = mtd["tmp"].sample().getOrientedLattice().getU().copy()

            for peak in mtd["tmp"]:
                peak.setGoniometerMatrix(RU)

            mtd["tmp"].sample().getOrientedLattice().setU(np.eye(3))

            if i == 0:
                CloneWorkspace(
                    InputWorkspace="tmp", OutputWorkspace="peaks_ws"
                )
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace="tmp",
                    RHSWorkspace="peaks_ws",
                    OutputWorkspace="peaks_ws",
                )

            DeleteWorkspace(Workspace="tmp")

        CloneWorkspace(InputWorkspace="peaks_ws", OutputWorkspace="peaks")

        DeleteWorkspace(Workspace="peaks_ws")

        mtd["peaks"].sample().getOrientedLattice().setU(np.eye(3))

        IndexPeaks(PeaksWorkspace="peaks", Tolerance=0.2)

        # for peak in mtd["peaks"]:
        #     peak.setIntensity(peak.getDSpacing())
        #     peak.setSigmaIntensity(1)

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

    def calibrate_instrument(self, iteration):
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
            TuneSamplePosition=True,
            CalibrateSize=self.instrument != "CORELLI",
            SearchRadiusSize=0.15,
            FixAspectRatio=False,
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

        inst = mtd[self.instrument].getInstrument()
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

        CloneWorkspace(InputWorkspace="peaks", OutputWorkspace="peaks_ws")

        for peak in mtd["peaks_ws"]:
            run = peak.getRunNumber()
            R = self.goniometer_dict[run]
            peak.setGoniometerMatrix(R)

        SaveNexus(
            InputWorkspace="peaks_ws",
            Filename=self._get_ouput("_{}.nxs".format(iteration)),
        )

        DeleteWorkspace(Workspace="peaks_ws")

        SCDCalibratePanels(
            PeakWorkspace="peaks",
            RecalculateUB=False,
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
            SearchRadiusT0=0,
            CalibrateL1=False,
            SearchRadiusL1=0.0,
            CalibrateBanks=False,
            SearchRadiusTransBank=0.0,
            SearchRadiusRotXBank=0,
            SearchRadiusRotYBank=0,
            SearchRadiusRotZBank=0,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.0,
            TuneSamplePosition=False,
            CalibrateSize=False,
            SearchRadiusSize=0.0,
            FixAspectRatio=True,
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

                ax.plot(x, (y / x - 1) * 100, ".", color="C0")
                ax.axhline(0, linestyle="-", color="k", linewidth=1)

                ax.set_title(key)
                ax.minorticks_on()
                ax.set_xlim(d_min, d_max)
                ax.set_ylim(-5, 5)
                ax.set_xlabel(r"$d_0$ [$\AA$]")
                ax.set_ylabel(r"$d/d_0-1$ [%]")

                pdf.savefig(fig)
                plt.close()

    def run(self):
        self.load_instrument()
        self.load_peaks()
        for iteration in range(self.interations):
            self.initialize_peaks()
            self.generate_diagnostic(iteration)
            self.calibrate_instrument(iteration)
        self.generate_diagnostic(self.interations)


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Calibration(params)
    norm.run()
