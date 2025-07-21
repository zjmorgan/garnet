import os
import sys

import yaml

import numpy as np

from mantid.simpleapi import (
    LoadNexus,
    LoadEmptyInstrument,
    LoadParameterFile,
    LoadIsawPeaks,
    ApplyInstrumentToPeaks,
    SCDCalibratePanels,
    MoveInstrumentComponent,
    mtd,
)


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

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )

    def _get_output_folder(self):
        calibration_folder = self.calibration_folder.format(self.instrument)
        return os.path.join(calibration_folder, self.output_folder)

    def _get_ouput(self, ext=".xml"):
        return os.path.join(self._get_output_folder, "calibration" + ext)

    def calibrate(self):
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
            SearchRadiusT0=10,
            CalibrateL1=True,
            SearchRadiusL1=0.2,
            CalibrateBanks=True,
            SearchRadiusTransBank=0.2,
            SearchRadiusRotXBank=5,
            SearchRadiusRotYBank=5,
            SearchRadiusRotZBank=5,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.01,
            TuneSamplePosition=False,
            CalibrateSize=self.instrument != "CORELLI",
            SearchRadiusSize=0.1,
            FixAspectRatio=False,
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

        inst = mtd["peaks"].getInstrument()
        sample_pos = inst.getComponentByName("sample-position").getPos()

        components = np.unique(mtd[self.peaks].column("BankName")).tolist()
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

    def run(self):
        self.load_instrument()
        self.calibrate()


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Calibration(params)
    norm.run()
