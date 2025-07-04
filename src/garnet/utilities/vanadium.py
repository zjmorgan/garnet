import os

import numpy as np

from mantid.simpleapi import (
    Load,
    LoadNexus,
    SaveNexus,
    LoadEmptyInstrument,
    LoadIsawDetCal,
    LoadParameterFile,
    ApplyCalibration,
    CompressEvents,
    NormaliseByCurrent,
    SortEvents,
    IntegrateFlux,
    Rebin,
    Minus,
    Divide,
    ConvertUnits,
    CropWorkspace,
    AddSampleLog,
    RemoveLogs,
    CreateGroupingWorkspace,
    GroupDetectors,
    RemoveMaskedSpectra,
    MaskDetectors,
    MaskDetectorsIf,
    MaskBTP,
    ExtractMask,
    SetSample,
    SetBeam,
    SphericalAbsorption,
    CylinderAbsorption,
    SolidAngle,
    CopyInstrumentParameters,
    mtd,
)


class Vanadium:
    def __init__(
        self,
        instrument="TOPAZ",
        van_ipts=31856,
        van_nos=None,
        bkg_ipts=31856,
        bkg_nos=None,
        output_folder="",
        calibration_folder="",
        detector_calibration=None,
        tube_calibration=None,
        instrument_definition=None,
        sample_shape="sphere",
        diameter=4,
        height=None,
        beam_diameter=None,
        k_limits=[1.8, 18],
        mask_options=None,
    ):
        self.instrument = instrument
        self.van_ipts = van_ipts
        self.van_nos = van_nos
        self.bkg_ipts = bkg_ipts
        self.bkg_nos = bkg_nos
        self.output_folder = output_folder
        self.detector_calibration = detector_calibration
        self.tube_calibration = tube_calibration
        self.instrument_definition = instrument_definition
        self.sample_shape = sample_shape
        self.diameter = diameter
        self.height = height
        self.beam_diameter = beam_diameter
        self.k_min, self.k_max = k_limits

        self.mask_options = mask_options or {}
        self.file_folder = "/SNS/{}/IPTS-{}/nexus/"
        self.file_name = "{}_{}.nxs.h5"
        self.vanadium_folder = "/SNS/{}/shared/Vanadium"

        self.lamda_min = 2 * np.pi / self.k_max
        self.lamda_max = 2 * np.pi / self.k_min
        self.lamda_step = (self.lamda_max - self.lamda_min) / 500

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )
        CreateGroupingWorkspace(
            InputWorkspace=self.instrument,
            GroupDetectorsBy="bank",
            OutputWorkspace="group",
        )

    def _join(self, ranges):
        return ",".join(
            [
                "{}-{}".format(*r) if isinstance(r, list) else str(r)
                for r in ranges
            ]
        )

    def apply_masks(self):
        if self.mask_options.get("banks") is not None:
            MaskBTP(Workspace=self.instrument, Bank=self.mask_options["banks"])

        if self.mask_options.get("pixels") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Pixel=self._join(self.mask_options["pixels"]),
            )

        if self.mask_options.get("tubes") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Tube=self._join(self.mask_options["tubes"]),
            )

        if self.mask_options.get("bank/tube") is not None:
            for bank, tube in self.mask_options["bank/tube"]:
                MaskBTP(
                    Workspace=self.instrument, Bank=str(bank), Tube=str(tube)
                )

        if self.mask_options.get("bank/tube/pixel") is not None:
            for bank, tube, pixel in self.mask_options["bank/tube/pixel"]:
                MaskBTP(
                    Workspace=self.instrument,
                    Bank=str(bank),
                    Tube=str(tube),
                    Pixel=str(pixel),
                )

        ExtractMask(InputWorkspace=self.instrument, OutputWorkspace="mask")

    def apply_calibration(self):
        if self.tube_calibration is not None:
            LoadNexus(
                Filename=self.tube_calibration,
                OutputWorkspace="tube_table",
            )
            ApplyCalibration(
                Workspace=self.instrument, CalibrationTable="tube_table"
            )

        if self.detector_calibration is not None:
            ext = os.path.splitext(self.detector_calibration)[1]
            if ext == ".xml":
                LoadParameterFile(
                    Workspace=self.instrument,
                    Filename=self.detector_calibration,
                )
            else:
                LoadIsawDetCal(
                    InputWorkspace=self.instrument,
                    Filename=self.detector_calibration,
                )

    def load_runs(self, workspace, ipts, run_nos):
        files_to_load = "+".join(
            [
                os.path.join(
                    self.file_folder.format(self.instrument, ipts),
                    self.file_name.format(self.instrument, run_no),
                )
                for run_no in run_nos
            ]
        )

        Load(Filename=files_to_load, NumberOfBins=1, OutputWorkspace=workspace)

        MaskDetectors(Workspace=workspace, MaskedWorkspace="mask")

        ConvertUnits(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            Target="Momentum",
        )

        CropWorkspace(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            XMin=self.k_min,
            XMax=self.k_max,
        )

        CompressEvents(
            InputWorkspace=workspace, Tolerance=1e-3, OutputWorkspace=workspace
        )

        NormaliseByCurrent(InputWorkspace=workspace, OutputWorkspace=workspace)

        pc = mtd[workspace].run().getProperty("gd_prtn_chrg")

        val = pc.valueAsStr
        uni = pc.units

        RemoveLogs(Workspace=workspace)

        logs = ["gd_prtn_chrg", "NormalizationFactor"]
        for log in logs:
            AddSampleLog(
                Workspace=workspace,
                LogName=log,
                LogText=val,
                LogUnit=uni,
                LogType="Number",
                NumberType="Double",
            )

    def subtract_background(self):
        Minus(
            LHSWorkspace="vanadium",
            RHSWorkspace="background",
            OutputWorkspace="vanadium",
        )

    def set_sample_geometry(self, x=0):
        if self.sample_shape == "sphere":
            shape = {
                "Shape": "Sphere",
                "Radius": self.diameter * 0.05,
                "Center": [0.0, 0.0, 0.0],
            }
        else:
            shape = {
                "Shape": "Cylinder",
                "Height": self.height * 0.1,
                "Radius": self.diameter * 0.05,
                "Axis": [0.0, 0.0, 1.0],
                "Center": [0.0, 0.0, 0.0],
            }

        a = (
            3.19199921
            + 0.13954993 * (2 * x - 1)
            - 0.02242883 * (2 * x - 1) ** 2
        )

        material = {
            "ChemicalFormula": "V{} Nb{}".format(1 - x, x),
            "ZParameter": 2.0,
            "UnitCellVolume": float(a**3),
        }

        SetSample(InputWorkspace="vanadium", Geometry=shape, Material=material)

        if self.beam_diameter is not None:
            beam = {"Shape": "Circle", "Radius": self.beam_diameter * 0.05}
            SetBeam(InputWorkspace="vanadium", Geometry=beam)

        mat = mtd["vanadium"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        self.sigma_a = sigma_a
        self.sigma_s = sigma_s

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective  # A^-3
        N = mat.totalAtoms

        self.n = n

        V = np.abs(
            mtd["vanadium"].sample().getShape().volume() * 100**3
        )  # cm^3

        rho = (n / N) / 0.6022 * M
        m = rho * V
        r = np.cbrt(0.75 / np.pi * V)

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        mu = mat.numberDensityEffective * (
            mat.totalScatterXSection() + mat.absorbXSection(1.8)
        )

        print("V\n")
        print("absoption cross section: {:.4f} barn\n".format(sigma_a))
        print("scattering cross section: {:.4f} barn\n".format(sigma_s))

        print("linear absorption coefficient: {:.4f} 1/cm\n".format(mu_a))
        print("linear scattering coefficient: {:.4f} 1/cm\n".format(mu_s))
        print("absorption parameter: {:.4f} \n".format(mu * r))

        print("total atoms: {:.4f}\n".format(N))
        print("molar mass: {:.4f} g/mol\n".format(M))
        print("number density: {:.4f} 1/A^3\n".format(n))

        print("mass density: {:.4f} g/cm^3\n".format(rho))
        print("volume: {:.4f} cm^3\n".format(V))
        print("mass: {:.4f} g\n".format(m))

    def apply_absorption_correction(self):
        ConvertUnits(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Target="Wavelength",
        )

        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Params=[self.lamda_min, self.lamda_step, self.lamda_max],
            PreserveEvents=True,
        )

        # MonteCarloAbsorption(
        #     InputWorkspace="vanadium",
        #     ResimulateTracksForDifferentWavelengths=False,
        #     SimulateScatteringPointIn="SampleOnly",
        #     OutputWorkspace="corr",
        # )

        if self.sample_shape == "cylinder":
            CylinderAbsorption(
                InputWorkspace="vanadium",
                OutputWorkspace="corr",
                AttenuationXSection=self.sigma_a,
                ScatteringXSection=self.sigma_s,
                SampleNumberDensity=self.n,
                NumberOfWavelengthPoints=5,
                CylinderSampleHeight=self.height * 0.1,
                CylinderSampleRadius=self.diameter * 0.05,
                NumberOfSlices=8,
                NumberOfAnnuli=8,
            )
        else:
            SphericalAbsorption(
                InputWorkspace="vanadium",
                OutputWorkspace="corr",
                AttenuationXSection=self.sigma_a,
                ScatteringXSection=self.sigma_s,
                SampleNumberDensity=self.n,
                SphericalSampleRadius=self.diameter * 0.05,
            )

        Divide(
            LHSWorkspace="vanadium",
            RHSWorkspace="corr",
            OutputWorkspace="vanadium",
        )

        GroupDetectors(
            InputWorkspace="vanadium",
            CopyGroupingFromWorkspace="group",
            Behaviour="Sum",
            PreserveEvents=False,
            OutputWorkspace="spectra",
        )

        Rebin(
            InputWorkspace="spectra",
            OutputWorkspace="norm",
            Params=[self.lamda_min, self.lamda_max, self.lamda_max],
            PreserveEvents=True,
        )

        Divide(
            LHSWorkspace="spectra",
            RHSWorkspace="norm",
            OutputWorkspace="spectra",
        )

        ConvertUnits(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Target="Momentum",
        )

        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Params=[self.k_min, self.k_max, self.k_max],
            PreserveEvents=True,
        )

    def process_data(self):
        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="solid_angle",
            Params=[self.k_min, self.k_max, self.k_max],
            PreserveEvents=False,
        )

        GroupDetectors(
            InputWorkspace="vanadium",
            CopyGroupingFromWorkspace="group",
            Behaviour="Sum",
            PreserveEvents=True,
            OutputWorkspace="flux",
        )

        RemoveMaskedSpectra(
            InputWorkspace="flux",
            MaskedWorkspace="flux",
            OutputWorkspace="flux",
        )

        Rebin(
            InputWorkspace="flux",
            OutputWorkspace="flux",
            Params=[self.k_min, self.k_max, self.k_max],
            PreserveEvents=True,
        )

        flux = mtd["flux"]
        for i in range(flux.getNumberHistograms()):
            el = flux.getSpectrum(i)
            if flux.readY(i)[0] > 0:
                el.divide(flux.readY(i)[0], flux.readE(i)[0])

        SortEvents(InputWorkspace="flux", SortBy="X Value")

        IntegrateFlux(
            InputWorkspace="flux", NPoints=500, OutputWorkspace="flux"
        )

        MaskDetectorsIf(
            InputWorkspace="solid_angle",
            Operator="LessEqual",
            OutputWorkspace="solid_angle",
        )

    def finalize_and_save(self):
        vanadium_folder = self.vanadium_folder.format(self.instrument)
        output_folder = os.path.join(vanadium_folder, self.output_folder)
        workspaces = ["background", "flux", "spectra", "solid_angle"]
        for workspace in workspaces:
            filename = os.path.join(output_folder, workspace + ".nxs")
            SaveNexus(InputWorkspace=workspace, Filename=filename)

    def run(self):
        self.load_instrument()
        self.apply_masks()
        self.apply_calibration()
        self.load_runs("vanadium", self.van_ipts, self.van_nos)
        self.load_runs("background", self.bkg_ipts, self.bkg_nos)
        self.subtract_background()
        self.set_sample_geometry()
        self.apply_absorption_correction()
        self.process_data()
        self.finalize_and_save()


params = {
    "instrument": "MANDI",
    "van_ipts": 8776,
    "van_nos": [11935],
    "bkg_ipts": 8776,
    "bkg_nos": [11936],
    "output_folder": "2025A_3mm_sphere_2A",
    "detector_calibration": "/SNS/MANDI/shared/calibration/2025A/MANDI_2025A.DetCal",
    "tube_calibration": None,
    "instrument_definition": None,
    "sample_shape": "sphere",
    "diameter": 3,  # mm
    "height": None,  # mm
    "beam_diameter": 1,  # mm
    "k_limits": [2.1, 6.28],
    "mask_options": {
        "pixels": [[0, 18], [237, 255]],
        "tubes": [[0, 18], [237, 255]],
        "banks": None,
        "bank/tube": None,
        "bank/tube/pixel": None,
    },
}

norm = Vanadium(**params)
norm.run()
