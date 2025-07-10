import os
import sys

import yaml

import numpy as np

import scipy.sparse
import scipy.special
import scipy.optimize

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
    NormaliseSpectra,
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
    SmoothNeighbours,
    CopyInstrumentParameters,
    ClearMaskFlag,
    mtd,
)


class WhittakerHenderson:
    def _build_difference_operator(self, n, d):
        """Build sparse d-th order difference operator D."""
        diagonals = [
            (-1) ** i * scipy.special.comb(d, i, exact=True) * np.ones(n - d)
            for i in range(d + 1)
        ]
        offsets = np.arange(d + 1)
        return scipy.sparse.diags(diagonals, offsets, shape=(n - d, n))

    def _build_system_matrix(self, n, lamda, d):
        """Build system matrix A = I + λ DᵗD."""
        D = self._build_difference_operator(n, d)
        I = scipy.sparse.diags([1.0], [0], shape=(n, n))
        return I + lamda * (D.T @ D)

    def smooth(self, y, lamda, d=2):
        """Apply Whittaker–Henderson smoothing."""
        y = np.asarray(y)
        A = self._build_system_matrix(len(y), lamda, d)
        return scipy.sparse.linalg.spsolve(A, y)

    def smoother_matrix(self, n, lamda, d=2):
        """Return dense smoother matrix S = (I + λ DᵗD)⁻¹."""
        A = self._build_system_matrix(n, lamda, d)
        return scipy.sparse.linalg.inv(A).toarray()

    def loocv(self, y, lamda, d=2):
        """Leave-one-out cross-validation score and smoothed fit."""
        y = np.asarray(y)
        n = len(y)
        S = self.smoother_matrix(n, lamda, d)
        y_hat = S @ y
        S_diag = np.diag(S)
        residuals = (y - y_hat) / (1 - S_diag)
        s_cv = np.sqrt(np.mean(residuals**2))
        return s_cv, y_hat

    def optimize(self, data, d=2, bounds=(2, 10), verbose=False):
        data = np.asarray(data)
        n_rows = data.shape[0]

        lamdas = np.logspace(*bounds, 25)

        loocv_scores = []
        for lamda in lamdas:
            scores = []
            for i in range(n_rows):
                try:
                    s_cv, _ = self.loocv(data[i], lamda, d)
                    scores.append(s_cv)
                except Exception:
                    pass
            loocv_scores.append(np.mean(scores) if scores else np.inf)

        best_idx = int(np.argmin(loocv_scores))
        best_lamda = lamdas[best_idx]

        return best_lamda


class Vanadium:
    def __init__(
        self,
        Instrument="TOPAZ",
        VanadiumIPTS=31856,
        VanadiumRuns=None,
        NoSampleIPTS=31856,
        NoSampleRuns=None,
        OutputFolder="",
        DetectorCalibration=None,
        TubeCalibration=None,
        InstrumentDefinition=None,
        SampleShape="sphere",
        Diameter=4,
        Height=None,
        BeamDiameter=None,
        MomentumLimits=[1.8, 18],
        MaskOptions=None,
        Grouping=[4, 4],
    ):
        self.instrument = Instrument

        self.van_ipts = VanadiumIPTS
        self.van_nos = VanadiumRuns

        self.bkg_ipts = NoSampleIPTS
        self.bkg_nos = NoSampleRuns

        self.output_folder = OutputFolder

        self.detector_calibration = DetectorCalibration
        self.tube_calibration = TubeCalibration
        self.instrument_definition = InstrumentDefinition

        self.sample_shape = SampleShape
        self.diameter = Diameter
        self.height = Height

        self.beam_diameter = BeamDiameter

        self.mask_options = MaskOptions or {}
        self.x_bins, self.y_bins = Grouping

        self.file_folder = "/SNS/{}/IPTS-{}/nexus/"
        self.file_name = "{}_{}.nxs.h5"
        self.vanadium_folder = "/SNS/{}/shared/Vanadium"

        self.n_bins = 10000

        self.k_min, self.k_max = MomentumLimits
        self.k_step = (self.k_max - self.k_min) / self.n_bins

        self.lamda_min = 2 * np.pi / self.k_max
        self.lamda_max = 2 * np.pi / self.k_min
        self.lamda_step = (self.lamda_max - self.lamda_min) / self.n_bins

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
        if self.mask_options.get("Banks") is not None:
            MaskBTP(Workspace=self.instrument, Bank=self.mask_options["Banks"])

        if self.mask_options.get("Pixels") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Pixel=self._join(self.mask_options["Pixels"]),
            )

        if self.mask_options.get("Tubes") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Tube=self._join(self.mask_options["Tubes"]),
            )

        if self.mask_options.get("BankTube") is not None:
            for bank, tube in self.mask_options["BankTube"]:
                MaskBTP(
                    Workspace=self.instrument, Bank=str(bank), Tube=str(tube)
                )

        if self.mask_options.get("BankTubePixel") is not None:
            for bank, tube, pixel in self.mask_options["BankTubePixel"]:
                MaskBTP(
                    Workspace=self.instrument,
                    Bank=str(bank),
                    Tube=str(tube),
                    Pixel=str(pixel),
                )

        ExtractMask(InputWorkspace=self.instrument, OutputWorkspace="mask")

        ClearMaskFlag(Workspace=self.instrument)

        SmoothNeighbours(
            InputWorkspace="mask",
            OutputWorkspace="pixels",
            SumPixelsX=self.x_bins,
            SumPixelsY=self.y_bins,
        )

        MaskDetectors(Workspace="pixels", MaskedWorkspace="mask")

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

        CopyInstrumentParameters(
            InputWorkspace=self.instrument,
            OutputWorkspace=workspace,
        )

        SmoothNeighbours(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            SumPixelsX=self.x_bins,
            SumPixelsY=self.y_bins,
        )

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

        Divide(
            LHSWorkspace="background",
            RHSWorkspace="pixels",
            OutputWorkspace="background",
            AllowDifferentNumberSpectra=True,
            WarnOnZeroDivide=False,
        )

    def _vanadium_niobium_lattice_constant(self, x):
        y = 2 * x - 1
        a = 3.19199921 + 0.13954993 * y - 0.02242883 * y**2
        return a

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

        a = self._vanadium_niobium_lattice_constant(x)

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

        if self.sample_shape == "cylinder":
            CylinderAbsorption(
                InputWorkspace="vanadium",
                OutputWorkspace="corr",
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

        wh = WhittakerHenderson()
        y = mtd["spectra"].extractY()
        lamda = wh.optimize(y)
        for i in range(y.shape[0]):
            z = wh.smooth(y[i], lamda)
            mtd["spectra"].setY(i, z)

        Rebin(
            InputWorkspace="spectra",
            OutputWorkspace="spectra",
            Params=[self.lamda_min, self.lamda_step, self.lamda_max],
            PreserveEvents=False,
        )

        Rebin(
            InputWorkspace="spectra",
            OutputWorkspace="norm",
            Params=[self.lamda_min, self.lamda_max, self.lamda_max],
            PreserveEvents=False,
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

        MaskDetectorsIf(
            InputWorkspace="flux",
            Operator="LessEqual",
            OutputWorkspace="flux",
        )

        RemoveMaskedSpectra(
            InputWorkspace="flux",
            MaskedWorkspace="flux",
            OutputWorkspace="flux",
        )

        SortEvents(InputWorkspace="flux", SortBy="X Value")

        Rebin(
            InputWorkspace="flux",
            OutputWorkspace="flux",
            Params=[self.k_min, self.k_step, self.k_max],
            PreserveEvents=False,
        )

        wh = WhittakerHenderson()
        y = mtd["flux"].extractY()
        lamda = wh.optimize(y)
        for i in range(y.shape[0]):
            z = wh.smooth(y[i], lamda)
            mtd["flux"].setY(i, z)

        IntegrateFlux(
            InputWorkspace="flux",
            NPoints=self.n_bins,
            OutputWorkspace="flux",
        )

        NormaliseSpectra(InputWorkspace="flux", OutputWorkspace="flux")

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


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Vanadium(**params)
    norm.run()
