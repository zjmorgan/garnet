import os

import numpy as np

import scipy.spatial.transform
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.ndimage
import scipy.linalg
import scipy.stats

import skimage.measure

from lmfit import Minimizer, Parameters, fit_report

from mantid.simpleapi import mtd
from mantid import config

config["Q.convention"] = "Crystallography"

config["MultiThreaded.MaxCores"] == "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TBB_THREAD_ENABLED"] = "0"

from garnet.plots.peaks import PeakPlot, PeakProfilePlot, PeakCentroidPlot
from garnet.config.instruments import beamlines
from garnet.reduction.ub import UBModel, Optimization, Reorient, lattice_group
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.data import DataModel
from garnet.reduction.plan import SubPlan
from garnet.reduction.parallel import ParallelProcessor


class Integration(SubPlan):
    def __init__(self, plan):
        super(Integration, self).__init__(plan)

        self.params = plan["Integration"]
        self.output = plan["OutputName"] + "_integration"

        self.validate_params()

    def validate_params(self):
        assert self.params["Cell"] in lattice_group.keys()
        assert self.params["Centering"] in centering_reflection.keys()
        assert self.params["MinD"] > 0
        assert self.params["Radius"] > 0

        if self.params.get("ModVec1") is None:
            self.params["ModVec1"] = [0, 0, 0]
        if self.params.get("ModVec2") is None:
            self.params["ModVec2"] = [0, 0, 0]
        if self.params.get("ModVec3") is None:
            self.params["ModVec3"] = [0, 0, 0]

        if self.params.get("MaxOrder") is None:
            self.params["MaxOrder"] = 0
        if self.params.get("CrossTerms") is None:
            self.params["CrossTerms"] = False

        assert len(self.params["ModVec1"]) == 3
        assert len(self.params["ModVec2"]) == 3
        assert len(self.params["ModVec3"]) == 3

        assert self.params["MaxOrder"] >= 0
        assert type(self.params["CrossTerms"]) is bool

        if self.params.get("ProfileFit") is None:
            self.params["ProfileFit"] = True

    @staticmethod
    def integrate_parallel(plan, runs, proc):
        plan["Runs"] = runs
        plan["ProcName"] = "_p{}".format(proc)

        instance = Integration(plan)
        instance.proc = proc
        instance.n_proc = 1

        return instance.laue_integrate()

    def integrate(self, n_proc=1):
        data = DataModel(beamlines[self.plan["Instrument"]])

        instance = Integration(self.plan)
        instance.n_proc = n_proc

        if data.laue:
            return instance.laue_integrate()
        else:
            return instance.monochromatic_integrate()

    def integrate_peaks(self, data):
        pp = ParallelProcessor(n_proc=self.n_proc)
        return pp.process_dict(data, self.fit_peaks)

    @staticmethod
    def combine_parallel(plan, files):
        instance = Integration(plan)

        return instance.combine(files)

    def combine(self, files):
        output_file = self.get_output_file()
        result_file = self.get_file(output_file, "")

        peaks = PeaksModel()

        for file in files:
            peaks.load_peaks(file, "tmp")
            peaks.combine_peaks("tmp", "combine")

        for file in files:
            os.remove(file)
            os.remove(os.path.splitext(file)[0] + ".mat")

        if mtd.doesExist("combine"):
            peaks.save_peaks(result_file, "combine")

            opt = Optimization("combine")
            opt.optimize_lattice(self.params["Cell"])

            ub_file = os.path.splitext(result_file)[0] + ".mat"

            ub = UBModel("combine")
            ub.save_UB(ub_file)

        self.cleanup()

    def laue_integrate(self):
        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        self.make_plot = True
        self.peak_plot = PeakPlot()

        runs = self.plan["Runs"]

        self.run = 0
        self.runs = len(runs)

        result_file = self.get_file(output_file, "")

        for run in runs:
            self.run += 1

            self.status = "{}: {:}/{:}".format(self.proc, self.run, len(runs))

            data.load_data(
                "data", self.plan["IPTS"], run, self.plan.get("Grouping")
            )

            data.load_generate_normalization(
                self.plan["VanadiumFile"], self.plan.get("FluxFile")
            )

            data.apply_calibration(
                "data",
                self.plan.get("DetectorCalibration"),
                self.plan.get("TubeCalibration"),
            )

            data.preprocess_detectors("data")

            data.crop_for_normalization("data")

            data.apply_mask("data", self.plan.get("MaskFile"))

            data.group_pixels("data")

            data.load_background(self.plan.get("BackgroundFile"), "data")

            data.load_clear_UB(self.plan["UBFile"], "data", run)

            lamda_min, lamda_max = data.wavelength_band

            d_min = self.params["MinD"]

            centering = self.params["Centering"]

            cell = self.params["Cell"]

            if self.params.get("Recalibrate"):
                ub = UBModel("data")

                const = ub.get_lattice_parameters()

                min_d, max_d = ub.get_primitive_cell_length_range(centering)

                const = ub.convert_conventional_to_primitive(centering)

                data.convert_to_Q_sample("data", "md", lorentz_corr=True)

                peaks.find_peaks("md", "peaks", max_d)

                peaks.integrate_peaks("md", "peaks", self.params["Radius"])

                peaks.remove_weak_peaks("peaks", 10)

                ub = UBModel("peaks")
                ub.determine_UB_with_lattice_parameters(*const)
                ub.index_peaks()
                ub.transform_primitive_to_conventional(centering)

                Reorient("peaks", cell)

                ub.copy_UB("data")

                ub_file = self.get_diagnostic_file("run#{}_ub".format(run))
                ub_file = os.path.splitext(ub_file)[0] + ".mat"

                ub.save_UB(ub_file)

                data.load_clear_UB(ub_file, "data", run)

            # pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))

            # peaks.save_peaks(pk_file, "peaks")

            # md_file = self.get_diagnostic_file("run#{}_data".format(run))

            # data.save_histograms(md_file, "md")

            data.convert_to_Q_sample("data", "md", lorentz_corr=False)

            peaks.predict_peaks(
                "data",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            self.peaks, self.data = peaks, data

            r_cut = self.params["Radius"]

            if not self.params.get("Recalibrate"):
                est_file = self.get_plot_file("centroid#{}".format(run))

                self.estimate_peak_centroid("peaks", r_cut, d_min, est_file)

                ub_file = self.get_diagnostic_file("run#{}_ub".format(run))

                opt = Optimization("peaks")
                opt.optimize_lattice("Fixed")

                ub_file = os.path.splitext(ub_file)[0] + ".mat"

                ub = UBModel("peaks")
                ub.save_UB(ub_file)

                data.load_clear_UB(ub_file, "data", run)

            peaks.predict_peaks(
                "data",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            self.predict_add_satellite_peaks(lamda_min, lamda_max)

            est_file = self.get_plot_file("profile#{}".format(run))

            params = self.estimate_peak_size("peaks", r_cut, est_file)

            fit = self.params["ProfileFit"]

            peak_dict = self.extract_peak_info("peaks", params, True, fit)

            results = self.integrate_peaks(peak_dict)

            self.update_peak_info("peaks", results)

            data.delete_workspace("data")

            peaks.combine_peaks("peaks", "combine")

            pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))

            peaks.save_peaks(pk_file, "peaks")

            data.delete_workspace("peaks")

            data.delete_workspace("md")

        peaks.remove_weak_peaks("combine")

        peaks.save_peaks(result_file, "combine")

        # ---

        if mtd.doesExist("combine"):
            opt = Optimization("combine")
            opt.optimize_lattice(cell)

            ub_file = os.path.splitext(result_file)[0] + ".mat"

            ub = UBModel("combine")
            ub.save_UB(ub_file)

        mtd.clear()

        return result_file

    def predict_add_satellite_peaks(self, lamda_min, lamda_max):
        if self.params["MaxOrder"] > 0:
            sat_min_d = self.params["MinD"]
            if self.params.get("SatMinD") is not None:
                sat_min_d = self.params["SatMinD"]

            self.peaks.predict_satellite_peaks(
                "peaks",
                "md",
                self.params["Centering"],
                lamda_min,
                lamda_max,
                sat_min_d,
                self.params["ModVec1"],
                self.params["ModVec2"],
                self.params["ModVec3"],
                self.params["MaxOrder"],
                self.params["CrossTerms"],
            )

    def monochromatic_integrate(self):
        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        runs = self.plan["Runs"]

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        self.run = 0
        self.runs = len(runs)

        if self.plan["Instrument"] == "WAND²":
            self.runs = 1
            self.run += 1

            data.load_data(
                "data", self.plan["IPTS"], runs, self.plan.get("Grouping")
            )

            data.load_generate_normalization(self.plan["VanadiumFile"], "data")

            data.convert_to_Q_sample("data", "md", lorentz_corr=True)

            md_file = self.get_diagnostic_file("run#{}_data".format(self.run))
            data.save_histograms(md_file, "md", sample_logs=True)

        else:
            for run in runs:
                self.run += 1

                data.load_data(
                    "data", self.plan["IPTS"], run, self.plan.get("Grouping")
                )

                data.load_generate_normalization(
                    self.plan["VanadiumFile"], "data"
                )

                data.convert_to_Q_sample("data", "md", lorentz_corr=True)

                if self.plan.get("UBFile") is None:
                    UB_file = output_file.replace(".nxs", ".mat")
                    data.save_UB(UB_file, "md_data")
                    self.plan["UBFile"] = UB_file

                data.load_clear_UB(self.plan["UBFile"], "md")

                peaks.predict_peaks(
                    "md",
                    "peaks",
                    self.params["Centering"],
                    self.params["MinD"],
                    lamda_min,
                    lamda_max,
                )

                if self.params["MaxOrder"] > 0:
                    peaks.predict_satellite_peaks(
                        "peaks",
                        "md",
                        self.params["MinD"],
                        lamda_min,
                        lamda_max,
                        self.params["ModVec1"],
                        self.params["ModVec2"],
                        self.params["ModVec3"],
                        self.params["MaxOrder"],
                        self.params["CrossTerms"],
                    )

                self.peaks, self.data = peaks, data

                params = self.estimate_peak_size("peaks", "md")

                self.fit_peaks("peaks", params)

                peaks.combine_peaks("peaks", "combine")

                md_file = self.get_diagnostic_file("run#{}_data".format(run))
                data.save_histograms(md_file, "md", sample_logs=True)

                pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))
                peaks.save_peaks(pk_file, "peaks")

        if self.plan["Instrument"] != "WAND²":
            peaks.remove_weak_peaks("combine")

            peaks.save_peaks(output_file, "combine")

        mtd.clear()

        return output_file

    def get_file(self, file, ws=""):
        """
        Update filename with identifier name and optional workspace name.

        Parameters
        ----------
        file : str
            Original file name.
        ws : str, optional
            Name of workspace. The default is ''.

        Returns
        -------
        output_file : str
            File with updated name for identifier and workspace name.

        """

        if len(ws) > 0:
            ws = "_" + ws

        return self.append_name(file).replace(".nxs", ws + ".nxs")

    def append_name(self, file):
        """
        Update filename with identifier name.

        Parameters
        ----------
        file : str
            Original file name.

        Returns
        -------
        output_file : str
            File with updated name for identifier name.

        """

        append = (
            self.cell_centering_name()
            + self.modulation_name()
            + self.resolution_name()
        )

        name, ext = os.path.splitext(file)

        return name + append + ext

    def cell_centering_name(self):
        """
        Lattice and reflection condition.

        Returns
        -------
        lat_ref : str
            Underscore separated strings.

        """

        cell = self.params["Cell"]
        centering = self.params["Centering"]

        return "_" + cell + "_" + centering

    def modulation_name(self):
        """
        Modulation vectors.

        Returns
        -------
        mod : str
            Underscore separated vectors and max order

        """

        mod = ""

        max_order = self.params.get("MaxOrder")
        mod_vec_1 = self.params.get("ModVec1")
        mod_vec_2 = self.params.get("ModVec2")
        mod_vec_3 = self.params.get("ModVec3")
        cross_terms = self.params.get("CrossTerms")

        if max_order > 0:
            for vec in [mod_vec_1, mod_vec_2, mod_vec_3]:
                if np.linalg.norm(vec) > 0:
                    mod += "_({},{},{})".format(*vec)
            if cross_terms:
                mod += "_mix"

        return mod

    def resolution_name(self):
        """
        Minimum d-spacing and starting radii

        Returns
        -------
        res_rad : str
            Underscore separated strings.

        """

        min_d = self.params["MinD"]
        max_r = self.params["Radius"]

        return "_d(min)={:.2f}".format(min_d) + "_r(max)={:.2f}".format(max_r)

    def estimate_peak_centroid(self, peaks_ws, r_cut, d_min, filename):
        peak_dict = self.extract_peak_info(peaks_ws, r_cut)

        center = PeakCentroid()

        c0, c1, c2, Q = center.fit(peak_dict)

        plot = PeakCentroidPlot(c0, c1, c2, Q, r_cut, d_min)
        plot.save_plot(filename)

        offsets = c0, c1, c2, Q
        self.update_peak_offsets(peaks_ws, offsets, peak_dict)

    def estimate_peak_size(self, peaks_ws, r_cut, filename):
        peak_dict = self.extract_peak_info(peaks_ws, r_cut)

        roi = PeakProfile(r_cut)

        params = roi.fit(peak_dict)

        x, y, e, Q, k = roi.extract_fit()

        plot = PeakProfilePlot(x, y, e, Q, k, params, r_cut)
        plot.save_plot(filename)

        return params

    def quick_fit(self, x0, x1, x2, d, n):
        d0 = x0[1, 0, 0] - x0[0, 0, 0]
        d1 = x1[0, 1, 0] - x1[0, 0, 0]
        d2 = x2[0, 0, 1] - x2[0, 0, 0]

        y1d_0 = np.nansum(d, axis=(1, 2)) / np.nansum(n, axis=(1, 2))
        e1d_0 = np.sqrt(np.nansum(d, axis=(1, 2))) / np.nansum(n, axis=(1, 2))
        y1d_0_fit = y1d_0.copy()

        y1d_1 = np.nansum(d, axis=(0, 2)) / np.nansum(n, axis=(0, 2))
        e1d_1 = np.sqrt(np.nansum(d, axis=(0, 2))) / np.nansum(n, axis=(0, 2))
        y1d_1_fit = y1d_1.copy()

        y1d_2 = np.nansum(d, axis=(0, 1)) / np.nansum(n, axis=(0, 1))
        e1d_2 = np.sqrt(np.nansum(d, axis=(0, 1))) / np.nansum(n, axis=(0, 1))
        y1d_2_fit = y1d_2.copy()

        y1d = [
            (y1d_0_fit, y1d_0, e1d_0),
            (y1d_1_fit, y1d_1, e1d_1),
            (y1d_2_fit, y1d_2, e1d_2),
        ]

        I1d_0 = np.nansum(y1d_0[2:-2] - np.nanmedian(y1d_0[0::8])) * d0
        I1d_1 = np.nansum(y1d_1[2:-2] - np.nanmedian(y1d_1[0::8])) * d1
        I1d_2 = np.nansum(y1d_2[2:-2] - np.nanmedian(y1d_2[0::8])) * d2

        I1d = [I1d_0, I1d_1, I1d_2]

        s1d_0 = np.nansum(e1d_0[2:-2] ** 2 + np.nanmedian(y1d_0[0::8])) * d0
        s1d_1 = np.nansum(e1d_1[2:-2] ** 2 + np.nanmedian(y1d_1[0::8])) * d1
        s1d_2 = np.nansum(e1d_2[2:-2] ** 2 + np.nanmedian(y1d_2[0::8])) * d2

        s1d = [s1d_0, s1d_1, s1d_2]

        y2d_0 = np.nansum(d, axis=0) / np.nansum(n, axis=0)
        e2d_0 = np.sqrt(np.nansum(d, axis=0)) / np.nansum(n, axis=0)
        y2d_0_fit = y2d_0.copy()

        y2d_1 = np.nansum(d, axis=1) / np.nansum(n, axis=1)
        e2d_1 = np.sqrt(np.nansum(d, axis=1)) / np.nansum(n, axis=1)
        y2d_1_fit = y2d_1.copy()

        y2d_2 = np.nansum(d, axis=2) / np.nansum(n, axis=2)
        e2d_2 = np.sqrt(np.nansum(d, axis=2)) / np.nansum(n, axis=2)
        y2d_2_fit = y2d_2.copy()

        y2d = [
            (y2d_0_fit, y2d_0, e2d_0),
            (y2d_1_fit, y2d_1, e2d_1),
            (y2d_2_fit, y2d_2, e2d_2),
        ]

        I2d_0 = (
            np.nansum(y2d_0[2:-2, 2:-2] - np.nanmedian(y2d_0[0::8, 0::8]))
            * d1
            * d2
        )
        I2d_1 = (
            np.nansum(y2d_1[2:-2, 2:-2] - np.nanmedian(y2d_1[0::8, 0::8]))
            * d0
            * d2
        )
        I2d_2 = (
            np.nansum(y2d_2[2:-2, 2:-2] - np.nanmedian(y2d_2[0::8, 0::8]))
            * d0
            * d1
        )

        I2d = [I2d_0, I2d_1, I2d_2]

        s2d_0 = (
            np.nansum(e2d_0[2:-2, 2:-2] ** 2 + np.nanmedian(y2d_0[0::8, 0::8]))
            * d1
            * d2
        )
        s2d_1 = (
            np.nansum(e2d_1[2:-2, 2:-2] ** 2 + np.nanmedian(y2d_1[0::8, 0::8]))
            * d0
            * d2
        )
        s2d_2 = (
            np.nansum(e2d_2[2:-2, 2:-2] ** 2 + np.nanmedian(y2d_2[0::8, 0::8]))
            * d0
            * d1
        )

        s2d = [s2d_0, s2d_1, s2d_2]

        y3d = d / n
        e3d = np.sqrt(d) / n
        y3d_fit = y3d.copy()

        I3d = (
            np.nansum(
                y3d[2:-2, 2:-2, 2:-2] - np.nanmedian(y3d[0::8, 0::8, 0::8])
            )
            * d0
            * d1
            * d2
        )
        s3d = (
            np.nansum(
                e3d[2:-2, 2:-2, 2:-2] ** 2
                + np.nanmedian(y3d[0::8, 0::8, 0::8])
            )
            * d0
            * d1
            * d2
        )

        best_prof = (
            (x0[:, 0, 0], *y1d[0]),
            (x1[0, :, 0], *y1d[1]),
            (x2[0, 0, :], *y1d[2]),
        )

        best_proj = (
            (x1[0, :, :], x2[0, :, :], *y2d[0]),
            (x0[:, 0, :], x2[:, 0, :], *y2d[1]),
            (x0[:, :, 0], x1[:, :, 0], *y2d[2]),
        )

        best_fit = (x0, x1, x2, y3d_fit, y3d, e3d)

        pk_data = np.nansum(d[0::8, 0::8, 0::8])
        pk_norm = np.nansum(n[0::8, 0::8, 0::8])

        bkg_data = np.nansum(d[2:-2, 2:-2, 2:-2])
        bkg_norm = np.nansum(n[2:-2, 2:-2, 2:-2])

        b = bkg_data / bkg_norm
        b_err = np.sqrt(bkg_data) / np.nansum(bkg_norm)

        N = 125

        intens = np.nansum(bkg_data / bkg_norm - b) * d0 * d1 * d2 * N
        sig = (
            np.sqrt(np.nansum(bkg_data / bkg_norm**2 + b_err**2))
            * d0
            * d1
            * d2
            * N
        )

        params = (intens, sig, b, b_err)

        data_norm_fit = ((x0, x1, x2), (d0, d1, d2), y3d), params

        info = [d0 * d1 * d2, b, b_err]

        intens_raw = (pk_data - bkg_data / N) * d0 * d1 * d2
        sig_raw = np.sqrt(pk_data + bkg_data / N) * d0 * d1 * d2

        info += [intens_raw, sig_raw]

        info += [N, pk_data, pk_norm, bkg_data, bkg_norm]

        intensity = [I1d, I2d, I3d]
        sigma = [s1d, s2d, s3d]
        redchi2 = [[0, 0, 0], [0, 0, 0], 0]

        S = (
            np.diag(
                [
                    x0[-1, 0, 0] - x0[0, 0, 0],
                    x1[0, -1, 0] - x1[0, 0, 0],
                    x2[0, 0, -1] - x2[0, 0, 0],
                ]
            )
            ** 2
            / np.cbrt(3) ** 2
            / 4
        )

        c = (
            np.array(
                [
                    x0[-1, 0, 0] + x0[0, 0, 0],
                    x1[0, -1, 0] + x1[0, 0, 0],
                    x2[0, 0, -1] + x2[0, 0, 0],
                ]
            )
            / 2
        )

        c0, c1, c2 = c
        r0, r1, r2 = np.sqrt(np.diag(S))
        v0, v1, v2 = np.eye(3)

        sphere = c0, c1, c2, r0, r1, r2, v0, v1, v2

        return (
            c,
            S,
            sphere,
            info,
            best_prof,
            best_proj,
            best_fit,
            data_norm_fit,
            redchi2,
            intensity,
            sigma,
            intens,
            sig,
        )

    def fit_peaks(self, key_value):
        key, value = key_value

        data_info, peak_info = value

        Q0, Q1, Q2, d, n, *interp, dQ, Q, kappa, projections = data_info

        gd, gn, val_mask, det_mask = interp

        peak_file, hkl, d_spacing, wavelength, angles, goniometer = peak_info

        ellipsoid = PeakEllipsoid()

        params, value = None, None
        if all(interp):
            try:
                args = (Q0, Q1, Q2, d, n, gd, gn, val_mask, det_mask, dQ, Q)
                params = ellipsoid.fit(*args)
            except Exception as e:
                print("Exception fitting data: {}".format(e))
                return key, value
        else:
            result = self.quick_fit(Q0, Q1, Q2, d, n)

            (
                c,
                S,
                sphere,
                info,
                best_prof,
                best_proj,
                best_fit,
                data_norm_fit,
                redchi2,
                intensity,
                sigma,
                intens,
                sig,
            ) = result

            shape = self.revert_ellipsoid_parameters(sphere, projections)

        print(self.status + " 2/2 {:}/{:}".format(key, self.total))

        if params is not None:
            c, S, *best_fit = ellipsoid.best_fit

            shape = self.revert_ellipsoid_parameters(params, projections)

            norm_params = Q0, Q1, Q2, d, n, val_mask, det_mask, c, S

            try:
                intens, sig = ellipsoid.integrate(*norm_params)
            except Exception as e:
                print("Exception extracting intensity: {}".format(e))
                return key, value

            info = ellipsoid.info
            best_prof = ellipsoid.best_prof
            best_proj = ellipsoid.best_proj
            data_norm_fit = ellipsoid.data_norm_fit
            redchi2 = ellipsoid.redchi2
            intensity = ellipsoid.intensity
            sigma = ellipsoid.sigma

        if self.make_plot:
            self.peak_plot.add_ellipsoid_fit(best_fit)

            self.peak_plot.add_profile_fit(best_prof)

            self.peak_plot.add_projection_fit(best_proj)

            self.peak_plot.add_ellipsoid(c, S)

            self.peak_plot.update_envelope(c, S)

            self.peak_plot.add_peak_info(
                hkl, d_spacing, wavelength, angles, goniometer
            )

            self.peak_plot.add_peak_stats(redchi2, intensity, sigma)

            self.peak_plot.add_data_norm_fit(*data_norm_fit)

            try:
                self.peak_plot.save_plot(peak_file)
            except Exception as e:
                print("Exception saving figure: {}".format(e))
                return key, None

            value = intens, sig, shape, [*info, *shape[:3]], hkl

        return key, value

    def detection_mask(self, data, min_size=10, connectivity=3):
        """
        Identify large connected low-value regions in a 3D array.

        Parameters
        ----------
        data : ndarray
            3D input array.
        min_size : int
            Minimum number of voxels for a region to be considered significant.
        connectivity : int
            Connectivity for region labeling. Use 1, 2, or 3 for 3D.

        Returns
        -------
        mask : ndarray (bool)
            Boolean mask where True marks large low-value regions.

        """

        coverage_mask = data == 0.0

        labeled, _ = skimage.measure.label(
            coverage_mask, connectivity=connectivity, return_num=True
        )

        filtered_mask = np.zeros_like(coverage_mask, dtype=bool)
        for region in skimage.measure.regionprops(labeled):
            if region.area >= min_size:
                filtered_mask[labeled == region.label] = True

        return ~filtered_mask

    def interpolate(self, x0, x1, x2, d, n):
        detection_mask = self.detection_mask(n)

        gd1 = scipy.ndimage.gaussian_filter(d.copy(), sigma=1)
        gn1 = scipy.ndimage.gaussian_filter(n.copy(), sigma=1)

        gd2 = scipy.ndimage.gaussian_filter(d.copy(), sigma=2)
        gn2 = scipy.ndimage.gaussian_filter(n.copy(), sigma=2)

        gd = d + gd1 + gd2
        gn = n + gn1 + gn2

        data_mask = np.isfinite(gn) & (gn > 0)

        gd[~data_mask] = np.nan
        gn[~data_mask] = np.nan

        return gd, gn, data_mask, detection_mask

    def extract_peak_info(self, peaks_ws, r_cut, norm=False, fit=True):
        """
        Obtain peak information for envelope determination.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r_cut : list or float
            Cutoff radius parameter(s).

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        UB = peak.get_UB()

        peak_dict = {}

        self.total = n_peak

        for i in range(n_peak):
            print(self.status + " 1/2 {:}/{:}".format(i, self.total))

            d_spacing = peak.get_d_spacing(i)
            Q = 2 * np.pi / d_spacing

            h, k, l = peak.get_hkl(i)

            hkl = [h, k, l]

            lamda = peak.get_wavelength(i)
            kappa = 2 * np.pi / lamda

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            peak.set_peak_intensity(i, 0, 0)

            goniometer = peak.get_goniometer_angles(i)

            peak_name = peak.get_peak_name(i)

            dQ = data.get_resolution_in_Q(lamda, two_theta)

            R = peak.get_goniometer_matrix(i)

            bin_params = UB, hkl, lamda, R, two_theta, az_phi, r_cut, fit

            bin_extent = self.bin_extent(*bin_params)

            bins, extents, projections, transform = bin_extent

            if norm:
                data.normalize_to_hkl("md", transform, extents, bins)

                d, _, Q0, Q1, Q2 = data.extract_bin_info("md_data")
                n, _, Q0, Q1, Q2 = data.extract_bin_info("md_norm")

                data.clear_norm("md")

            else:
                d, _, Q0, Q1, Q2 = data.bin_in_Q(
                    "md", extents, bins, projections
                )

                n = 1.0 * (data.extract_counts("md_bin") > 0)

            if fit:
                interp = self.interpolate(Q0, Q1, Q2, d, n)
            else:
                interp = [None] * 4

            data_info = (Q0, Q1, Q2, d, n, *interp, dQ, Q, kappa, projections)

            peak_file = self.get_plot_file(peak_name)

            directory = os.path.dirname(peak_file)

            os.makedirs(directory, exist_ok=True)

            peak_info = (peak_file, hkl, d_spacing, lamda, angles, goniometer)

            peak_dict[i] = data_info, peak_info

        return peak_dict

    def update_peak_offsets(self, peaks_ws, offsets, peak_dict):
        peak = PeakModel(peaks_ws)

        c0, c1, c2, Q = offsets

        for i, value in peak_dict.items():
            if value is not None:
                data_info, peak_info = peak_dict[i]

                projections = data_info[-1]

                W = np.column_stack(projections)

                vec = [c0[i] + Q[i], c1[i], c2[i]]

                if np.isfinite(vec).all():
                    Q0, Q1, Q2 = np.dot(W, vec)

                    peak.set_peak_center(i, Q0, Q1, Q2)

    def update_peak_info(self, peaks_ws, peak_dict):
        peak = PeakModel(peaks_ws)

        for i, value in peak_dict.items():
            if value is not None:
                I, sigma, shape, info, hkl = value

                peak.set_peak_intensity(i, I, sigma)

                peak.set_peak_shape(i, *shape)

                peak.add_diagonstic_info(i, info)

    def bin_axes(self, R, two_theta, az_phi):
        two_theta = np.deg2rad(two_theta)
        az_phi = np.deg2rad(az_phi)

        kf_hat = np.array(
            [
                np.sin(two_theta) * np.cos(az_phi),
                np.sin(two_theta) * np.sin(az_phi),
                np.cos(two_theta),
            ]
        )

        ki_hat = np.array([0, 0, 1])

        n = kf_hat - ki_hat
        n /= np.linalg.norm(n)

        v = np.cross(ki_hat, kf_hat)
        v /= np.linalg.norm(v)

        u = np.cross(v, n)
        u /= np.linalg.norm(u)

        return R.T @ n, R.T @ u, R.T @ v

    def project_ellipsoid_parameters(self, params, projections):
        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W.T, [c0, c1, c2]), r0, r1, r2, *np.dot(W.T, V).T

    def revert_ellipsoid_parameters(self, params, projections):
        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W, [c0, c1, c2]), r0, r1, r2, *np.dot(W, V).T

    def trasform_Q(self, Q0, Q1, Q2, projections):
        W = np.column_stack(projections)

        return np.einsum("ij,j...->i...", W, [Q0, Q1, Q2])

    def bin_extent(self, UB, hkl, lamda, R, two_theta, az_phi, r_cut, fit):
        n, u, v = self.bin_axes(R, two_theta, az_phi)

        projections = [n, u, v]

        W = np.column_stack(projections)
        Wp = np.linalg.inv(W.T @ (2 * np.pi * UB)).T

        transform = Wp.tolist()

        h, k, l = hkl

        Q0, Q1, Q2 = 2 * np.pi * np.dot(W.T, np.dot(UB, [h, k, l]))

        n_bins = 15 if fit else 9

        if type(r_cut) is float or fit:
            dQ_cut = 3 * [r_cut]
        else:
            (r0, r1, r2), (dr0, dr1, dr2) = r_cut
            kappa = 2 * np.pi / lamda
            Q = 2 * kappa * np.sin(0.5 * np.deg2rad(two_theta))
            dQ_cut = [
                2 * r0 * (1 + dr0 * kappa),
                2 * r1 * (1 + dr1 * kappa),
                2 * r2 * (1 + dr2 * Q),
            ]

        bin_sizes = np.array(dQ_cut) / n_bins

        Q0_box, Q1_box, Q2_box = [], [], []

        points = [
            [h - 0.5, k, l],
            [h + 0.5, k, l],
            [h, k - 0.5, l],
            [h, k + 0.5, l],
            [h, k, l - 0.5],
            [h, k, l + 0.5],
        ]

        for point in points:
            Qp_0, Qp_1, Qp_2 = 2 * np.pi * np.dot(W.T, np.dot(UB, point))
            Q0_box.append(Qp_0 - Q0)
            Q1_box.append(Qp_1 - Q1)
            Q2_box.append(Qp_2 - Q2)

        dQ0_cut, dQ1_cut, dQ2_cut = dQ_cut

        dQ0 = np.min([np.max(np.abs(Q0_box)), dQ0_cut])
        dQ1 = np.min([np.max(np.abs(Q1_box)), dQ1_cut])
        dQ2 = np.min([np.max(np.abs(Q2_box)), dQ2_cut])

        extents = np.array(
            [[Q0 - dQ0, Q0 + dQ0], [Q1 - dQ1, Q1 + dQ1], [Q2 - dQ2, Q2 + dQ2]]
        )

        if fit:
            min_adjusted = np.floor(extents[:, 0] / bin_sizes) * bin_sizes
            max_adjusted = np.ceil(extents[:, 1] / bin_sizes) * bin_sizes

            bins = ((max_adjusted - min_adjusted) / bin_sizes).astype(int)

            bins[bins < 10] = 10
            bins[bins > 30] = 30

            extents = np.vstack((min_adjusted, max_adjusted)).T

        else:
            bins = np.array([n_bins, n_bins, n_bins])

        return bins, extents, projections, transform


class PeakCentroid:
    def __init__(self):
        pass

    def sigma_clip(self, array, sigma=3, maxiters=3):
        array = np.array(array, dtype=float)
        y = array[np.isfinite(array)]
        mask = np.ones_like(y, dtype=bool)

        for _ in range(maxiters):
            data = y[mask]

            med = np.median(data)
            mad = scipy.stats.median_abs_deviation(data, scale="normal")

            if mad <= 0:
                break

            dev = np.abs(y - med)
            new_mask = dev < (sigma * mad)

            if np.all(new_mask == mask):
                break

            mask = new_mask.copy()

        data = y[mask]
        med = np.median(data)
        mad = scipy.stats.median_abs_deviation(data, scale="normal")

        return med, mad

    def model(self, y, e, x0, x1, x2):
        # B, B_err = self.sigma_clip(y, maxiters=3)
        B = np.nanpercentile(y, 15)

        w = y - B
        w[w < 0] = 0
        w *= w

        wgt = np.nansum(w)

        if wgt > 0:
            c0 = np.nansum(x0 * w) / wgt
            c1 = np.nansum(x1 * w) / wgt
            c2 = np.nansum(x2 * w) / wgt

        else:
            c0 = (x0[1, 0, 0] + x0[0, 0, 0]) / 2
            c1 = (x1[0, 1, 0] + x1[0, 0, 0]) / 2
            c2 = (x2[0, 0, 1] + x2[0, 0, 0]) / 2

        return c0, c1, c2

    def extract_info(self, peak_dict):
        ys, es = [], []
        x0s, x1s, x2s = [], [], []

        Qs, ks = [], []

        for key in peak_dict.keys():
            data_info = peak_dict[key][0]
            Q0, Q1, Q2, _, _, *interp, dQ, Q, k, projections = data_info
            d, n, val_mask, det_mask = interp

            x0 = Q0 - Q
            x1 = Q1
            x2 = Q2

            y = d.copy()  # / n
            e = np.sqrt(d)  # / n

            ys.append(y)
            es.append(e)

            x0s.append(x0)
            x1s.append(x1)
            x2s.append(x2)

            Qs.append(Q)
            ks.append(k)

        return ys, es, x0s, x1s, x2s, Qs, ks

    def fit(self, peak_dict):
        args = self.extract_info(peak_dict)

        c0s, c1s, c2s, Qs = [], [], [], []
        for y, e, x0, x1, x2, Q, k in zip(*args):
            c0, c1, c2 = self.model(y, e, x0, x1, x2)
            c0s.append(c0)
            c1s.append(c1)
            c2s.append(c2)
            Qs.append(Q)

        return np.array(c0s), np.array(c1s), np.array(c2s), np.array(Qs)


class PeakProfile:
    def __init__(self, r_cut):
        self.params = Parameters()

        self.params.add("r0", value=r_cut / 2, min=0.001, max=2 * r_cut)
        self.params.add("r1", value=r_cut / 2, min=0.001, max=2 * r_cut)
        self.params.add("r2", value=r_cut / 2, min=0.001, max=2 * r_cut)

        self.params.add("dr0", value=0, min=0, max=r_cut * 10, vary=True)
        self.params.add("dr1", value=0, min=0, max=r_cut * 10, vary=True)
        self.params.add("dr2", value=0, min=0, max=r_cut * 10, vary=True)

    def calculate_fit(self, y, z, e):
        y_hat = np.exp(-0.5 * z**2)

        y_bar = np.nanmean(y)
        y_hat_bar = np.nanmean(y_hat)

        w = 1 / (e**2 + (0.05 * y) ** 2 + 1e-16)
        w[np.isinf(w)] = np.nan
        y[np.isinf(y)] = np.nan

        sum_w = np.nansum(w)
        num = np.nansum(w * (y_hat - y_hat_bar) * (y - y_bar))
        den = np.nansum(w * (y_hat - y_hat_bar) ** 2)

        A = num / den if den > 0 else 0
        B = y_bar - A * y_hat_bar
        M = 2

        if A <= 0 or not np.isfinite(A):
            if sum_w > 0:
                A, B, M = 0, np.nansum(w * y) / sum_w, 1
            else:
                A, B, M = 0, 0, 0

        y_fit = A * y_hat + B

        residuals = y_fit - (A * y_hat + B)

        N = np.nansum(np.isfinite(y))

        A_err, B_err = 0, 0

        if N >= 3 and den > 0 and A > 0:
            residual_var = np.nansum(w * residuals**2) / (N - M)

            var_A = residual_var / den
            var_B = residual_var * (1 / sum_w + (y_hat_bar**2 / den))

            A_err = np.sqrt(var_A)
            B_err = np.sqrt(var_B)

            if A == 0:
                if sum_w > 0:
                    B_err = np.sqrt(residual_var / sum_w)
                else:
                    B_err = 0
                A_err = 0

        return A, B, A_err, B_err, y_fit

    def model(self, r, delta, y, e, x, kappa):
        scale = np.sqrt(scipy.stats.chi2.ppf(0.997, df=1))

        sigma = r / scale * (1 + delta * kappa)

        z = x / sigma

        A, B, A_err, B_err, y_fit = self.calculate_fit(y, z, e)

        xc = 0
        if A > 0:
            w = y - B
            mask = np.abs(z) < scale
            sum_w = np.nansum(w[mask])
            if sum_w > 0 and sum_w < np.inf:
                sum_wx = np.nansum(w[mask] * x[mask])
                xc = sum_wx / sum_w
                z = (x - xc) / sigma
                A, B, A_err, B_err, y_fit = self.calculate_fit(y, z, e)

        return y_fit, x - xc, A, B, A_err, B_err

    def objective(self, params, *args):
        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value

        dr0 = params["dr0"].value
        dr1 = params["dr1"].value
        dr2 = params["dr2"].value

        y0s, y1s, y2s, e0s, e1s, e2s, x0s, x1s, x2s, Qs, ks = args

        res = []
        for y0, e0, x0, k in zip(y0s, e0s, x0s, ks):
            y0_fit, c0, A, B, A_err, B_err = self.model(r0, dr0, y0, e0, x0, k)
            res.append(((y0_fit - y0) / e0).flatten())
        for y1, e1, x1, Q in zip(y1s, e1s, x1s, Qs):
            y1_fit, c1, A, B, A_err, B_err = self.model(r1, dr1, y1, e1, x1, k)
            res.append(((y1_fit - y1) / e1).flatten())
        for y2, e2, x2, Q in zip(y2s, e2s, x2s, Qs):
            y2_fit, c2, A, B, A_err, B_err = self.model(r2, dr2, y2, e2, x2, Q)
            res.append(((y2_fit - y2) / e2).flatten())

        return np.concatenate(res)

    def extract_fit(self):
        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        dr0 = self.params["dr0"].value
        dr1 = self.params["dr1"].value
        dr2 = self.params["dr2"].value

        y0c, e0c, x0c = [], [], []
        y1c, e1c, x1c = [], [], []
        y2c, e2c, x2c = [], [], []

        Qs, ks = [], []

        for y0, y1, y2, e0, e1, e2, x0, x1, x2, Q, k in zip(*self.args):
            y0_fit, c0, A0, B0, A0_err, B0_err = self.model(
                r0, dr0, y0, e0, x0, k
            )
            y1_fit, c1, A1, B1, A1_err, B1_err = self.model(
                r1, dr1, y1, e1, x1, k
            )
            y2_fit, c2, A2, B2, A2_err, B2_err = self.model(
                r2, dr2, y2, e2, x2, Q
            )
            if (
                A0 > B0
                and B0 > 0
                and A1 > B1
                and B1 > 0
                and A2 > B2
                and B2 > 0
            ):
                y0_hat = y0 - B0
                e0_hat = np.sqrt(e0**2 + B0_err**2)
                e0_hat = np.abs(y0_hat / A0) * np.sqrt(
                    (A0_err / A0) ** 2 + (e0_hat / y0_hat) ** 2
                )
                y0_hat = y0_hat / A0
                x0c += c0.tolist()
                y0c += y0_hat.tolist()
                e0c += e0_hat.tolist()

                y1_hat = y1 - B1
                e1_hat = np.sqrt(e1**2 + B1_err**2)
                e1_hat = np.abs(y1_hat / A1) * np.sqrt(
                    (A1_err / A1) ** 2 + (e1_hat / y1_hat) ** 2
                )
                y1_hat = y1_hat / A1

                x1c += c1.tolist()
                y1c += y1_hat.tolist()
                e1c += e1_hat.tolist()

                y2_hat = y2 - B2
                e2_hat = np.sqrt(e2**2 + B2_err**2)
                e2_hat = np.abs(y2_hat / A2) * np.sqrt(
                    (A2_err / A2) ** 2 + (e2_hat / y2_hat) ** 2
                )
                y2_hat = y2_hat / A2

                x2c += c2.tolist()
                y2c += y2_hat.tolist()
                e2c += e2_hat.tolist()

                Qs.append(Q)
                ks.append(k)

        x0c = np.array(x0c)
        x1c = np.array(x1c)
        x2c = np.array(x2c)

        y0c = np.array(y0c)
        y1c = np.array(y1c)
        y2c = np.array(y2c)

        e0c = np.array(e0c)
        e1c = np.array(e1c)
        e2c = np.array(e2c)

        Qs = np.array(Qs)
        ks = np.array(ks)

        return (x0c, x1c, x2c), (y0c, y1c, y2c), (e0c, e1c, e2c), Qs, ks

    def extract_info(self, peak_dict):
        y0s, e0s, x0s = [], [], []
        y1s, e1s, x1s = [], [], []
        y2s, e2s, x2s = [], [], []

        Qs, ks = [], []

        for key in peak_dict.keys():
            data_info = peak_dict[key][0]
            Q0, Q1, Q2, _, _, *interp, dQ, Q, k, projections = data_info
            d, n, val_mask, det_mask = interp

            d0 = np.nansum(d, axis=(1, 2))
            n0 = np.nansum(n, axis=(1, 2))
            y0 = d0 / n0
            e0 = np.sqrt(d0) / n0
            x0 = Q0[:, 0, 0] - Q

            y0s.append(y0)
            e0s.append(e0)
            x0s.append(x0)

            d1 = np.nansum(d, axis=(0, 2))
            n1 = np.nansum(n, axis=(0, 2))
            y1 = d1 / n1
            e1 = np.sqrt(d1) / n1
            x1 = Q1[0, :, 0]

            y1s.append(y1)
            e1s.append(e1)
            x1s.append(x1)

            d2 = np.nansum(d, axis=(0, 1))
            n2 = np.nansum(n, axis=(0, 1))
            y2 = d2 / n2
            e2 = np.sqrt(d2) / n2
            x2 = Q2[0, 0, :]

            y2s.append(y2)
            e2s.append(e2)
            x2s.append(x2)

            Qs.append(Q)
            ks.append(k)

        return y0s, y1s, y2s, e0s, e1s, e2s, x0s, x1s, x2s, Qs, ks

    def fit(self, peak_dict):
        self.args = self.extract_info(peak_dict)

        out = Minimizer(
            self.objective, self.params, fcn_args=self.args, nan_policy="omit"
        )

        result = out.minimize(method="least_squares", loss="soft_l1")

        self.params = result.params

        r0 = result.params["r0"].value
        r1 = result.params["r1"].value
        r2 = result.params["r2"].value

        dr0 = result.params["dr0"].value
        dr1 = result.params["dr1"].value
        dr2 = result.params["dr2"].value

        return (r0, r1, r2), (dr0, dr1, dr2)


class PeakSphere:
    def __init__(self, r_cut):
        self.params = Parameters()

        if np.isclose(r_cut, 0.04) or r_cut < 0.04:
            r_cut = 0.2

        self.params.add("sigma", value=r_cut / 6, min=0.01, max=r_cut / 4)

    def model(self, x, A, sigma):
        z = x / sigma

        return A * (
            scipy.special.erf(z / np.sqrt(2))
            - np.sqrt(2 / np.pi) * z * np.exp(-0.5 * z**2)
        )

    def residual(self, params, x, y):
        A = params["A"]
        sigma = params["sigma"]

        y_fit = self.model(x, A, sigma)

        diff = y_fit - y
        diff[~np.isfinite(diff)] = 1e9

        return diff

    def fit(self, x, y):
        scale = np.sqrt(scipy.stats.chi2.ppf(0.997, df=1))
        y_max = np.max(y)

        y[y < 0] = 0

        if np.isclose(y_max, 0):
            y_max = np.inf

        self.params.add("A", value=y_max, min=0, max=100 * y_max, vary=True)

        out = Minimizer(
            self.residual, self.params, fcn_args=(x, y), nan_policy="omit"
        )

        result = out.minimize(method="least_squares", loss="soft_l1")

        self.params = result.params

        return scale * result.params["sigma"].value

    def best_fit(self, r):
        A = self.params["A"].value
        sigma = self.params["sigma"].value

        return self.model(r, A, sigma), A, sigma


class PeakEllipsoid:
    def __init__(self):
        self.params = Parameters()

    def update_constraints(self, x0, x1, x2, dx):
        dx0 = x0[:, 0, 0][1] - x0[:, 0, 0][0]
        dx1 = x1[0, :, 0][1] - x1[0, :, 0][0]
        dx2 = x2[0, 0, :][1] - x2[0, 0, :][0]

        r0_max = (x0[:, 0, 0][-1] - x0[:, 0, 0][0]) / 2
        r1_max = (x1[0, :, 0][-1] - x1[0, :, 0][0]) / 2
        r2_max = (x2[0, 0, :][-1] - x2[0, 0, :][0]) / 2

        c0 = (x0[:, 0, 0][-1] + x0[:, 0, 0][0]) / 2
        c1 = (x1[0, :, 0][-1] + x1[0, :, 0][0]) / 2
        c2 = (x2[0, 0, :][-1] + x2[0, 0, :][0]) / 2

        c0_min, c1_min, c2_min = (
            c0 - r0_max,
            c1 - r1_max,
            c2 - r2_max,
        )
        c0_max, c1_max, c2_max = (
            c0 + r0_max,
            c1 + r1_max,
            c2 + r2_max,
        )

        r0 = 4 * dx0
        r1 = 4 * dx1
        r2 = 4 * dx2

        self.params.add("c0", value=c0, min=c0_min, max=c0_max)
        self.params.add("c1", value=c1, min=c1_min, max=c1_max)
        self.params.add("c2", value=c2, min=c2_min, max=c2_max)

        self.params.add("r0", value=r0, min=dx0, max=r0_max)
        self.params.add("r1", value=r1, min=dx1, max=r1_max)
        self.params.add("r2", value=r2, min=dx2, max=r2_max)

        self.params.add("u0", value=0.0, min=-np.pi / 2, max=np.pi / 2)
        self.params.add("u1", value=0.0, min=-np.pi / 2, max=np.pi / 2)
        self.params.add("u2", value=0.0, min=-np.pi / 2, max=np.pi / 2)

    def S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):
        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def det_S(self, r0, r1, r2, u0, u1, u2):
        S = self.S_matrix(r0, r1, r2, u0, u1, u2)
        return np.linalg.det(S)

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):
        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S

    def data_norm(self, d, n, rel_err=0.05, abs_err=1):
        mask = (n > 0) & np.isfinite(n)

        d[~mask] = np.nan
        n[~mask] = np.nan

        # c_bkg = np.nanmean(c)

        y_int = d / n
        e_int = np.sqrt(d + (rel_err * d) ** 2 + abs_err**2) / n

        return y_int, e_int

    def normalize(self, x0, x1, x2, d, n, mode="3d"):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == "1d_0":
            d_int = np.nansum(d, axis=(1, 2))
            n_int = np.nanmean(n / dx1 / dx2, axis=(1, 2))
        elif mode == "1d_1":
            d_int = np.nansum(d, axis=(0, 2))
            n_int = np.nanmean(n / dx0 / dx2, axis=(0, 2))
        elif mode == "1d_2":
            d_int = np.nansum(d, axis=(0, 1))
            n_int = np.nanmean(n / dx0 / dx1, axis=(0, 1))
        elif mode == "2d_0":
            d_int = np.nansum(d, axis=0)
            n_int = np.nanmean(n / dx0, axis=0)
        elif mode == "2d_1":
            d_int = np.nansum(d, axis=1)
            n_int = np.nanmean(n / dx1, axis=1)
        elif mode == "2d_2":
            d_int = np.nansum(d, axis=2)
            n_int = np.nanmean(n / dx2, axis=2)
        elif mode == "3d":
            d_int = d.copy()
            n_int = n.copy()

        y_int, e_int = self.data_norm(d_int, n_int)

        return y_int, e_int

    def filter_array(self, data, size=3):
        array = np.array(data)

        array[~np.isfinite(array)] = 0

        result = scipy.ndimage.gaussian_filter(
            array, size=size, mode="constant", cval=0
        )

        return result

    def ellipsoid_covariance(self, inv_S, mode="3d", perc=99.7):
        if mode == "3d":
            scale = scipy.stats.chi2.ppf(perc / 100, df=3)
            inv_var = inv_S * scale
        elif mode == "2d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[1:, 1:] * scale
        elif mode == "2d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[0::2, 0::2] * scale
        elif mode == "2d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[:2, :2] * scale
        elif mode == "1d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[0, 0] * scale
        elif mode == "1d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[1, 1] * scale
        elif mode == "1d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[2, 2] * scale

        return inv_var

    def chi_2_fit(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
            m = 11
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
            m = 9
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
            m = 9
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
            m = 9
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
            m = 5
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
            m = 5
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2
            m = 5
            k = 1

        mask = (d2 <= 2 ** (2 / k)) & np.isfinite(y) & (e > 0)

        n = np.sum(mask)

        dof = n - m

        if dof <= 0:
            return np.inf
        else:
            return np.nansum(((y_fit[mask] - y[mask]) / e[mask]) ** 2) / dof

    def estimate_intensity(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == "3d":
            dx = np.prod([dx0, dx1, dx2])
            k = 3
        elif mode == "2d_0":
            dx = np.prod([dx1, dx2])
            k = 2
        elif mode == "2d_1":
            dx = np.prod([dx0, dx2])
            k = 2
        elif mode == "2d_2":
            dx = np.prod([dx0, dx1])
            k = 2
        elif mode == "1d_0":
            dx = dx0
            k = 1
        elif mode == "1d_1":
            dx = dx1
            k = 1
        elif mode == "1d_2":
            dx = dx2
            k = 1

        pk = (d2 <= 1**2) & np.isfinite(y) & (e > 0)
        bkg = (d2 > 1**2) & (d2 < 2 ** (2 / k)) & (e > 0)

        # b = np.nansum(y[bkg]/e[bkg]**2)/np.nansum(1/e[bkg]**2)
        # b_err = 1/np.sqrt(np.nansum(1/e[bkg]**2))

        b = np.nanmean(y[bkg])
        b_err = np.sqrt(np.nanmean(e[bkg] ** 2))

        I = np.nansum(y[pk] - b) * dx
        sig = np.sqrt(np.nansum(e[pk] ** 2 + b_err**2)) * dx

        if np.isclose(sig, 0) or I < sig:
            sig = float("inf")

        return I, sig

    def gaussian(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2

        return np.exp(-0.5 * d2)

    def lorentzian(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode) / (2 * np.log(2))

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            k = 1

        return 1 / (1 + d2) ** (0.5 * (1 + k))

    def inv_S_deriv_r(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2 / r0**3, 0, 0]) @ U.T
        dinv_S1 = U @ np.diag([0, -2 / r1**3, 0]) @ U.T
        dinv_S2 = U @ np.diag([0, 0, -2 / r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_u(self, r0, r1, r2, u0, u1, u2):
        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        U = self.U_matrix(u0, u1, u2)
        dU0, dU1, dU2 = self.U_deriv_u(u0, u1, u2)

        dinv_S0 = dU0 @ V @ U.T + U @ V @ dU0.T
        dinv_S1 = dU1 @ V @ U.T + U @ V @ dU1.T
        dinv_S2 = dU2 @ V @ U.T + U @ V @ dU2.T

        return dinv_S0, dinv_S1, dinv_S2

    def U_deriv_u(self, u0, u1, u2, delta=1e-6):
        dU0 = self.U_matrix(u0 + delta, u1, u2) - self.U_matrix(
            u0 - delta, u1, u2
        )
        dU1 = self.U_matrix(u0, u1 + delta, u2) - self.U_matrix(
            u0, u1 - delta, u2
        )
        dU2 = self.U_matrix(u0, u1, u2 + delta) - self.U_matrix(
            u0, u1, u2 - delta
        )

        return 0.5 * dU0 / delta, 0.5 * dU1 / delta, 0.5 * dU2 / delta

    def gaussian_integral(self, inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        return np.sqrt((2 * np.pi) ** k * det)

    def lorentzian_integral(self, inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode) / (2 * np.log(2))

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        return (
            scipy.special.gamma(0.5)
            * np.sqrt(np.pi * det)
            / scipy.special.gamma(0.5 * (1 + k))
        )

    def gaussian_integral_jac_S(self, inv_S, d_inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        g = np.sqrt((2 * np.pi) ** k * det)

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
        elif mode == "2d_0":
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g0 = g1 * 0
        elif mode == "2d_1":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g1 = g2 * 0
        elif mode == "2d_2":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = g0 * 0
        elif mode == "1d_0":
            g0 = d_inv_var[0] * inv_var
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            g1 = d_inv_var[1] * inv_var
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            g2 = d_inv_var[2] * inv_var
            g0 = g1 = g2 * 0

        return 0.5 * g * np.array([g0, g1, g2])

    def lorentzian_integral_jac_S(self, inv_S, d_inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        l = (
            scipy.special.gamma(0.5)
            * np.sqrt(np.pi * det)
            / scipy.special.gamma(0.5 * (1 + k))
        )

        inv_var = self.ellipsoid_covariance(inv_S, mode) / (2 * np.log(2))
        d_inv_var = [
            self.ellipsoid_covariance(val, mode) / (2 * np.log(2))
            for val in d_inv_S
        ]

        if mode == "3d":
            l0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            l1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            l2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
        elif mode == "2d_0":
            l1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            l2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            l0 = l1 * 0
        elif mode == "2d_1":
            l0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            l2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            l1 = l2 * 0
        elif mode == "2d_2":
            l0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            l1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            l2 = l0 * 0
        elif mode == "1d_0":
            l0 = d_inv_var[0] * inv_var
            l1 = l2 = l0 * 0
        elif mode == "1d_1":
            l1 = d_inv_var[1] * inv_var
            l2 = l0 = l1 * 0
        elif mode == "1d_2":
            l2 = d_inv_var[2] * inv_var
            l0 = l1 = l2 * 0

        return 0.5 * l * np.array([l0, l1, l2])

    def gaussian_jac_c(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g0 = g1 * 0
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g1 = g2 * 0
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1 = np.einsum("ij,j...->i...", inv_var, dx)
            g2 = g0 * 0
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = inv_var * dx
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = inv_var * dx
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = inv_var * dx
            g0 = g1 = g2 * 0

        g = np.exp(-0.5 * d2)

        return g * np.array([g0, g1, g2])

    def lorentzian_jac_c(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode) / (2 * np.log(2))

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0, l1, l2 = np.einsum("ij,j...->i...", inv_var, dx)
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l1, l2 = np.einsum("ij,j...->i...", inv_var, dx)
            l0 = l1 * 0
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0, l2 = np.einsum("ij,j...->i...", inv_var, dx)
            l1 = l2 * 0
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0, l1 = np.einsum("ij,j...->i...", inv_var, dx)
            l2 = l0 * 0
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            l0 = inv_var * dx
            l1 = l2 = l0 * 0
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            l1 = inv_var * dx
            l2 = l0 = l1 * 0
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            l2 = inv_var * dx
            l0 = l1 = l2 * 0
            k = 1

        lp = (k + 1) / (1 + d2) ** (0.5 * (3 + k))

        return lp * np.array([l0, l1, l2])

    def gaussian_jac_S(self, x0, x1, x2, c, inv_S, d_inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g0 = g1 * 0
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g1 = g2 * 0
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = g0 * 0
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = d_inv_var[0] * dx**2
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = d_inv_var[1] * dx**2
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = d_inv_var[2] * dx**2
            g0 = g1 = g2 * 0

        g = np.exp(-0.5 * d2)

        return -0.5 * g * np.array([g0, g1, g2])

    def lorentzian_jac_S(self, x0, x1, x2, c, inv_S, d_inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode) / (2 * np.log(2))
        d_inv_var = [
            self.ellipsoid_covariance(val, mode) / (2 * np.log(2))
            for val in d_inv_S
        ]

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            l1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            l2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            l2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            l0 = l1 * 0
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            l2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            l1 = l2 * 0
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            l0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            l1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            l2 = l0 * 0
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            l0 = d_inv_var[0] * dx**2
            l1 = l2 = l0 * 0
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            l1 = d_inv_var[1] * dx**2
            l2 = l0 = l1 * 0
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            l2 = d_inv_var[2] * dx**2
            l0 = l1 = l2 * 0
            k = 1

        lp = (k + 1) / (1 + d2) ** (0.5 * (3 + k))

        return -0.5 * lp * np.array([l0, l1, l2])

    def residual_1d(self, params, x0, x1, x2, ys, es):
        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A0 = params["A1d_0"]
        A1 = params["A1d_1"]
        A2 = params["A1d_2"]

        B0 = params["B1d_0"]
        B1 = params["B1d_1"]
        B2 = params["B1d_2"]

        C0 = params["C1d_0"]
        C1 = params["C1d_1"]
        C2 = params["C1d_2"]

        H0 = params["H1d_0"]
        H1 = params["H1d_1"]
        H2 = params["H1d_2"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "1d_0")
        y1_gauss = self.gaussian(*args, "1d_1")
        y2_gauss = self.gaussian(*args, "1d_2")

        y0_lorentz = self.lorentzian(*args, "1d_0")
        y1_lorentz = self.lorentzian(*args, "1d_1")
        y2_lorentz = self.lorentzian(*args, "1d_2")

        diff = []

        y0_fit = A0 * y0_gauss + H0 * y0_lorentz + B0 + C0 * (x0[:, 0, 0] - c0)
        y1_fit = A1 * y1_gauss + H1 * y1_lorentz + B1 + C1 * (x1[0, :, 0] - c1)
        y2_fit = A2 * y2_gauss + H2 * y2_lorentz + B2 + C2 * (x2[0, 0, :] - c2)

        res = (y0_fit - y0) / e0

        diff += res.flatten().tolist()

        res = (y1_fit - y1) / e1

        diff += res.flatten().tolist()

        res = (y2_fit - y2) / e2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_1d(self, params, x0, x1, x2, ys, es):
        params_list = [name for name, par in params.items()]

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A0 = params["A1d_0"]
        A1 = params["A1d_1"]
        A2 = params["A1d_2"]

        # B0 = params['B1d_0']
        # B1 = params['B1d_1']
        # B2 = params['B1d_2']

        C0 = params["C1d_0"]
        C1 = params["C1d_1"]
        C2 = params["C1d_2"]

        H0 = params["H1d_0"]
        H1 = params["H1d_1"]
        H2 = params["H1d_2"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "1d_0")
        y1_gauss = self.gaussian(*args, "1d_1")
        y2_gauss = self.gaussian(*args, "1d_2")

        y0_lorentz = self.lorentzian(*args, "1d_0")
        y1_lorentz = self.lorentzian(*args, "1d_1")
        y2_lorentz = self.lorentzian(*args, "1d_2")

        dA0 = y0_gauss / e0
        dA1 = y1_gauss / e1
        dA2 = y2_gauss / e2

        dH0 = y0_lorentz / e0
        dH1 = y1_lorentz / e1
        dH2 = y2_lorentz / e2

        dB0 = 1 / e0
        dB1 = 1 / e1
        dB2 = 1 / e2

        dC0 = (x0[:, 0, 0] - c0) / e0
        dC1 = (x1[0, :, 0] - c1) / e1
        dC2 = (x2[0, 0, :] - c2) / e2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_0")
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_1")
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_2")

        yc0_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="1d_0")
        yc1_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="1d_1")
        yc2_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="1d_2")

        dc0_0, dc1_0, dc2_0 = (A0 * yc0_gauss + H0 * yc0_lorentz) / e0
        dc0_1, dc1_1, dc2_1 = (A1 * yc1_gauss + H1 * yc1_lorentz) / e1
        dc0_2, dc1_2, dc2_2 = (A2 * yc2_gauss + H2 * yc2_lorentz) / e2

        dc0_0 -= C0 / e0
        dc1_1 -= C1 / e1
        dc2_2 -= C2 / e2

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_0")
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_1")
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_2")

        yr0_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="1d_0"
        )
        yr1_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="1d_1"
        )
        yr2_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="1d_2"
        )

        dr0_0, dr1_0, dr2_0 = (A0 * yr0_gauss + H0 * yr0_lorentz) / e0
        dr0_1, dr1_1, dr2_1 = (A1 * yr1_gauss + H1 * yr1_lorentz) / e1
        dr0_2, dr1_2, dr2_2 = (A2 * yr2_gauss + H2 * yr2_lorentz) / e2

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_0")
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_1")
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_2")

        yu0_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="1d_0"
        )
        yu1_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="1d_1"
        )
        yu2_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="1d_2"
        )

        du0_0, du1_0, du2_0 = (A0 * yu0_gauss + H0 * yu0_lorentz) / e0
        du0_1, du1_1, du2_1 = (A1 * yu1_gauss + H1 * yu1_lorentz) / e1
        du0_2, du1_2, du2_2 = (A2 * yu2_gauss + H2 * yu2_lorentz) / e2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)

        n01 = n0 + n1
        n012 = n01 + n2

        jac = np.zeros((n_params, n012))

        jac[params_list.index("A1d_0"), :n0] = dA0.flatten()
        jac[params_list.index("H1d_0"), :n0] = dH0.flatten()
        jac[params_list.index("B1d_0"), :n0] = dB0.flatten()
        jac[params_list.index("C1d_0"), :n0] = dC0.flatten()
        jac[params_list.index("c0"), :n0] = dc0_0.flatten()
        jac[params_list.index("c1"), :n0] = dc1_0.flatten()
        jac[params_list.index("c2"), :n0] = dc2_0.flatten()
        jac[params_list.index("r0"), :n0] = dr0_0.flatten()
        jac[params_list.index("r1"), :n0] = dr1_0.flatten()
        jac[params_list.index("r2"), :n0] = dr2_0.flatten()
        jac[params_list.index("u0"), :n0] = du0_0.flatten()
        jac[params_list.index("u1"), :n0] = du1_0.flatten()
        jac[params_list.index("u2"), :n0] = du2_0.flatten()

        jac[params_list.index("A1d_1"), n0:n01] = dA1.flatten()
        jac[params_list.index("H1d_1"), n0:n01] = dH1.flatten()
        jac[params_list.index("B1d_1"), n0:n01] = dB1.flatten()
        jac[params_list.index("C1d_1"), n0:n01] = dC1.flatten()
        jac[params_list.index("c0"), n0:n01] = dc0_1.flatten()
        jac[params_list.index("c1"), n0:n01] = dc1_1.flatten()
        jac[params_list.index("c2"), n0:n01] = dc2_1.flatten()
        jac[params_list.index("r0"), n0:n01] = dr0_1.flatten()
        jac[params_list.index("r1"), n0:n01] = dr1_1.flatten()
        jac[params_list.index("r2"), n0:n01] = dr2_1.flatten()
        jac[params_list.index("u0"), n0:n01] = du0_1.flatten()
        jac[params_list.index("u1"), n0:n01] = du1_1.flatten()
        jac[params_list.index("u2"), n0:n01] = du2_1.flatten()

        jac[params_list.index("A1d_2"), n01:n012] = dA2.flatten()
        jac[params_list.index("H1d_2"), n01:n012] = dH2.flatten()
        jac[params_list.index("B1d_2"), n01:n012] = dB2.flatten()
        jac[params_list.index("C1d_2"), n01:n012] = dC2.flatten()
        jac[params_list.index("c0"), n01:n012] = dc0_2.flatten()
        jac[params_list.index("c1"), n01:n012] = dc1_2.flatten()
        jac[params_list.index("c2"), n01:n012] = dc2_2.flatten()
        jac[params_list.index("r0"), n01:n012] = dr0_2.flatten()
        jac[params_list.index("r1"), n01:n012] = dr1_2.flatten()
        jac[params_list.index("r2"), n01:n012] = dr2_2.flatten()
        jac[params_list.index("u0"), n01:n012] = du0_2.flatten()
        jac[params_list.index("u1"), n01:n012] = du1_2.flatten()
        jac[params_list.index("u2"), n01:n012] = du2_2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        diff = np.concatenate([1 / e.flatten() for e in es])

        mask = np.isfinite(diff)

        return jac[ind][:, mask]

    def residual_2d(self, params, x0, x1, x2, ys, es):
        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A0 = params["A2d_0"]
        A1 = params["A2d_1"]
        A2 = params["A2d_2"]

        B0 = params["B2d_0"]
        B1 = params["B2d_1"]
        B2 = params["B2d_2"]

        C01 = params["C2d_01"]
        C02 = params["C2d_02"]

        C10 = params["C2d_10"]
        C12 = params["C2d_12"]

        C20 = params["C2d_20"]
        C21 = params["C2d_21"]

        H0 = params["H2d_0"]
        H1 = params["H2d_1"]
        H2 = params["H2d_2"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "2d_0")
        y1_gauss = self.gaussian(*args, "2d_1")
        y2_gauss = self.gaussian(*args, "2d_2")

        y0_lorentz = self.lorentzian(*args, "2d_0")
        y1_lorentz = self.lorentzian(*args, "2d_1")
        y2_lorentz = self.lorentzian(*args, "2d_2")

        diff = []

        y0_fit = (
            A0 * y0_gauss
            + H0 * y0_lorentz
            + B0
            + C01 * (x1[0, :, :] - c1)
            + C02 * (x2[0, :, :] - c2)
        )
        y1_fit = (
            A1 * y1_gauss
            + H1 * y1_lorentz
            + B1
            + C10 * (x0[:, 0, :] - c0)
            + C12 * (x2[:, 0, :] - c2)
        )
        y2_fit = (
            A2 * y2_gauss
            + H2 * y2_lorentz
            + B2
            + C20 * (x0[:, :, 0] - c0)
            + C21 * (x1[:, :, 0] - c1)
        )

        res = (y0_fit - y0) / e0

        diff += res.flatten().tolist()

        res = (y1_fit - y1) / e1

        diff += res.flatten().tolist()

        res = (y2_fit - y2) / e2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_2d(self, params, x0, x1, x2, ys, es):
        params_list = [name for name, par in params.items()]

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A0 = params["A2d_0"]
        A1 = params["A2d_1"]
        A2 = params["A2d_2"]

        # B0 = params['B2d_0']
        # B1 = params['B2d_1']
        # B2 = params['B2d_2']

        C01 = params["C2d_01"]
        C02 = params["C2d_02"]

        C10 = params["C2d_10"]
        C12 = params["C2d_12"]

        C20 = params["C2d_20"]
        C21 = params["C2d_21"]

        H0 = params["H2d_0"]
        H1 = params["H2d_1"]
        H2 = params["H2d_2"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "2d_0")
        y1_gauss = self.gaussian(*args, "2d_1")
        y2_gauss = self.gaussian(*args, "2d_2")

        y0_lorentz = self.lorentzian(*args, "2d_0")
        y1_lorentz = self.lorentzian(*args, "2d_1")
        y2_lorentz = self.lorentzian(*args, "2d_2")

        dA0 = y0_gauss / e0
        dA1 = y1_gauss / e1
        dA2 = y2_gauss / e2

        dH0 = y0_lorentz / e0
        dH1 = y1_lorentz / e1
        dH2 = y2_lorentz / e2

        dB0 = 1 / e0
        dB1 = 1 / e1
        dB2 = 1 / e2

        dC01 = (x1[0, :, :] - c1) / e0
        dC02 = (x2[0, :, :] - c2) / e0

        dC10 = (x0[:, 0, :] - c0) / e1
        dC12 = (x2[:, 0, :] - c2) / e1

        dC20 = (x0[:, :, 0] - c0) / e2
        dC21 = (x1[:, :, 0] - c1) / e2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_0")
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_1")
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_2")

        yc0_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="2d_0")
        yc1_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="2d_1")
        yc2_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="2d_2")

        dc0_0, dc1_0, dc2_0 = (A0 * yc0_gauss + H0 * yc0_lorentz) / e0
        dc0_1, dc1_1, dc2_1 = (A1 * yc1_gauss + H1 * yc1_lorentz) / e1
        dc0_2, dc1_2, dc2_2 = (A2 * yc2_gauss + H2 * yc2_lorentz) / e2

        dc1_0 -= C01 / e0
        dc2_0 -= C02 / e0

        dc0_1 -= C10 / e1
        dc2_1 -= C12 / e1

        dc0_2 -= C20 / e2
        dc1_2 -= C21 / e2

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_0")
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_1")
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_2")

        yr0_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="2d_0"
        )
        yr1_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="2d_1"
        )
        yr2_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode="2d_2"
        )

        dr0_0, dr1_0, dr2_0 = (A0 * yr0_gauss + H0 * yr0_lorentz) / e0
        dr0_1, dr1_1, dr2_1 = (A1 * yr1_gauss + H1 * yr1_lorentz) / e1
        dr0_2, dr1_2, dr2_2 = (A2 * yr2_gauss + H2 * yr2_lorentz) / e2

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_0")
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_1")
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_2")

        yu0_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="2d_0"
        )
        yu1_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="2d_1"
        )
        yu2_lorentz = self.lorentzian_jac_S(
            x0, x1, x2, c, inv_S, du, mode="2d_2"
        )

        du0_0, du1_0, du2_0 = (A0 * yu0_gauss + H0 * yu0_lorentz) / e0
        du0_1, du1_1, du2_1 = (A1 * yu1_gauss + H1 * yu1_lorentz) / e1
        du0_2, du1_2, du2_2 = (A2 * yu2_gauss + H2 * yu2_lorentz) / e2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)

        n01 = n0 + n1
        n012 = n01 + n2

        jac = np.zeros((n_params, n012))

        jac[params_list.index("A2d_0"), :n0] = dA0.flatten()
        jac[params_list.index("H2d_0"), :n0] = dH0.flatten()
        jac[params_list.index("B2d_0"), :n0] = dB0.flatten()
        jac[params_list.index("C2d_01"), :n0] = dC01.flatten()
        jac[params_list.index("C2d_02"), :n0] = dC02.flatten()
        jac[params_list.index("c0"), :n0] = dc0_0.flatten()
        jac[params_list.index("c1"), :n0] = dc1_0.flatten()
        jac[params_list.index("c2"), :n0] = dc2_0.flatten()
        jac[params_list.index("r0"), :n0] = dr0_0.flatten()
        jac[params_list.index("r1"), :n0] = dr1_0.flatten()
        jac[params_list.index("r2"), :n0] = dr2_0.flatten()
        jac[params_list.index("u0"), :n0] = du0_0.flatten()
        jac[params_list.index("u1"), :n0] = du1_0.flatten()
        jac[params_list.index("u2"), :n0] = du2_0.flatten()

        jac[params_list.index("A2d_1"), n0:n01] = dA1.flatten()
        jac[params_list.index("H2d_1"), n0:n01] = dH1.flatten()
        jac[params_list.index("B2d_1"), n0:n01] = dB1.flatten()
        jac[params_list.index("C2d_10"), n0:n01] = dC10.flatten()
        jac[params_list.index("C2d_12"), n0:n01] = dC12.flatten()
        jac[params_list.index("c0"), n0:n01] = dc0_1.flatten()
        jac[params_list.index("c1"), n0:n01] = dc1_1.flatten()
        jac[params_list.index("c2"), n0:n01] = dc2_1.flatten()
        jac[params_list.index("r0"), n0:n01] = dr0_1.flatten()
        jac[params_list.index("r1"), n0:n01] = dr1_1.flatten()
        jac[params_list.index("r2"), n0:n01] = dr2_1.flatten()
        jac[params_list.index("u0"), n0:n01] = du0_1.flatten()
        jac[params_list.index("u1"), n0:n01] = du1_1.flatten()
        jac[params_list.index("u2"), n0:n01] = du2_1.flatten()

        jac[params_list.index("A2d_2"), n01:n012] = dA2.flatten()
        jac[params_list.index("H2d_2"), n01:n012] = dH2.flatten()
        jac[params_list.index("B2d_2"), n01:n012] = dB2.flatten()
        jac[params_list.index("C2d_20"), n01:n012] = dC20.flatten()
        jac[params_list.index("C2d_21"), n01:n012] = dC21.flatten()
        jac[params_list.index("c0"), n01:n012] = dc0_2.flatten()
        jac[params_list.index("c1"), n01:n012] = dc1_2.flatten()
        jac[params_list.index("c2"), n01:n012] = dc2_2.flatten()
        jac[params_list.index("r0"), n01:n012] = dr0_2.flatten()
        jac[params_list.index("r1"), n01:n012] = dr1_2.flatten()
        jac[params_list.index("r2"), n01:n012] = dr2_2.flatten()
        jac[params_list.index("u0"), n01:n012] = du0_2.flatten()
        jac[params_list.index("u1"), n01:n012] = du1_2.flatten()
        jac[params_list.index("u2"), n01:n012] = du2_2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        diff = np.concatenate([1 / e.flatten() for e in es])

        mask = np.isfinite(diff)

        return jac[ind][:, mask]

    def residual_3d(self, params, x0, x1, x2, y, e):
        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A = params["A3d"]
        B = params["B3d"]
        H = params["H3d"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, "3d")
        y_lorentz = self.lorentzian(*args, "3d")

        diff = []

        y_fit = A * y_gauss + H * y_lorentz + B

        res = (y_fit - y) / e

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_3d(self, params, x0, x1, x2, y, e):
        params_list = [name for name, par in params.items()]

        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        r0 = params["r0"]
        r1 = params["r1"]
        r2 = params["r2"]

        u0 = params["u0"]
        u1 = params["u1"]
        u2 = params["u2"]

        A = params["A3d"]
        # B = params['B3d']
        H = params["H3d"]

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, "3d")
        y_lorentz = self.lorentzian(*args, "3d")

        dA = y_gauss / e

        dH = y_lorentz / e

        dB = 1 / e

        yc_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="3d")
        yc_lorentz = self.lorentzian_jac_c(x0, x1, x2, c, inv_S, mode="3d")

        dc0, dc1, dc2 = (A * yc_gauss + H * yc_lorentz) / e

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="3d")
        yr_lorentz = self.lorentzian_jac_S(x0, x1, x2, c, inv_S, dr, mode="3d")

        dr0, dr1, dr2 = (A * yr_gauss + H * yr_lorentz) / e

        yu_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="3d")
        yu_lorentz = self.lorentzian_jac_S(x0, x1, x2, c, inv_S, du, mode="3d")

        du0, du1, du2 = (A * yu_gauss + H * yu_lorentz) / e

        n, n_params = y.size, len(params)
        jac = np.zeros((n_params, n))

        jac[params_list.index("A3d"), :n] = dA.flatten()
        jac[params_list.index("H3d"), :n] = dH.flatten()
        jac[params_list.index("B3d"), :n] = dB.flatten()
        jac[params_list.index("c0"), :n] = dc0.flatten()
        jac[params_list.index("c1"), :n] = dc1.flatten()
        jac[params_list.index("c2"), :n] = dc2.flatten()
        jac[params_list.index("r0"), :n] = dr0.flatten()
        jac[params_list.index("r1"), :n] = dr1.flatten()
        jac[params_list.index("r2"), :n] = dr2.flatten()
        jac[params_list.index("u0"), :n] = du0.flatten()
        jac[params_list.index("u1"), :n] = du1.flatten()
        jac[params_list.index("u2"), :n] = du2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        mask = np.isfinite(1 / e.flatten())

        return jac[ind][:, mask]

    def regularization(self, params, lamda=1):
        beta = np.array([params[key] for key in params.keys()])
        ridge = lamda * np.array(beta)

        return ridge

    def regularization_jac(self, params, lamda=1):
        beta = np.array([params[key] for key in params.keys()])

        ridge = lamda * np.eye(len(beta))

        return ridge

    def residual(self, params, args_1d, args_2d, args_3d):
        cost_1d = self.residual_1d(params, *args_1d)
        cost_2d = self.residual_2d(params, *args_2d)
        cost_3d = self.residual_3d(params, *args_3d)

        # ridge = self.regularization(params)

        cost = np.concatenate([cost_1d, cost_2d, cost_3d])

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):
        jac_1d = self.jacobian_1d(params, *args_1d)
        jac_2d = self.jacobian_2d(params, *args_2d)
        jac_3d = self.jacobian_3d(params, *args_3d)

        # ridge = self.regularization_jac(params)

        jac = np.column_stack([jac_1d, jac_2d, jac_3d])

        return jac

    def loss(self, r):
        return np.abs(r).sum()

    def extract_result(self, result, args_1d, args_2d, args_3d):
        x0, x1, x2, y1d, e1d = args_1d
        x0, x1, x2, y2d, e2d = args_2d
        x0, x1, x2, y3d, e3d = args_3d

        y1d_0, y1d_1, y1d_2 = y1d
        y2d_0, y2d_1, y2d_2 = y2d

        e1d_0, e1d_1, e1d_2 = e1d
        e2d_0, e2d_1, e2d_2 = e2d

        self.redchi2 = []
        self.intensity = []
        self.sigma = []

        self.params = result.params

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c0_err = self.params["c0"].stderr
        c1_err = self.params["c1"].stderr
        c2_err = self.params["c2"].stderr

        if c0_err is None:
            c0_err = c0
        if c1_err is None:
            c1_err = c1
        if c2_err is None:
            c2_err = c2

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c_err = c0_err, c1_err, c2_err

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        args = x0, x1, x2, c, inv_S

        A0 = self.params["A1d_0"]
        A1 = self.params["A1d_1"]
        A2 = self.params["A1d_2"]

        H0 = self.params["H1d_0"]
        H1 = self.params["H1d_1"]
        H2 = self.params["H1d_2"]

        B0 = self.params["B1d_0"]
        B1 = self.params["B1d_1"]
        B2 = self.params["B1d_2"]

        C0 = self.params["C1d_0"]
        C1 = self.params["C1d_1"]
        C2 = self.params["C1d_2"]

        y1d_0_fit = (
            A0 * self.gaussian(*args, "1d_0")
            + H0 * self.lorentzian(*args, "1d_0")
            + B0
            + C0 * (x0[:, 0, 0] - c0)
        )
        y1d_1_fit = (
            A1 * self.gaussian(*args, "1d_1")
            + H1 * self.lorentzian(*args, "1d_1")
            + B1
            + C1 * (x1[0, :, 0] - c1)
        )
        y1d_2_fit = (
            A2 * self.gaussian(*args, "1d_2")
            + H2 * self.lorentzian(*args, "1d_2")
            + B2
            + C2 * (x2[0, 0, :] - c2)
        )

        y1d_0_fit[~(np.isfinite(y1d_0) & (e1d_0 > 0))] = np.nan
        y1d_1_fit[~(np.isfinite(y1d_1) & (e1d_1 > 0))] = np.nan
        y1d_2_fit[~(np.isfinite(y1d_2) & (e1d_2 > 0))] = np.nan

        y1 = [
            (y1d_0_fit, y1d_0, e1d_0),
            (y1d_1_fit, y1d_1, e1d_1),
            (y1d_2_fit, y1d_2, e1d_2),
        ]

        chi2_1d = []
        chi2_1d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y1[0], "1d_0"))
        chi2_1d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y1[1], "1d_1"))
        chi2_1d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y1[2], "1d_2"))

        self.redchi2.append(chi2_1d)

        I0, s0 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y1[0], "1d_0")
        I1, s1 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y1[1], "1d_1")
        I2, s2 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y1[2], "1d_2")

        self.intensity.append([I0, I1, I2])
        self.sigma.append([s0, s1, s2])

        # ---

        A0 = self.params["A2d_0"]
        A1 = self.params["A2d_1"]
        A2 = self.params["A2d_2"]

        H0 = self.params["H2d_0"]
        H1 = self.params["H2d_1"]
        H2 = self.params["H2d_2"]

        B0 = self.params["B2d_0"]
        B1 = self.params["B2d_1"]
        B2 = self.params["B2d_2"]

        C01 = self.params["C2d_01"]
        C02 = self.params["C2d_02"]

        C10 = self.params["C2d_10"]
        C12 = self.params["C2d_12"]

        C20 = self.params["C2d_20"]
        C21 = self.params["C2d_21"]

        y2d_0_fit = (
            A0 * self.gaussian(*args, "2d_0")
            + H0 * self.lorentzian(*args, "2d_0")
            + B0
            + C01 * (x1[0, :, :] - c1)
            + C02 * (x2[0, :, :] - c2)
        )
        y2d_1_fit = (
            A1 * self.gaussian(*args, "2d_1")
            + H1 * self.lorentzian(*args, "2d_1")
            + B1
            + C10 * (x0[:, 0, :] - c0)
            + C12 * (x2[:, 0, :] - c2)
        )
        y2d_2_fit = (
            A2 * self.gaussian(*args, "2d_2")
            + H2 * self.lorentzian(*args, "2d_2")
            + B2
            + C20 * (x0[:, :, 0] - c0)
            + C21 * (x1[:, :, 0] - c1)
        )

        y2d_0_fit[~(np.isfinite(y2d_0) & (e2d_0 > 0))] = np.nan
        y2d_1_fit[~(np.isfinite(y2d_1) & (e2d_1 > 0))] = np.nan
        y2d_2_fit[~(np.isfinite(y2d_2) & (e2d_2 > 0))] = np.nan

        y2 = [
            (y2d_0_fit, y2d_0, e2d_0),
            (y2d_1_fit, y2d_1, e2d_1),
            (y2d_2_fit, y2d_2, e2d_2),
        ]

        chi2_2d = []
        chi2_2d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y2[0], "2d_0"))
        chi2_2d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y2[1], "2d_1"))
        chi2_2d.append(self.chi_2_fit(x0, x1, x2, c, inv_S, *y2[2], "2d_2"))

        self.redchi2.append(chi2_2d)

        I0, s0 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y2[0], "2d_0")
        I1, s1 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y2[1], "2d_1")
        I2, s2 = self.estimate_intensity(x0, x1, x2, c, inv_S, *y2[2], "2d_2")

        self.intensity.append([I0, I1, I2])
        self.sigma.append([s0, s1, s2])

        # ---

        B = self.params["B3d"].value
        A = self.params["A3d"].value
        H = self.params["H3d"].value

        y3d_fit = (
            A * self.gaussian(*args, "3d")
            + H * self.lorentzian(*args, "3d")
            + B
        )

        y3d_fit[~(np.isfinite(y3d) & (e3d >= 0))] = np.nan

        y3 = (y3d_fit, y3d, e3d)

        chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, *y3, "3d")

        self.redchi2.append(chi2)

        I, s = self.estimate_intensity(x0, x1, x2, c, inv_S, *y3, "3d")

        self.intensity.append(I)
        self.sigma.append(s)

        return c, c_err, inv_S, y1, y2, y3

    def estimate_envelope(self, x0, x1, x2, d, n, gd, gn, report_fit=False):
        y = gd / gn
        e = np.sqrt(gd) / gn

        if (np.array(e.shape) < 3).any():
            return None

        if np.sum(d) < 3 * np.sqrt(np.sum(d)):
            return None

        y1d_0, e1d_0 = self.normalize(x0, x1, x2, gd, gn, mode="1d_0")
        y1d_1, e1d_1 = self.normalize(x0, x1, x2, gd, gn, mode="1d_1")
        y1d_2, e1d_2 = self.normalize(x0, x1, x2, gd, gn, mode="1d_2")

        y0, y1, y2 = y1d_0.copy(), y1d_1.copy(), y1d_2.copy()

        y0_min = np.nanmin(y0)
        y0_max = np.nanmax(y0)

        if np.isclose(y0_max, y0_min) or (y0 > 0).sum() <= 3:
            return None

        y1_min = np.nanmin(y1)
        y1_max = np.nanmax(y1)

        if np.isclose(y1_max, y1_min) or (y1 > 0).sum() <= 3:
            return None

        y2_min = np.nanmin(y2)
        y2_max = np.nanmax(y2)

        if np.isclose(y2_max, y2_min) or (y2 > 0).sum() <= 3:
            return None

        self.params.add("A1d_0", value=y0_max, min=0, max=2 * y0_max)
        self.params.add("A1d_1", value=y1_max, min=0, max=2 * y1_max)
        self.params.add("A1d_2", value=y2_max, min=0, max=2 * y2_max)

        self.params.add("H1d_0", value=0, min=0, max=2 * y0_max, vary=False)
        self.params.add("H1d_1", value=0, min=0, max=2 * y1_max, vary=False)
        self.params.add("H1d_2", value=0, min=0, max=2 * y2_max, vary=False)

        self.params.add("B1d_0", value=y0_min, min=-y0_max, max=5 * y0_max)
        self.params.add("B1d_1", value=y1_min, min=-y1_max, max=5 * y1_max)
        self.params.add("B1d_2", value=y2_min, min=-y2_max, max=5 * y2_max)

        C0_max = (y0_max - y0_min) / (x0[-1, 0, 0] - x0[0, 0, 0])
        C1_max = (y1_max - y1_min) / (x1[0, -1, 0] - x1[0, 0, 0])
        C2_max = (y2_max - y2_min) / (x2[0, 0, -1] - x2[0, 0, 0])

        self.params.add(
            "C1d_0", value=0, min=-2 * C0_max, max=2 * C0_max, vary=False
        )
        self.params.add(
            "C1d_1", value=0, min=-2 * C1_max, max=2 * C1_max, vary=False
        )
        self.params.add(
            "C1d_2", value=0, min=-2 * C2_max, max=2 * C2_max, vary=False
        )

        y1d = [y1d_0, y1d_1, y1d_2]
        e1d = [e1d_0, e1d_1, e1d_2]

        args_1d = [x0, x1, x2, y1d, e1d]

        y2d_0, e2d_0 = self.normalize(x0, x1, x2, d, n, mode="2d_0")
        y2d_1, e2d_1 = self.normalize(x0, x1, x2, d, n, mode="2d_1")
        y2d_2, e2d_2 = self.normalize(x0, x1, x2, d, n, mode="2d_2")

        y0, y1, y2 = y2d_0.copy(), y2d_1.copy(), y2d_2.copy()

        y0_min = np.nanmin(y0)
        y0_max = np.nanmax(y0)

        if np.isclose(y0_max, y0_min) or (y0 > 0).sum() <= 3:
            return None

        y1_min = np.nanmin(y1)
        y1_max = np.nanmax(y1)

        if np.isclose(y1_max, y1_min) or (y1 > 0).sum() <= 3:
            return None

        y2_min = np.nanmin(y2)
        y2_max = np.nanmax(y2)

        if np.isclose(y2_max, y2_min) or (y2 > 0).sum() <= 3:
            return None

        self.params.add("A2d_0", value=y0_max, min=0, max=2 * y0_max)
        self.params.add("A2d_1", value=y1_max, min=0, max=2 * y1_max)
        self.params.add("A2d_2", value=y2_max, min=0, max=2 * y2_max)

        self.params.add("H2d_0", value=0, min=0, max=2 * y0_max, vary=False)
        self.params.add("H2d_1", value=0, min=0, max=2 * y1_max, vary=False)
        self.params.add("H2d_2", value=0, min=0, max=2 * y2_max, vary=False)

        self.params.add("B2d_0", value=y0_min, min=-y0_max, max=5 * y0_max)
        self.params.add("B2d_1", value=y1_min, min=-y1_max, max=5 * y1_max)
        self.params.add("B2d_2", value=y2_min, min=-y2_max, max=5 * y2_max)

        C01_max = (y0_max - y0_min) / (x1[0, -1, 0] - x1[0, 0, 0])
        C02_max = (y0_max - y0_min) / (x2[0, 0, -1] - x2[0, 0, 0])

        C10_max = (y1_max - y1_min) / (x0[-1, 0, 0] - x0[0, 0, 0])
        C12_max = (y1_max - y1_min) / (x2[0, 0, -1] - x2[0, 0, 0])

        C20_max = (y2_max - y2_min) / (x0[-1, 0, 0] - x0[0, 0, 0])
        C21_max = (y2_max - y2_min) / (x1[0, -1, 0] - x1[0, 0, 0])

        self.params.add(
            "C2d_01", value=0, min=-2 * C01_max, max=2 * C01_max, vary=False
        )
        self.params.add(
            "C2d_02", value=0, min=-2 * C02_max, max=2 * C02_max, vary=False
        )

        self.params.add(
            "C2d_10", value=0, min=-2 * C10_max, max=2 * C10_max, vary=False
        )
        self.params.add(
            "C2d_12", value=0, min=-2 * C12_max, max=2 * C12_max, vary=False
        )

        self.params.add(
            "C2d_20", value=0, min=-2 * C20_max, max=2 * C20_max, vary=False
        )
        self.params.add(
            "C2d_21", value=0, min=-2 * C21_max, max=2 * C21_max, vary=False
        )

        y2d = [y2d_0, y2d_1, y2d_2]
        e2d = [e2d_0, e2d_1, e2d_2]

        args_2d = [x0, x1, x2, y2d, e2d]

        y3d, e3d = self.normalize(x0, x1, x2, d, n, mode="3d")

        y_min = np.nanmin(y3d)
        y_max = np.nanmax(y3d)

        if np.isclose(y_max, y_min) or (y > 0).sum() <= 3:
            return None

        self.params.add("A3d", value=y_max, min=0, max=2 * y_max)

        self.params.add("H3d", value=0, min=0, max=2 * y_max, vary=False)

        self.params.add("B3d", value=y_min, min=-2 * y_max, max=2 * y_max)

        args_3d = [x0, x1, x2, y3d, e3d]

        self.params["c0"].set(vary=True)
        self.params["c1"].set(vary=True)
        self.params["c2"].set(vary=True)

        out = Minimizer(
            self.residual,
            self.params,
            fcn_args=(args_1d, args_2d, args_3d),
            # nan_policy="omit",
        )

        result = out.minimize(
            method="leastsq",
            Dfun=self.jacobian,
            max_nfev=200,
            col_deriv=True,
        )

        if report_fit:
            print(fit_report(result))

        self.params = result.params

        y = d / n
        e = np.sqrt(d) / n

        if (np.array(e.shape) < 3).any():
            return None

        if np.sum(d) < 3 * np.sqrt(np.sum(d)):
            return None

        y1d_0, e1d_0 = self.normalize(x0, x1, x2, d, n, mode="1d_0")
        y1d_1, e1d_1 = self.normalize(x0, x1, x2, d, n, mode="1d_1")
        y1d_2, e1d_2 = self.normalize(x0, x1, x2, d, n, mode="1d_2")

        y1d = [y1d_0, y1d_1, y1d_2]
        e1d = [e1d_0, e1d_1, e1d_2]

        args_1d = [x0, x1, x2, y1d, e1d]

        y2d_0, e2d_0 = self.normalize(x0, x1, x2, d, n, mode="2d_0")
        y2d_1, e2d_1 = self.normalize(x0, x1, x2, d, n, mode="2d_1")
        y2d_2, e2d_2 = self.normalize(x0, x1, x2, d, n, mode="2d_2")

        y2d = [y2d_0, y2d_1, y2d_2]
        e2d = [e2d_0, e2d_1, e2d_2]

        args_2d = [x0, x1, x2, y2d, e2d]

        y3d, e3d = self.normalize(x0, x1, x2, d, n, mode="3d")

        args_3d = [x0, x1, x2, y3d, e3d]

        self.params["c0"].set(vary=False)
        self.params["c1"].set(vary=False)
        self.params["c2"].set(vary=False)

        self.params["u0"].set(vary=False)
        self.params["u1"].set(vary=False)
        self.params["u2"].set(vary=False)

        # r0 = self.params["r0"].value
        # r1 = self.params["r1"].value
        # r2 = self.params["r2"].value

        self.params["r0"].set(vary=False)
        self.params["r1"].set(vary=False)
        self.params["r2"].set(vary=False)

        # self.params["r1"].set(expr="{}*r0".format(r1 / r0))
        # self.params["r2"].set(expr="{}*r0".format(r2 / r0))

        out = Minimizer(
            self.residual,
            self.params,
            fcn_args=(args_1d, args_2d, args_3d),
            nan_policy="omit",
        )

        result = out.minimize(
            method="leastsq",
            Dfun=self.jacobian,
            max_nfev=30,
            col_deriv=True,
        )

        return self.extract_result(result, args_1d, args_2d, args_3d)

    def calculate_intensity(self, A, H, r0, r1, r2, u0, u1, u2, mode="3d"):
        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)
        g = self.gaussian_integral(inv_S, mode)
        l = self.lorentzian_integral(inv_S, mode)

        return A * g, H * l

    def voxels(self, x0, x1, x2):
        return (
            x0[1, 0, 0] - x0[0, 0, 0],
            x1[0, 1, 0] - x1[0, 0, 0],
            x2[0, 0, 1] - x2[0, 0, 0],
        )

    def voxel_volume(self, x0, x1, x2):
        return np.prod(self.voxels(x0, x1, x2))

    def fit(
        self,
        x0_prof,
        x1_proj,
        x2_proj,
        d_val,
        n_val,
        gd,
        gn,
        val_mask,
        det_mask,
        dx,
        xmod,
    ):
        x0 = x0_prof - xmod
        x1 = x1_proj.copy()
        x2 = x2_proj.copy()

        y, e = d_val / n_val, np.sqrt(d_val) / n_val

        if (np.array(y.shape) <= 3).any():
            return None

        self.update_constraints(x0, x1, x2, dx)

        y_max = np.nanmax(y)

        if y_max <= 0 or np.sum(val_mask) < 11:
            return None

        coords = np.argwhere(det_mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0) + 1

        y = y[i0:j0, i1:j1, i2:j2].copy()
        e = e[i0:j0, i1:j1, i2:j2].copy()

        d = d_val[i0:j0, i1:j1, i2:j2].copy()
        n = n_val[i0:j0, i1:j1, i2:j2].copy()

        gd = gd[i0:j0, i1:j1, i2:j2].copy()
        gn = gn[i0:j0, i1:j1, i2:j2].copy()

        if (np.array(y.shape) <= 3).any():
            return None

        x0 = x0[i0:j0, i1:j1, i2:j2].copy()
        x1 = x1[i0:j0, i1:j1, i2:j2].copy()
        x2 = x2[i0:j0, i1:j1, i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print("Invalid data")
            return None

        weights = None
        try:
            weights = self.estimate_envelope(x0, x1, x2, d, n, gd, gn)
        except Exception as e:
            print("Exception estimating envelope: {}".format(e))
            return None

        if weights is None:
            print("Invalid weight estimate")
            return None

        c, c_err, inv_S, vals1d, vals2d, vals3d = weights

        if not np.linalg.det(inv_S) > 0:
            print("Improper optimal covariance")
            return None

        S = np.linalg.inv(inv_S)

        V, W = np.linalg.eigh(S)

        c0, c1, c2 = c

        c0 += xmod
        c = c0, c1, c2

        r0, r1, r2 = np.sqrt(V)

        v0, v1, v2 = W.T

        fitting = (x0 + xmod, x1, x2, *vals3d)

        self.peak_pos = c, c_err

        self.best_fit = c, S, *fitting

        self.best_prof = (
            (x0[:, 0, 0] + xmod, *vals1d[0]),
            (x1[0, :, 0], *vals1d[1]),
            (x2[0, 0, :], *vals1d[2]),
        )

        self.best_proj = (
            (x1[0, :, :], x2[0, :, :], *vals2d[0]),
            (x0[:, 0, :] + xmod, x2[:, 0, :], *vals2d[1]),
            (x0[:, :, 0] + xmod, x1[:, :, 0], *vals2d[2]),
        )

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def peak_roi(self, x0, x1, x2, c, S, det_mask, p=0.997):
        c0, c1, c2 = c

        x = np.array([x0 - c0, x1 - c1, x2 - c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum("ij,jklm,iklm->klm", S_inv, x, x)

        pk = ellipsoid <= 1  # & val_mask

        bkg = (ellipsoid > 1) & (ellipsoid < np.cbrt(2) ** 2)  # & val_mask

        scale = scipy.stats.chi2.ppf(p, df=3)

        inv_var = scale * S_inv

        d2 = np.einsum("i...,ij,j...->...", x, inv_var, x)

        det = 1 / np.linalg.det(inv_var)

        y = np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** 3 * det)

        d3x = self.voxel_volume(x0, x1, x2)

        val_mask = pk & det_mask

        return pk, bkg, np.nansum(y[val_mask]) * d3x

    def extract_raw_intensity(self, counts, pk, bkg):
        d = counts.copy()
        d[np.isinf(d)] = np.nan

        d_pk = d[pk].copy()
        d_bkg = d[bkg].copy()

        b = np.nanmean(d_bkg)
        b_err = np.sqrt(np.nanmean(d_bkg))

        intens = np.nansum(d_pk - b)
        sig = np.sqrt(np.nansum(d_pk + b_err**2))

        if not sig > 0:
            sig = float("inf")

        return intens, sig

    def extract_intensity(self, d, n, pk, bkg):
        d_pk = d[pk].copy()
        n_pk = n[pk].copy()

        d_bkg = d[bkg].copy()
        n_bkg = n[bkg].copy()

        bkg_cnts = np.nansum(d_bkg)
        bkg_norm = np.nansum(n_bkg)

        if bkg_cnts == 0.0:
            bkg_cnts = float("nan")
        if bkg_norm == 0.0:
            bkg_norm = float("nan")

        b = bkg_cnts / bkg_norm
        b_err = np.sqrt(bkg_cnts) / bkg_norm

        vox = float(np.nansum(n_pk > 0))

        if not np.isfinite(b):
            b = 0
        if not np.isfinite(b_err):
            b_err = 0

        pk_cnts = np.nansum(d_pk)
        pk_norm = np.nansum(n_pk)

        if pk_cnts == 0.0:
            pk_cnts = float("nan")
        if pk_cnts == 0.0:
            pk_norm = float("nan")

        intens = vox * (pk_cnts / pk_norm - b)
        sig = vox * np.sqrt(pk_cnts / pk_norm**2 + b_err**2)

        if not sig > 0:
            sig = float("inf")

        return intens, sig, b, b_err, vox, pk_cnts, pk_norm, bkg_cnts, bkg_norm

    def integrate(self, x0, x1, x2, d, n, val_mask, det_mask, c, S):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d3x = self.voxel_volume(x0, x1, x2)

        pk, bkg, vol_fract = self.peak_roi(x0, x1, x2, c, S, det_mask)

        d[np.isinf(d)] = np.nan
        n[np.isinf(n)] = np.nan

        result = self.extract_intensity(d, n, pk, bkg)

        intens, sig, b, b_err, N, pk_data, pk_norm, bkg_data, bkg_norm = result

        intens *= d3x
        sig *= d3x

        # intens /= vol_fract
        # sig /= vol_fract

        self.intensity.append(intens)
        self.sigma.append(sig)

        self.weights = (x0[pk], x1[pk], x2[pk]), d[pk].copy()

        self.info = [d3x, b, b_err]

        freq = d / n  # - np.nanmean(d[bkg] / n[bkg])
        freq[~(pk | bkg)] = np.nan

        intens_raw, sig_raw = self.extract_raw_intensity(d, pk, bkg)

        self.info += [intens_raw, sig_raw]

        self.info += [N, pk_data, pk_norm, bkg_data, bkg_norm]

        if not np.isfinite(sig):
            sig = float("inf")

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        return intens, sig

    def sigma_clip(self, array, sigma=3, maxiters=5):
        array = np.array(array, dtype=float)
        y = array[np.isfinite(array)]
        mask = np.ones_like(y, dtype=bool)

        for _ in range(maxiters):
            data = y[mask]

            med = np.median(data)
            mad = scipy.stats.median_abs_deviation(data, scale="normal")

            if mad <= 0:
                break

            dev = np.abs(y - med)
            new_mask = dev < (sigma * mad)

            if np.all(new_mask == mask):
                break

            mask = new_mask.copy()

        data = y[mask]
        med = np.median(data)
        mad = scipy.stats.median_abs_deviation(data, scale="normal")

        return med, mad

    def background(self, y, e):
        weights = np.abs(y) / e
        mask = np.isfinite(weights)

        values = y[mask]

        if values.size <= 1:
            return 0, 0

        indices = np.argsort(values)
        sorted_values = values[indices]
        sorted_weights = weights[indices]

        cumulative_weights = np.cumsum(sorted_weights)
        half_weight = cumulative_weights[-1] / 2

        median_index = np.searchsorted(cumulative_weights, half_weight)
        b = sorted_values[median_index]

        dev = np.abs(values - b)
        mad_ind = np.searchsorted(np.cumsum(weights), half_weight)
        mad = dev[indices][mad_ind]

        b_err = 1.4826 * mad

        return b, b_err
