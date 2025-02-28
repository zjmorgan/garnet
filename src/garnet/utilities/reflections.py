import os

# import sys

from mantid.simpleapi import (
    LoadNexus,
    FilterPeaks,
    SortPeaksWorkspace,
    CombinePeaksWorkspaces,
    StatisticsOfPeaksWorkspace,
    SaveHKL,
    SaveIsawUB,
    CloneWorkspace,
    SetGoniometer,
    SetSample,
    AddAbsorptionWeightedPathLengths,
    mtd,
)

import re

import matplotlib.pyplot as plt
import numpy as np

import scipy.optimize
import scipy.interpolate

# from mantid.geometry import PointGroupFactory
from mantid import config

config["Q.convention"] = "Crystallography"

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
from PyPDF2 import PdfMerger

import argparse

point_group_dict = {
    "-1": "-1 (Triclinic)",
    "1": "1 (Triclinic)",
    "2": "2 (Monoclinic, unique axis b)",
    "m": "m (Monoclinic, unique axis b)",
    "2/m": "2/m (Monoclinic, unique axis b)",
    "112": "112 (Monoclinic, unique axis c)",
    "11m": "11m (Monoclinic, unique axis c)",
    "112/m": "112/m (Monoclinic, unique axis c)",
    "222": "222 (Orthorhombic)",
    "2mm": "2mm (Orthorhombic)",
    "m2m": "m2m (Orthorhombic)",
    "mm2": "mm2 (Orthorhombic)",
    "mmm": "mmm (Orthorhombic)",
    "3": "3 (Trigonal - Hexagonal)",
    "32": "32 (Trigonal - Hexagonal)",
    "312": "312 (Trigonal - Hexagonal)",
    "321": "321 (Trigonal - Hexagonal)",
    "31m": "31m (Trigonal - Hexagonal)",
    "3m": "3m (Trigonal - Hexagonal)",
    "3m1": "3m1 (Trigonal - Hexagonal)",
    "-3": "-3 (Trigonal - Hexagonal)",
    "-31m": "-31m (Trigonal - Hexagonal)",
    "-3m": "-3m (Trigonal - Hexagonal)",
    "-3m1": "-3m1 (Trigonal - Hexagonal)",
    "3 r": "3 r (Trigonal - Rhombohedral)",
    "32 r": "32 r (Trigonal - Rhombohedral)",
    "3m r": "3m r (Trigonal - Rhombohedral)",
    "-3 r": "-3 r (Trigonal - Rhombohedral)",
    "-3m r": "-3m r (Trigonal - Rhombohedral)",
    "4": "4 (Tetragonal)",
    "4/m": "4/m (Tetragonal)",
    "4mm": "4mm (Tetragonal)",
    "422": "422 (Tetragonal)",
    "-4": "-4 (Tetragonal)",
    "-42m": "-42m (Tetragonal)",
    "-4m2": "-4m2 (Tetragonal)",
    "4/mmm": "4/mmm (Tetragonal)",
    "6": "6 (Hexagonal)",
    "6/m": "6/m (Hexagonal)",
    "6mm": "6mm (Hexagonal)",
    "622": "622 (Hexagonal)",
    "-6": "-6 (Hexagonal)",
    "-62m": "-62m (Hexagonal)",
    "-6m2": "-6m2 (Hexagonal)",
    "6/mmm": "6/mmm (Hexagonal)",
    "23": "23 (Cubic)",
    "m-3": "m-3 (Cubic)",
    "432": "432 (Cubic)",
    "-43m": "-43m (Cubic)",
    "m-3m": "m-3m (Cubic)",
}


class AbsorptionCorrection:
    def __init__(
        self,
        peaks,
        chemical_formula,
        z_parameter,
        u_vector=[0, 0, 1],
        v_vector=[1, 0, 0],
        params=[0.1, 0.2, 0.4],
        shape="plate",
        filename=None,
    ):
        assert "PeaksWorkspace" in str(type(mtd[peaks]))

        self.peaks = peaks

        volume = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert volume > 0

        self.volume = volume

        assert self.verify_chemical_formula(chemical_formula)

        self.chemical_formula = chemical_formula

        assert z_parameter > 0
        self.z_parameter = z_parameter

        assert len(u_vector) == 3
        assert len(v_vector) == 3

        assert not np.isclose(np.linalg.norm(np.cross(u_vector, v_vector)), 0)

        self.u_vector = u_vector
        self.v_vector = v_vector

        if shape == "sphere":
            assert len(params) == 1
        elif shape == "cylinder":
            assert len(params) == 2
        else:
            assert len(params) == 3

        self.shape = shape
        self.params = params

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.set_shape()
        self.set_material()
        self.set_orientation()
        self.calculate_correction()
        self.write_absortion_parameters()
        self.apply_correction()

    def verify_chemical_formula(self, formula):
        pattern = (
            r"(?:\((?:[A-Z][a-z]?\d+)\)|[A-Z][a-z]?)(?:\d+(?:\.\d+)?|\.\d+)?"
        )

        parts = re.split(r"[-\s]+", formula.strip())

        return all(re.fullmatch(pattern, part) for part in parts)

    def set_shape(self):
        self.UB = mtd[self.peaks].sample().getOrientedLattice().getUB().copy()

        u = np.dot(self.UB, self.u_vector)
        v = np.dot(self.UB, self.v_vector)

        u /= np.linalg.norm(u)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        T = np.column_stack([v, w, u])

        gon = mtd[self.peaks].run().getGoniometer()

        gon.setR(T)
        gamma, beta, alpha = gon.getEulerAngles("ZYX")

        if self.shape == "sphere":
            shape = ' \
            <sphere id="sphere"> \
            <radius val="{}" /> \
            <centre x="0.0" y="0.0" z="0.0" /> \
            <rotate x="{}" y="{}" z="{}" /> \
            </sphere> \
            '.format(
                self.params[0] / 2000, alpha, beta, gamma
            )
        elif self.shape == "cylinder":
            shape = ' \
            <cylinder id="cylinder"> \
            <centre-of-bottom-base r="0.0" t="0.0" p="0.0" />   \
            <axis x="0.0" y="1.0" z="0" /> \
            <radius val="{}" />  \
            <height val="{}" />  \
            <rotate x="{}" y="{}" z="{}" /> \
            </cylinder> \
          '.format(
                self.params[0] / 2000,
                self.params[1] / 1000,
                alpha,
                beta,
                gamma,
            )
        else:
            shape = ' \
            <cuboid id="cuboid"> \
            <width val="{}" /> \
            <height val="{}"  /> \
            <depth val="{}" /> \
            <centre x="0.0" y="0.0" z="0.0"  /> \
            <rotate x="{}" y="{}" z="{}" /> \
            </cuboid> \
            <algebra val="cuboid" /> \
          '.format(
                self.params[0] / 1000,
                self.params[1] / 1000,
                self.params[2] / 1000,
                alpha,
                beta,
                gamma,
            )

        self.shape_dict = {"Shape": "CSG", "Value": shape}

    def set_material(self):
        self.mat_dict = {
            "ChemicalFormula": self.chemical_formula,
            "ZParameter": float(self.z_parameter),
            "UnitCellVolume": self.volume,
        }

    def set_orientation(self):
        Rs = [peak.getGoniometerMatrix() for peak in mtd[self.peaks]]
        matrix_dict = {}

        runs = []
        for peak in mtd[self.peaks]:
            R = peak.getGoniometerMatrix()

            matrix_tuple = tuple(R.flatten())

            if matrix_tuple in matrix_dict:
                run = matrix_dict[matrix_tuple]
            else:
                ind = np.isclose(Rs, R).all(axis=(1, 2))
                i = -1 if not np.any(ind) else ind.tolist().index(True)
                run = i + 1
                matrix_dict[matrix_tuple] = run

            runs.append(run)
            peak.setRunNumber(run)

        self.runs = np.unique(runs).astype(int).tolist()
        self.Rs = Rs

    def calculate_correction(self):
        peaks = self.peaks + "_corr"

        filename = os.path.splitext(self.filename)[0] + "_abs.pdf"

        with PdfPages(filename) as pdf:
            for i, (R, run) in enumerate(zip(self.Rs, self.runs)):
                FilterPeaks(
                    InputWorkspace=self.peaks,
                    FilterVariable="RunNumber",
                    FilterValue=run,
                    Operator="=",
                    OutputWorkspace="_tmp",
                )

                R = mtd["_tmp"].getPeak(0).getGoniometerMatrix()

                mtd["_tmp"].run().getGoniometer().setR(R)
                omega, chi, phi = (
                    mtd["_tmp"].run().getGoniometer().getEulerAngles("YZY")
                )

                SetGoniometer(
                    Workspace="_tmp",
                    Axis0="{},0,1,0,1".format(omega),
                    Axis1="{},0,0,1,1".format(chi),
                    Axis2="{},0,1,0,1".format(phi),
                )

                SetSample(
                    InputWorkspace="_tmp",
                    Geometry=self.shape_dict,
                    Material=self.mat_dict,
                )

                AddAbsorptionWeightedPathLengths(
                    InputWorkspace="_tmp", ApplyCorrection=False
                )

                if i == 0:
                    CloneWorkspace(
                        InputWorkspace="_tmp", OutputWorkspace=peaks
                    )
                else:
                    CombinePeaksWorkspaces(
                        LHSWorkspace=peaks,
                        RHSWorkspace="_tmp",
                        OutputWorkspace=peaks,
                    )

                hkl = np.eye(3)
                s = np.matmul(self.UB, hkl)

                reciprocal_lattice = np.matmul(R, s)

                shape = mtd["_tmp"].sample().getShape()
                mesh = shape.getMesh() * 100

                mesh_polygon = Poly3DCollection(
                    mesh,
                    edgecolors="k",
                    facecolors="w",
                    alpha=0.5,
                    linewidths=1,
                )

                fig, ax = plt.subplots(
                    subplot_kw={"projection": "mantid3d", "proj_type": "persp"}
                )
                ax.add_collection3d(mesh_polygon)

                ax.set_title("run #{}".format(1 + i))
                ax.set_xlabel("x [cm]")
                ax.set_ylabel("y [cm]")
                ax.set_zlabel("z [cm]")

                ax.set_mesh_axes_equal(mesh)
                ax.set_box_aspect((1, 1, 1))

                colors = ["r", "g", "b"]
                origin = (
                    ax.get_xlim3d()[1],
                    ax.get_ylim3d()[1],
                    ax.get_zlim3d()[1],
                )
                origin = (0, 0, 0)
                lims = ax.get_xlim3d()
                factor = (lims[1] - lims[0]) / 3

                for j in range(3):
                    vector = reciprocal_lattice[:, j]
                    vector_norm = vector / np.linalg.norm(vector)
                    ax.quiver(
                        origin[0],
                        origin[1],
                        origin[2],
                        vector_norm[0],
                        vector_norm[1],
                        vector_norm[2],
                        length=factor,
                        color=colors[j],
                        linestyle="-",
                    )

                    ax.view_init(vertical_axis="y", elev=27, azim=50)

                pdf.savefig(fig, dpi=100, bbox_inches=None)
                plt.close(fig)
                plt.close("all")

        CloneWorkspace(InputWorkspace=peaks, OutputWorkspace=self.peaks)

    def write_absortion_parameters(self):
        mat = mtd["_tmp"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective  # A^-3
        N = mat.totalAtoms

        vol = mtd["_tmp"].sample().getShape().volume()

        V = np.abs(vol * 100**3)  # cm^3

        rho = (n / N) / 0.6022 * M
        m = rho * V * 1000  # mg
        r = np.cbrt(0.75 * np.pi * V)

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        mu = mat.numberDensityEffective * (
            mat.totalScatterXSection() + mat.absorbXSection(1.8)
        )

        lines = [
            "{}\n".format(self.chemical_formula),
            "absoption cross section: {:.4f} barn\n".format(sigma_a),
            "scattering cross section: {:.4f} barn\n".format(sigma_s),
            "linear absorption coefficient: {:.4f} 1/cm\n".format(mu_a),
            "linear scattering coefficient: {:.4f} 1/cm\n".format(mu_s),
            "absorption parameter: {:.4f} \n".format(mu * r),
            "total atoms: {:.4f}\n".format(N),
            "molar mass: {:.4f} g/mol\n".format(M),
            "number density: {:.4f} 1/A^3\n".format(n),
            "mass density: {:.4f} g/cm^3\n".format(rho),
            "volume: {:.4f} cm^3\n".format(V),
            "mass: {:.4f} mg\n".format(m),
        ]

        for line in lines:
            print(line)

        if self.filename is not None:
            filename = os.path.splitext(self.filename)[0] + "_abs.txt"

            with open(filename, "w") as f:
                for line in lines:
                    f.write(line)

    def apply_correction(self):
        peaks = self.peaks + "_corr"

        mat = mtd["_tmp"].sample().getMaterial()

        for peak in mtd[peaks]:
            lamda = peak.getWavelength()
            Tbar = peak.getAbsorptionWeightedPathLength()

            mu = mat.numberDensityEffective * (
                mat.totalScatterXSection() + mat.absorbXSection(lamda)
            )

            corr = np.exp(mu * Tbar)

            peak.setIntensity(peak.getIntensity() * corr)
            peak.setSigmaIntensity(peak.getSigmaIntensity() * corr)


class PrunePeaks:
    def __init__(self, peaks, filename=None):
        assert "PeaksWorkspace" in str(type(mtd[peaks]))

        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.remove_non_integrated()
        self.remove_off_centered()

        self.models = ["type I gaussian", "type I lorentzian", "type II"]

        self.spherical_extinction()
        self.extract_info()

        self.n_iter = 7

        V = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert V > 0

        self.V = V

        self.extinction_prune()
        self.workspaces = self.save_extinction()

    def remove_non_integrated(self):
        for peak in mtd[self.peaks]:
            shape = eval(peak.getPeakShape().toJSON())

            if shape["shape"] == "none":
                peak.setSigmaIntensity(peak.getIntensity())

            elif (
                shape["radius0"] == 0
                or shape["radius1"] == 0
                or shape["radius2"] == 0
            ):
                peak.setSigmaIntensity(peak.getIntensity())

    def remove_off_centered(self, relerr=0.05):
        ol = mtd[self.peaks].sample().getOrientedLattice()

        for peak in mtd[self.peaks]:
            h, k, l = peak.getHKL()
            d0 = ol.d(h, k, l)
            if np.abs(peak.getDSpacing() / d0 - 1) > relerr:
                peak.setSigmaIntensity(peak.getIntensity())

    def spherical_extinction(self):
        f1 = {}
        f2 = {}

        directory = os.path.dirname(os.path.abspath(__file__))

        for model in self.models:
            if "gaussian" in model:
                filename = "secondary_extinction_gaussian_sphere.csv"
            elif "lorentzian" in model:
                filename = "secondary_extinction_lorentzian_sphere.csv"
            else:
                filename = "primary_extinction_sphere.csv"

            fname = os.path.join(directory, filename)

            data = np.loadtxt(
                fname, skiprows=1, delimiter=",", usecols=np.arange(91)
            )
            theta = np.loadtxt(fname, delimiter=",", max_rows=1)

            f1[model] = scipy.interpolate.interp1d(
                2 * np.deg2rad(theta),
                data[0],
                kind="linear",
                fill_value="extrapolate",
            )
            f2[model] = scipy.interpolate.interp1d(
                2 * np.deg2rad(theta),
                data[1],
                kind="linear",
                fill_value="extrapolate",
            )

        self.f1 = f1
        self.f2 = f2

    def extinction_xs(self, g, F2, two_theta, lamda, Tbar, V):
        a = 1e-4  # Ang

        xs = a**2 / V**2 * lamda**3 * g / np.sin(two_theta) * Tbar * F2

        return xs

    def extinction_xp(self, r2, F2, lamda, V):
        a = 1e-4  # Ang

        xp = a**2 / V**2 * lamda**2 * r2 * F2

        return xp

    def extinction_xi(self, two_theta, lamda, Tbar, model):
        V = self.V
        if model == "type II":
            xi = self.extinction_xp(1, 1, lamda, V)
        else:
            xi = self.extinction_xs(1, 1, two_theta, lamda, Tbar, V)
        return xi

    def extinction_correction(self, param, F2, two_theta, lamda, Tbar, model):
        c1, c2 = self.f1[model](two_theta), self.f2[model](two_theta)

        V = self.V

        if model == "type II":
            xp = self.extinction_xp(param, F2, lamda, V)
            yp = 1 / (1 + c1 * xp**c2)
            return yp
        else:
            xs = self.extinction_xs(param, F2, two_theta, lamda, Tbar, V)
            ys = 1 / (1 + c1 * xs**c2)
            return ys

    def weighted_median(self, values, weights):
        sorted_indices = np.argsort(values)
        values_sorted = values[sorted_indices]
        weights_sorted = weights[sorted_indices]
        cumulative_weights = np.cumsum(weights_sorted)
        midpoint = cumulative_weights[-1] / 2
        return values_sorted[np.searchsorted(cumulative_weights, midpoint)]

    def scale_factor(
        self, y_hat_val, y_val, e_val, iterations=3, threshold=1.486
    ):
        mask = y_hat_val != 0
        y_hat, y, e = (
            y_hat_val[mask].copy(),
            y_val[mask].copy(),
            e_val[mask].copy(),
        )

        for iteration in range(iterations):
            z = y / y_hat
            w = np.abs(y_hat) / e
            c = self.weighted_median(z, w)

            residuals = (c * y_hat - y) / e

            mad = np.median(np.abs(residuals - np.median(residuals)))
            mask = residuals <= threshold * mad

            y_hat, y, e = y_hat[mask], y[mask], e[mask]

            if len(y) == 0:
                break

        return c

    def cost(self, x, model):
        wr = []

        for key in self.peaks_dict.keys():
            I, sig, lamda, two_theta, Tbar, k = self.peaks_dict[key]
            diff = self.residual(x, I, sig, two_theta, lamda, Tbar, k, model)
            wr += diff

        wr = np.array(wr)

        return wr[np.isfinite(wr)]

    def residual(self, x, I, sig, two_theta, lamda, Tbar, k, model):
        p = x[0]

        I0 = np.copy(k)

        y = self.extinction_correction(p, I0, two_theta, lamda, Tbar, model)

        s = self.scale_factor(y, I, sig)

        return ((s * y - I) / sig).tolist()

    def extract_info(self):
        peaks_dict = {}

        for peak in mtd[self.peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            key = (h, k, l, m, n, p)
            key_neg = tuple(-val for val in key)

            key = max(key, key_neg)

            lamda = peak.getWavelength()
            two_theta = peak.getScattering()

            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()
            Tbar = peak.getAbsorptionWeightedPathLength() * 1e8  # Angstrom

            items = peaks_dict.get(key)

            if items is None:
                items = [], [], [], [], []

            items[0].append(I)
            items[1].append(sig)
            items[2].append(lamda)
            items[3].append(two_theta)
            items[4].append(Tbar)

            peaks_dict[key] = items

        for key in peaks_dict.keys():
            items = peaks_dict.get(key)
            items = [np.array(item) for item in items]
            peaks_dict[key] = [*items, np.nanmedian(items[0])]

        self.peaks_dict = peaks_dict

    def process_model(self, model):
        app = "_{}.pdf".format(model).replace(" ", "_")

        filename = os.path.splitext(self.filename)[0] + app

        with PdfPages(filename) as pdf:
            I_max, xi = 0, []
            for key, value in self.peaks_dict.items():
                I = np.nanmedian(value[0])
                if I > I_max:
                    I_max = I
                self.peaks_dict[key][-1] = I
                I, sig, lamda, two_theta, Tbar, k = value
                xi += self.extinction_xi(
                    two_theta, lamda, Tbar, model
                ).tolist()

            fit_dict = {}

            fig, ax = plt.subplots(1, 1, layout="constrained")

            chi2dof = []

            for i_iter in range(self.n_iter):
                sol = scipy.optimize.least_squares(
                    self.cost,
                    x0=(10 / I_max / np.median(xi)),
                    bounds=([0], [np.inf]),
                    args=(model,),
                    loss="soft_l1",
                )

                scales, I0s = [], []

                chi2dof.append(
                    np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
                )

                for key, value in self.peaks_dict.items():
                    I, sig, lamda, two_theta, Tbar, k = value

                    param = sol.x[0]

                    I0 = np.copy(k)

                    y = self.extinction_correction(
                        param, I0, two_theta, lamda, Tbar, model
                    )

                    s = self.scale_factor(y, I, sig)

                    self.peaks_dict[key][-1] = s

                    scales.append(s)
                    I0s.append(I0)

                    fit_dict[key] = s * y, s

                scales = np.array(scales)
                I0s = np.array(I0s)

                mask = (I0s > 0) & (scales > 0)

                ax.plot(scales[mask], I0s[mask], ".")
            ax.minorticks_on()
            ax.set_xlabel("scale $s$")
            ax.set_ylabel("$I_0$")
            pdf.savefig()

            fig, ax = plt.subplots(1, 1, layout="constrained")
            ax.errorbar(np.arange(len(chi2dof)) + 1, chi2dof, fmt="-o")
            ax.minorticks_on()
            ax.set_xlabel("Interation #")
            ax.set_ylabel("$\chi^2$")
            # ax.set_yscale('log')
            pdf.savefig()

        familiy_dict = {}

        for key in self.peaks_dict.keys():
            I, sig, lamda, two_theta, Tbar, k = self.peaks_dict[key]

            wl = np.linspace(0, np.max(lamda), 200)

            slope, intercept = scipy.stats.siegelslopes(Tbar, lamda)

            tbar = slope * wl + intercept

            d = np.mean(lamda / (2 * np.sin(0.5 * two_theta)))

            tt = 2 * np.abs(np.arcsin(wl / (2 * d)))

            param = sol.x[0]

            y = self.extinction_correction(param, k, tt, wl, tbar, model)

            familiy_dict[key] = wl, k * y

        return model, fit_dict, sol.x[0], familiy_dict

    def extinction_prune(self):
        self.filter_dict = {}
        self.parameter_dict = {}
        self.model_dict = {}
        self.lamda_max = np.max(mtd[self.peaks].column("Wavelength"))

        with Pool(processes=3) as pool:
            results = pool.map(self.process_model, self.models)

        for model, fit_dict, param, familiy_dict in results:
            self.filter_dict[model] = fit_dict
            self.parameter_dict[model] = param
            self.model_dict[model] = familiy_dict

    def create_figure(self, key, relerr=0.1):
        label = "_({},{},{},{},{},{}).pdf".format(*key)
        filename = os.path.splitext(self.filename)[0] + label

        I, sig, lamda, two_theta, Tbar, c = self.peaks_dict[key]

        sort = np.argsort(lamda)
        I, sig, lamda = I[sort].copy(), sig[sort].copy(), lamda[sort].copy()

        fig, ax = plt.subplots(
            3, 1, sharex=True, sharey=True, layout="constrained"
        )

        for i, model in enumerate(self.models):
            I_fit = self.filter_dict[model][key][0][sort]

            mask = (I_fit - I) / I_fit < relerr

            ax[i].errorbar(
                lamda[mask], I[mask], sig[mask], fmt="o", color="C0", zorder=2
            )
            ax[i].errorbar(
                lamda[~mask],
                I[~mask],
                sig[~mask],
                fmt="s",
                color="C2",
                zorder=2,
            )

            lamda_fit, I_fit = self.model_dict[model][key]

            ax[i].plot(
                lamda_fit,
                I_fit,
                color="C1",
                lw=1,
                zorder=0,
                label="{}".format(model),
            )

            ax[i].legend(shadow=True)
            ax[i].minorticks_on()
            ax[i].set_xlim(0, self.lamda_max)
            ax[i].set_ylabel("$I$ [arb. unit]")

        ax[0].set_title(str(key))
        ax[2].set_xlabel("$\lambda$ [$\AA$]")

        fig.savefig(filename)
        plt.close(fig)

        return filename

    def save_extinction(self, relerr=0.1):
        filename = os.path.splitext(self.filename)[0] + "_ext.pdf"

        with Pool() as pool:
            pdf_files = pool.map(self.create_figure, self.peaks_dict.keys())

        merger = PdfMerger()
        for pdf_file in pdf_files:
            with open(pdf_file, "rb") as f:
                merger.append(f)
        with open(filename, "wb") as f:
            merger.write(f)
        merger.close()

        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                os.remove(pdf_file)

        workspaces = []

        for model in self.models:
            peaks = model.replace(" ", "_")
            workspaces.append(peaks)

            CloneWorkspace(InputWorkspace=self.peaks, OutputWorkspace=peaks)

            for peak in mtd[peaks]:
                h, k, l = [int(val) for val in peak.getIntHKL()]
                m, n, p = [int(val) for val in peak.getIntMNP()]

                key = (h, k, l, m, n, p)
                key_neg = tuple(-val for val in key)

                key = max(key, key_neg)

                I, sig, lamda, two_theta, A, Tbar = self.peaks_dict[key]

                param = self.parameter_dict[model]
                s = self.filter_dict[model][key][1]

                I = peak.getIntensity()
                # sig = peak.getSigmaIntensity()
                two_theta = peak.getScattering()
                lamda = peak.getWavelength()
                Tbar = peak.getAbsorptionWeightedPathLength() * 1e8  # Ang

                y = self.extinction_correction(
                    param, s, two_theta, lamda, Tbar, model
                )

                I_fit = s * y

                if (I_fit - I) / I_fit > relerr:
                    peak.setSigmaIntensity(I)

        return workspaces


class Peaks:
    def __init__(self, peaks, filename, scale=None, point_group=None):
        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        if scale is not None:
            assert scale > 0

        self.scale = scale

        if point_group is not None:
            assert point_group in point_group_dict.keys()
            point_groups = [point_group]
        else:
            point_groups = list(point_group_dict.keys())

        self.point_groups = point_groups

    def rescale_intensities(self):
        scale = 1 if self.scale is None else self.scale
        if mtd[self.peaks].getNumberPeaks() > 1 and self.scale is None:
            I_max = max(mtd[self.peaks].column("Intens"))
            if I_max > 0:
                scale = 1e4 / I_max
            self.scale = scale

        _, indices = np.unique(mtd[self.peaks].column(0), return_inverse=True)

        for i, peak in zip(indices.tolist(), mtd[self.peaks]):
            peak.setIntensity(scale * peak.getIntensity())
            peak.setSigmaIntensity(scale * peak.getSigmaIntensity())
            peak.setRunNumber(1)

        filename = os.path.splitext(self.filename)[0] + "_scale.txt"
        with open(filename, "w") as f:
            f.write("{:.4e}".format(scale))

    def load_peaks(self):
        LoadNexus(Filename=self.filename, OutputWorkspace=self.peaks)

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
        )

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="Intens",
            SortAscending=False,
        )

        self.rescale_intensities()

    def save_peaks(self, name=None):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        filename = os.path.splitext(self.filename)[0] + app

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="Signal/Noise",
            FilterValue=3,
            Operator=">",
        )

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="PeakNumber",
            SortAscending=True,
            OutputWorkspace=peaks,
        )

        SaveHKL(
            InputWorkspace=peaks,
            Filename=filename + ".hkl",
            DirectionCosines=True,
        )

        SaveIsawUB(InputWorkspace=peaks, Filename=filename + ".mat")

        self.resort_hkl(filename + ".hkl")

        self.calculate_statistics(peaks, filename + "_symm.txt")

    def calculate_statistics(self, name, filename):
        point_groups, R_merge = [], []
        for point_group in self.point_groups:
            StatisticsOfPeaksWorkspace(
                InputWorkspace=name,
                PointGroup=point_group_dict[point_group],
                OutputWorkspace="stats",
                EquivalentIntensities="Median",
                SigmaCritical=3,
                WeightedZScore=True,
            )

            R_merge.append(mtd["StatisticsTable"].toDict()["Rmerge"][0])
            point_groups.append(point_group)

        i = np.argmin(R_merge)
        point_group = point_groups[i]

        StatisticsOfPeaksWorkspace(
            InputWorkspace=name,
            PointGroup=point_group_dict[point_group],
            OutputWorkspace="stats",
            EquivalentIntensities="Median",
            SigmaCritical=3,
            WeightedZScore=True,
        )
        self.point_group = [point_group]

        ws = mtd["StatisticsTable"]

        column_names = ws.getColumnNames()
        col_widths = [max(len(str(name)), 8) for name in column_names]

        cols = [
            " ".join(
                name.ljust(col_widths[i])
                for i, name in enumerate(column_names)
            )
        ]

        for i in range(ws.rowCount()):
            row_values = []
            for j, val in enumerate(ws.row(i).values()):
                if isinstance(val, float):
                    val = "{:.2f}".format(val)
                row_values.append(str(val).ljust(col_widths[j]))

            cols.append(" ".join(row_values))

        table = "\n".join(cols)

        with open(filename, "w") as f:
            f.write("{}\n".format(point_group))
            f.write(table)

    def resort_hkl(self, filename):
        ol = mtd[self.peaks].sample().getOrientedLattice()

        UB = ol.getUB()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        max_order = ol.getMaxOrder()

        hkl_widths = [4, 4, 4]
        info_widths = [8, 8, 4, 8, 8, 9, 9, 9, 9, 9, 9, 6, 7, 7, 4, 9, 8, 7, 7]

        if max_order > 0:
            hkl_widths += hkl_widths

        col_widths = hkl_widths + info_widths

        h, k, l, m, n, p = [], [], [], [], [], []

        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                start = 0
                columns = []
                for width in col_widths:
                    columns.append(line[start : start + width].strip())
                    start += width
                h.append(columns[0])
                k.append(columns[1])
                l.append(columns[2])
                m.append(columns[4] if max_order > 0 else 0)
                n.append(columns[5] if max_order > 0 else 0)
                p.append(columns[6] if max_order > 0 else 0)

        h = np.array(h).astype(int)
        k = np.array(k).astype(int)
        l = np.array(l).astype(int)

        m = np.array(m).astype(int)
        n = np.array(n).astype(int)
        p = np.array(p).astype(int)

        mod_HKL = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        hkl = np.stack([h, k, l]) + np.einsum("ij,jm->im", mod_HKL, [m, n, p])

        s = np.linalg.norm(np.einsum("ij,jm->im", UB, hkl), axis=0)

        hkls = np.round(np.column_stack([*hkl, s]) * 1000, 1).astype(int)
        sort = np.lexsort(hkls.T).tolist()
        with open(filename, "w") as f:
            for i in sort[1:]:
                f.write(lines[i])
            f.write(lines[sort[0]])


def main():
    parser = argparse.ArgumentParser(description="Corrections for integration")

    parser.add_argument(
        "filename",
        type=str,
        help="Peaks Workspace",
    )

    parser.add_argument(
        "-w", "--wobble", action="store_false", help="Off-centering correction"
    )

    parser.add_argument(
        "-f",
        "--formula",
        type=str,
        default="Yb3-Al5-O12",
        help="Chemical formula",
    )

    parser.add_argument(
        "-z",
        "--zparameter",
        type=float,
        default="8",
        help="Number of formula units",
    )

    parser.add_argument(
        "-g",
        "--pointgroup",
        type=str,
        default=None,
        help="Point group symmetry",
    )

    parser.add_argument(
        "-u",
        "--uvector",
        nargs="+",
        type=float,
        default=[0, 0, 1],
        help="Miller indices along beam",
    )

    parser.add_argument(
        "-v",
        "--vvector",
        nargs="+",
        type=float,
        default=[1, 0, 0],
        help="Miller indices in plane",
    )

    parser.add_argument(
        "-s",
        "--shape",
        type=str,
        default="sphere",
        help="Sample shape sphere, cylinder, plate",
    )

    parser.add_argument(
        "-p",
        "--parameters",
        nargs="+",
        type=float,
        default=[0],
        help="Length (diameter), height, width in millimeters",
    )

    parser.add_argument(
        "-c", "--scale", type=float, default=None, help="Scale factor"
    )

    args = parser.parse_args()

    peaks = Peaks("peaks", args.filename, args.scale, args.pointgroup)
    peaks.load_peaks()

    if (np.array(args.parameters) > 0).all():
        AbsorptionCorrection(
            "peaks",
            args.formula,
            args.zparameter,
            u_vector=args.uvector,
            v_vector=args.vvector,
            params=args.parameters,
            shape=args.shape,
            filename=args.filename,
        )

    peaks.save_peaks()

    prune = PrunePeaks("peaks", filename=args.filename)

    for workspace in prune.workspaces:
        peaks.save_peaks(workspace)


if __name__ == "__main__":
    main()
