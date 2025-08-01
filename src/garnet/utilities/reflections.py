import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

from mantid.simpleapi import (
    LoadNexus,
    FilterPeaks,
    SortPeaksWorkspace,
    CombinePeaksWorkspaces,
    StatisticsOfPeaksWorkspace,
    CreatePeaksWorkspace,
    SaveHKL,
    SaveReflections,
    SaveIsawUB,
    LoadIsawUB,
    LoadIsawSpectrum,
    CloneWorkspace,
    CopySample,
    SetGoniometer,
    SetSample,
    AddAbsorptionWeightedPathLengths,
    mtd,
)

import re

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(13)

import scipy.optimize
import scipy.interpolate

from scipy.spatial.transform import Rotation

# from statsmodels.nonparametric.smoothers_lowess import lowess

from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
)

from mantid.kernel import V3D

from mantid import config

config["Q.convention"] = "Crystallography"

from matplotlib import colormaps

from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
from PyPDF2 import PdfMerger

import argparse

from garnet.reduction.ub import Optimization

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


class WobbleCorrection:
    def __init__(
        self,
        peaks,
        filename=None,
    ):
        assert "PeaksWorkspace" in str(type(mtd[peaks]))

        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.extract_info()
        self.refine_centering()

    def extract_info(self):
        peak_dict = {}

        ol = mtd[self.peaks].sample().getOrientedLattice()
        mod_HKL = ol.getModHKL().copy()

        for peak in mtd[self.peaks]:
            h, k, l = peak.getIntHKL()
            m, n, p = peak.getIntMNP()

            hkl = np.array([h, k, l]) + np.dot(mod_HKL, [m, n, p])

            key = tuple(np.round(hkl, 3).tolist())
            key_neg = tuple(-val for val in key)

            key = max(key, key_neg)

            R = peak.getGoniometerMatrix()
            I = peak.getIntensity()
            sigma = peak.getSigmaIntensity()
            lamda = peak.getWavelength()

            items = peak_dict.get(key)

            if items is None:
                items = [], [], [], []
            items[0].append(I)
            items[1].append(sigma)
            items[2].append(lamda)
            items[3].append(R)
            peak_dict[key] = items

        for key in peak_dict.keys():
            items = peak_dict[key]
            peak_dict[key] = [np.array(item) for item in items]

        self.peak_dict = peak_dict

    def scale_model(self, coeffs, lamda, R):
        b, c, rx, rz = coeffs

        x, y, z = (
            np.einsum("kij,j->ik", R, [rx, 0, rz])
            + np.array([c, 0, 0])[:, np.newaxis]
        )

        return np.exp(-0.5 * x**2 / (1 + b * lamda) ** 2)

    def cost(self, coeffs, sigma=0.1):
        diff = []

        for key in self.peak_dict.keys():
            I, sig, lamda, R = self.peak_dict[key]

            s = self.scale_model(coeffs, lamda, R)

            y = np.divide(I, s)

            i, j = np.triu_indices(len(lamda), 1)

            d = np.abs(lamda[i] - lamda[j])

            w = np.exp(-((d / sigma) ** 2))

            y_bar = 0.5 * (y[i] + y[j])

            diff += (w * (y[i] / y_bar - 1)).flatten().tolist()
            diff += (w * (y[j] / y_bar - 1)).flatten().tolist()

        # diff += list(coeffs)
        return np.nansum(np.square(diff))

    def refine_centering(self):
        sol = scipy.optimize.differential_evolution(
            self.cost,
            x0=[0, 0, 0, 0.0],
            bounds=np.array([(-1, -1, -1, 0), (1, 1, 1, 1)]).T,
            # verbose=2,
            workers=-1,
        )

        # assert sol.status >= 1

        coeffs = sol.x

        b, c, rx, rz = coeffs

        lines = [
            "spread: {:.4f} 1/A^3\n".format(b),
            "center: {:.4f} \n".format(c),
            "in-plane displacement: {:.4f} \n".format(rx),
            "beam displacement: {:.4f} \n".format(rz),
        ]

        for line in lines:
            print(line)

        if self.filename is not None:
            filename = os.path.splitext(self.filename)[0] + "_off.txt"

            with open(filename, "w") as f:
                for line in lines:
                    f.write(line)

        omega = np.linspace(0, 360, 361)

        R = [
            Rotation.from_euler("y", val, degrees=True).as_matrix()
            for val in omega
        ]

        lamda = mtd[self.peaks].column("Wavelength")

        fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

        lamda_min, lamda_max = np.min(lamda), np.max(lamda)

        norm = Normalize(vmin=lamda_min, vmax=lamda_max)
        cmap = colormaps["turbo"]

        fig, ax = plt.subplots(1, 1)
        for lamda in np.linspace(lamda_min, lamda_max, 5):
            s = self.scale_model(coeffs, lamda, R)
            ax.plot(omega, s, color=cmap(norm(lamda)))
        ax.set_xlabel("Rotation angle")
        ax.set_ylabel("Scale factor")
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(fmt_str_form)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, label="$\lambda$ [$\AA$]")
        cb.minorticks_on()

        if self.filename is not None:
            filename = os.path.splitext(self.filename)[0] + "_off.pdf"
            fig.savefig(filename)

        for peak in mtd[self.peaks]:
            R = peak.getGoniometerMatrix()
            lamda = peak.getWavelength()
            s = self.scale_model(coeffs, lamda, [R])[0]
            peak.setIntensity(1 / s * peak.getIntensity())
            peak.setSigmaIntensity(1 / s * peak.getSigmaIntensity())


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

        if shape == "plate":
            assert len(params) == 3
        elif shape == "cylinder":
            assert len(params) == 2
        # else:
        #     assert len(params) == 1

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

        self.apply_correction()

        with PdfPages(filename) as pdf:
            for i, (R, run) in enumerate(zip(self.Rs, self.runs)):
                FilterPeaks(
                    InputWorkspace=peaks,
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
                CloneWorkspace(InputWorkspace="_tmp", OutputWorkspace=peaks)
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace=peaks,
                    RHSWorkspace="_tmp",
                    OutputWorkspace=peaks,
                )

        CloneWorkspace(InputWorkspace=peaks, OutputWorkspace=self.peaks)

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

        self.models = ["type I gaussian", "type I lorentzian", "type II"]

        self.spherical_extinction()
        self.extract_info()

        self.n_iter = 7

        V = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert V > 0

        self.V = V

        lamda = np.array(mtd[self.peaks].column("Wavelength"))

        kde = scipy.stats.gaussian_kde(lamda)

        x = np.linspace(lamda.min(), lamda.max(), 1000)

        self.kde = scipy.interpolate.make_interp_spline(x, kde(x), k=1)

        self.extinction_prune()
        self.workspaces, self.parameters = self.save_extinction()

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
        a = 1e-5  # Ang

        xs = a**2 / V**2 * lamda**3 * g / np.sin(two_theta) * Tbar * F2

        return xs

    def extinction_xp(self, r2, F2, lamda, V):
        a = 1e-5  # Ang

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

    def scale_factor(self, y_hat_val, y_val, lamda):
        ratios = y_val / y_hat_val
        weights = self.kde(lamda)
        c = self.weighted_median(ratios, weights)

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

        s = self.scale_factor(y, I, lamda)

        return ((s * y - I) * I / sig).tolist()

    def extract_info(self):
        peaks_dict = {}

        ol = mtd[self.peaks].sample().getOrientedLattice()
        mod_HKL = ol.getModHKL().copy()

        for peak in mtd[self.peaks]:
            h, k, l = peak.getIntHKL()
            m, n, p = peak.getIntMNP()

            hkl = np.array([h, k, l]) + np.dot(mod_HKL, [m, n, p])

            key = tuple(np.round(hkl, 3).tolist())
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

                    s = self.scale_factor(y, I, lamda)

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

    def outlier_detection(self, key, model, c1=0.05, c4=1):
        I, sig, lamda, two_theta, A, Tbar = self.peaks_dict[key]
        I_fit = self.filter_dict[model][key][0]

        n = I_fit.size

        sig_ext = np.median(sig)
        sig_int = (
            np.median(np.abs(I - I_fit) * np.sqrt(n / (n - 1)))
            if n > 1
            else np.inf
        )

        zcrit = np.sqrt(scipy.stats.chi2.ppf(1 - 1 / (2 * n), df=1))

        t = (
            np.max(
                [c1 * np.median(I), c4 * zcrit * np.max([sig_ext, sig_int])]
            )
            if n > 1
            else -np.inf
        )

        mask = np.abs(I_fit - I) < t

        return I_fit, I, sig, lamda, Tbar, t, mask

    def create_figure(self, key):
        label = "_({} {} {}).pdf".format(*key)
        filename = os.path.splitext(self.filename)[0] + label

        I, sig, lamda, two_theta, Tbar, c = self.peaks_dict[key]

        sort = np.argsort(lamda)
        I, sig, lamda = I[sort].copy(), sig[sort].copy(), lamda[sort].copy()

        fig, ax = plt.subplots(
            3, 1, sharex=True, sharey=True, layout="constrained"
        )

        for i, model in enumerate(self.models):
            I_fit, I, sig, *_, mask = self.outlier_detection(key, model)

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

    def save_extinction(self, zmax=6):
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
        parameters = []

        for model in self.models:
            peaks = model.replace(" ", "_")
            workspaces.append(peaks)

            CloneWorkspace(InputWorkspace=self.peaks, OutputWorkspace=peaks)

            ol = mtd[peaks].sample().getOrientedLattice()
            mod_HKL = ol.getModHKL().copy()

            outliers = {}
            for key in self.peaks_dict.keys():
                outliers[key] = self.outlier_detection(key, model)

            for key in outliers.keys():
                items = outliers[key]
                items = [np.array(item) for item in items]
                outliers[key] = items

            weights = {}

            for peak in mtd[peaks]:
                h, k, l = [int(val) for val in peak.getIntHKL()]
                m, n, p = [int(val) for val in peak.getIntMNP()]

                hkl = np.array([h, k, l]) + np.dot(mod_HKL, [m, n, p])

                key = tuple(np.round(hkl, 3).tolist())
                key_neg = tuple(-val for val in key)

                key = max(key, key_neg)

                param = self.parameter_dict[model]
                s = self.filter_dict[model][key][1]

                I = peak.getIntensity()
                sig = peak.getSigmaIntensity()
                two_theta = peak.getScattering()
                lamda = peak.getWavelength()
                Tbar = peak.getAbsorptionWeightedPathLength() * 1e8  # Ang

                y = self.extinction_correction(
                    param, s, two_theta, lamda, Tbar, model
                )

                I_fit = s * y

                *_, t, mask = outliers[key]

                diff = np.abs(I - I_fit)

                if diff > t:
                    peak.setSigmaIntensity(I)
                else:
                    peak.setBinCount(diff)
                    key = (h, k, l, m, n, p)
                    items = weights.get(key)
                    if items is None:
                        items = [], [], [], [], []
                    items[0].append(I_fit)
                    items[1].append(I)
                    items[2].append(sig)
                    items[3].append(lamda)
                    items[4].append(Tbar)
                    weights[key] = items

            for key in weights.keys():
                items = weights[key]
                items = [np.array(item) for item in items]
                weights[key] = items

            print(model)

            for peak in mtd[peaks]:
                h, k, l = [int(val) for val in peak.getIntHKL()]
                m, n, p = [int(val) for val in peak.getIntMNP()]

                key = (h, k, l, m, n, p)

                items = weights.get(key)
                if items is not None:
                    if len(items[0]) == 1:
                        print("->", key)
                        peak.setSigmaIntensity(peak.getIntensity())

            outlier = {}

            for key in weights.keys():
                I_fit, I, sig, lamda, Tbar = weights[key]

                N = I_fit.size

                sig_ext = np.median(sig)
                sig_int = (
                    np.median(np.abs(I - I_fit) * np.sqrt(N / (N - 1)))
                    if N > 1
                    else np.inf
                )

                sigma = np.max([sig_ext, sig_int])

                z = (I - I_fit) / sigma

                weight = (1 - (z / zmax) ** 2) ** 2
                weight[np.abs(z) > zmax] = 0.01

                outlier[key] = I_fit, I, sig, lamda, Tbar, weight

            for key in outlier.keys():
                items = outlier.get(key)
                items = [np.array(item) for item in items]
                outlier[key] = items

            parameters.append(outlier)

        return workspaces, parameters


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

        self.max_order = 0
        self.modUB = np.zeros((3, 3))
        self.modHKL = np.zeros((3, 3))

    def refine_UB(self, peaks):
        opt = Optimization(peaks)

        ol = mtd[peaks].sample().getOrientedLattice()

        a, b, c = ol.a(), ol.b(), ol.c()
        alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

        if np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90):
            cell = "Cubic"
        elif np.allclose([a, b], c) and np.allclose([alpha, beta], gamma):
            cell = "Rhombohedral"
        elif np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90):
            cell = "Tetragonal"
        elif (
            np.isclose(a, b)
            and np.allclose([alpha, beta], 90)
            and np.isclose(gamma, 120)
        ):
            cell = "Hexagonal"
        elif np.allclose([alpha, beta, gamma], 90):
            cell = "Orthorhombic"
        elif np.allclose([alpha, gamma], 90):
            cell = "Monoclinic"
        else:
            cell = "Triclinic"

        opt.optimize_lattice(cell)

    def rescale_intensities(self):
        maximal = 10000.00
        scale = 1 if self.scale is None else self.scale
        if mtd[self.peaks].getNumberPeaks() > 1 and self.scale is None:
            I = np.array(mtd[self.peaks].column("Intens"))
            I0 = np.nanpercentile(I, 99)
            scale = maximal / I0
            self.scale = scale

        # _, indices = np.unique(mtd[self.peaks].column(0), return_inverse=True)

        indices = np.arange(mtd[self.peaks].getNumberPeaks())
        for i, peak in zip(indices.tolist(), mtd[self.peaks]):
            peak.setIntensity(scale * peak.getIntensity())
            peak.setSigmaIntensity(scale * peak.getSigmaIntensity())
            if peak.getIntensity() > 10 * maximal:
                peak.setSigmaIntensity(peak.getIntensity())
            peak.setPeakNumber(peak.getRunNumber())
            peak.setRunNumber(1)

        filename = os.path.splitext(self.filename)[0] + "_scale.txt"
        with open(filename, "w") as f:
            f.write("{:.4e}".format(scale))

    def remove_off_centered(self):
        aluminum = CrystalStructure(
            "4.05 4.05 4.05", "F m -3 m", "Al 0 0 0 1.0 0.005"
        )

        copper = CrystalStructure(
            "3.61 3.61 3.61", "F m -3 m", "Cu 0 0 0 1.0 0.005"
        )

        generator_al = ReflectionGenerator(aluminum)
        generator_cu = ReflectionGenerator(copper)

        d = np.array(mtd[self.peaks].column("DSpacing"))

        d_min = np.nanmin(d)
        d_max = np.nanmax(d)

        hkls_al = generator_al.getUniqueHKLsUsingFilter(
            d_min, d_max, ReflectionConditionFilter.StructureFactor
        )

        hkls_cu = generator_cu.getUniqueHKLsUsingFilter(
            d_min, d_max, ReflectionConditionFilter.StructureFactor
        )

        d_al = np.unique(generator_al.getDValues(hkls_al))
        d_cu = np.unique(generator_cu.getDValues(hkls_cu))

        ol = mtd[self.peaks].sample().getOrientedLattice()

        Q_vol = []
        powder_err = []
        peak_err = []
        Q0_mod = []

        for peak in mtd[self.peaks]:
            h, k, l = peak.getHKL()
            d0 = ol.d(h, k, l)
            powder_err.append(peak.getDSpacing() / d0 - 1)
            Q0 = 2 * np.pi * ol.getUB() @ np.array([h, k, l])
            peak_err.append(peak.getQSampleFrame() - Q0)
            Q0_mod.append(2 * np.pi / d0)

            shape = peak.getPeakShape()
            if shape.shapeName() == "ellipsoid":
                ellipsoid = eval(shape.toJSON())

                r0 = ellipsoid["radius0"]
                r1 = ellipsoid["radius1"]
                r2 = ellipsoid["radius2"]

            else:
                r0 = r1 = r2 = 1e-6

            vol = 4 / 3 * np.pi * r0 * r1 * r2

            Q_vol.append(vol)

        Q0_mod = np.array(Q0_mod)
        powder_err = np.array(powder_err)
        peak_err = np.array(peak_err) / Q0_mod[:, np.newaxis]
        Q_vol = np.array(Q_vol)

        powder_Q1, powder_Q3 = np.nanpercentile(powder_err, [25, 75])
        peak_Q1, peak_Q3 = np.nanpercentile(peak_err, [25, 75], axis=0)

        powder_IQR = powder_Q3 - powder_Q1
        peak_IQR = peak_Q3 - peak_Q1

        powder_min = powder_Q1 - 3 * powder_IQR
        powder_max = powder_Q3 + 3 * powder_IQR

        peak_min = peak_Q1 - 3 * peak_IQR
        peak_max = peak_Q3 + 3 * peak_IQR

        vol_Q1, vol_Q3 = np.nanpercentile(Q_vol, [25, 75])
        vol_IQR = vol_Q3 - vol_Q1

        vol_cut = vol_Q3 + 1.5 * vol_IQR

        filename = os.path.splitext(self.filename)[0]

        fig, ax = plt.subplots(5, 1, layout="constrained")
        ax[4].set_xlabel("$|Q|$ [$\AA^{-1}$]")
        ax[0].set_ylabel("$d/d_0-1$")
        ax[1].set_ylabel("$\Delta{Q_1}/|Q|$")
        ax[2].set_ylabel("$\Delta{Q_2}/|Q|$")
        ax[3].set_ylabel("$\Delta{Q_3}/|Q|$")
        ax[4].set_ylabel("$V$")
        ax[0].minorticks_on()
        ax[1].minorticks_on()
        ax[2].minorticks_on()
        ax[3].minorticks_on()
        ax[4].minorticks_on()
        ax[0].plot(Q0_mod, powder_err, ".", color="C0")
        ax[1].plot(Q0_mod, peak_err[:, 0], ".", color="C1", rasterized=True)
        ax[2].plot(Q0_mod, peak_err[:, 1], ".", color="C2", rasterized=True)
        ax[3].plot(Q0_mod, peak_err[:, 2], ".", color="C3", rasterized=True)
        ax[4].plot(Q0_mod, Q_vol, ".", color="C4", rasterized=True)
        ax[0].axhline(powder_min, color="k", linestyle="--", linewidth=1)
        ax[0].axhline(powder_max, color="k", linestyle="--", linewidth=1)
        ax[1].axhline(peak_min[0], color="k", linestyle="--", linewidth=1)
        ax[1].axhline(peak_max[0], color="k", linestyle="--", linewidth=1)
        ax[2].axhline(peak_min[1], color="k", linestyle="--", linewidth=1)
        ax[2].axhline(peak_max[1], color="k", linestyle="--", linewidth=1)
        ax[3].axhline(peak_min[2], color="k", linestyle="--", linewidth=1)
        ax[3].axhline(peak_max[2], color="k", linestyle="--", linewidth=1)
        ax[4].axhline(vol_cut, color="k", linestyle="--", linewidth=1)

        for i in range(5):
            for d in d_al:
                ax[i].axvline(
                    2 * np.pi / d, color="k", linestyle=":", linewidth=1
                )

            for d in d_cu:
                ax[i].axvline(
                    2 * np.pi / d, color="k", linestyle=":", linewidth=1
                )

        fig.savefig(filename + "_cont.pdf")

        for i, peak in enumerate(mtd[self.peaks]):
            powder = powder_err[i] > powder_max or powder_err[i] < powder_min
            contamination = (peak_err[i] > peak_max) | (peak_err[i] < peak_min)
            background = Q_vol[i] > vol_cut
            if contamination.any() or powder.any() or background:
                peak.setSigmaIntensity(peak.getIntensity())

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

    def load_spectrum(self, filename, instrument):
        LoadIsawSpectrum(
            SpectraFile=filename,
            OutputWorkspace="spectrum",
            InstrumentName=instrument,
        )

    def load_peaks(self):
        LoadNexus(Filename=self.filename, OutputWorkspace=self.peaks)

        shape_dict = {}

        s = np.array(
            [
                [sx, sy, sz]
                for sx in [-1, 1]
                for sy in [-1, 1]
                for sz in [-1, 1]
            ]
        )

        for peak in mtd[self.peaks]:
            c = np.array(peak.getQSampleFrame())

            shape = peak.getPeakShape()

            if shape.shapeName() == "ellipsoid":
                ellipsoid = eval(shape.toJSON())

                v0 = [float(val) for val in ellipsoid["direction0"].split(" ")]
                v1 = [float(val) for val in ellipsoid["direction1"].split(" ")]
                v2 = [float(val) for val in ellipsoid["direction2"].split(" ")]

                r0 = ellipsoid["radius0"]
                r1 = ellipsoid["radius1"]
                r2 = ellipsoid["radius2"]

            else:
                r0 = r1 = r2 = 1e-6
                v0, v1, v2 = np.eye(3).tolist()

            run = peak.getRunNumber()
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]
            key = (run, h, k, l, m, n, p)

            r = np.array([r0, r1, r2])

            U = np.column_stack([v0, v1, v2])
            V_inv = np.diag(1 / r)
            S_inv = np.dot(np.dot(U, V_inv), U.T)

            local_corners = s * r
            global_corners = c + local_corners @ U.T
            min_corner = np.min(global_corners, axis=0)
            max_corner = np.max(global_corners, axis=0)

            shape_dict[key] = c, S_inv, min_corner, max_corner

        self.shape_dict = shape_dict

        ub_file = self.filename.replace(".nxs", ".mat")

        if os.path.exists(ub_file):
            LoadIsawUB(Filename=ub_file, InputWorkspace=self.peaks)

        self.remove_off_centered()
        self.remove_non_integrated()

        run_info = mtd[self.peaks].run()
        run_keys = run_info.keys()

        keys = ["h", "k", "l", "m", "n", "p", "run"]
        vals = ["intens", "sig", "vol"]

        info_dict = {}
        norm_dict = {}

        items = keys + vals

        log_info = np.all(
            ["peaks_{}".format(item) in run_keys for item in items]
        )

        if log_info:
            h = run_info.getLogData("peaks_h").value
            k = run_info.getLogData("peaks_k").value
            l = run_info.getLogData("peaks_l").value
            m = run_info.getLogData("peaks_m").value
            n = run_info.getLogData("peaks_n").value
            p = run_info.getLogData("peaks_p").value
            run = run_info.getLogData("peaks_run").value

            N = run_info.getLogData("peaks_voxels").value
            vol = run_info.getLogData("peaks_vol").value
            pk_data = run_info.getLogData("peaks_pk_data").value
            pk_norm = run_info.getLogData("peaks_pk_norm").value
            bkg_data = run_info.getLogData("peaks_bkg_data").value
            bkg_norm = run_info.getLogData("peaks_bkg_norm").value

            intens = run_info.getLogData("peaks_intens").value
            sig = run_info.getLogData("peaks_sig").value

            for i in range(len(run)):
                key = (run[i], h[i], k[i], l[i], m[i], n[i], p[i])
                vals = (
                    N[i],
                    vol[i],
                    pk_data[i],
                    pk_norm[i],
                    bkg_data[i],
                    bkg_norm[i],
                )
                info_dict[key] = vals
                norm_dict[key] = (intens[i], sig[i])

        filename = os.path.splitext(self.filename)[0]

        self.info_dict = info_dict
        self.norm_dict = norm_dict

        x, y = [], []

        for peak in mtd[self.peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getRunNumber())
            key = (run, h, k, l, m, n, p)
            N, vol, pk_data, pk_norm, bkg_data, bkg_norm = self.info_dict[key]

            lamda = peak.getWavelength()
            two_theta = peak.getScattering()

            norm = np.log10(bkg_norm / pk_norm)

            Q = 4 * np.pi / lamda * np.sin(0.5 * two_theta)

            x.append(Q)
            y.append(norm)

        filename = os.path.splitext(self.filename)[0]

        x, y = np.array(x), np.array(y)

        ratio_Q1, ratio_Q3 = np.nanpercentile(y, [25, 75])

        ratio_IQR = ratio_Q3 - ratio_Q1

        ratio_min = ratio_Q1 - 1.5 * ratio_IQR
        ratio_max = ratio_Q3 + 1.5 * ratio_IQR

        fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        ax.set_xlabel("$|Q|$ [$\AA^{-1}$]")
        ax.plot(x, y, ".", rasterized=True)
        ax.minorticks_on()
        ax.set_ylabel("Log peak-background norm")
        ax.axhline(ratio_max, color="k", linestyle="--", linewidth=1)
        ax.axhline(ratio_min, color="k", linestyle="--", linewidth=1)
        fig.savefig(filename + "_norm.pdf")

        for peak in mtd[self.peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getRunNumber())
            key = (run, h, k, l, m, n, p)
            N, vol, pk_data, pk_norm, bkg_data, bkg_norm = self.info_dict[key]

            norm = np.log10(bkg_norm / pk_norm)

            if not np.isfinite(norm) or norm < ratio_min or norm > ratio_max:
                peak.setSigmaIntensity(peak.getIntensity())

        lamda = np.array(mtd[self.peaks].column("Wavelength"))

        kde = scipy.stats.gaussian_kde(lamda)

        x = np.linspace(lamda.min(), lamda.max(), 1000)

        pdf = kde(x)

        cdf = scipy.integrate.cumulative_trapezoid(pdf, x, initial=0)
        cdf /= cdf[-1]

        lower_bound = x[np.searchsorted(cdf, 0.01)]
        upper_bound = x[np.searchsorted(cdf, 0.99)]

        filename = os.path.splitext(self.filename)[0]

        fig, ax = plt.subplots(layout="constrained")
        ax.hist(lamda, bins=100, density=True, color="C0")
        ax.set_xlabel("$\lambda$ [$\AA$]")
        ax.minorticks_on()
        ax.plot(x, pdf, color="C1")
        ax.axvline(lower_bound, color="k", linestyle="--", linewidth=1)
        ax.axvline(upper_bound, color="k", linestyle="--", linewidth=1)
        fig.savefig(filename + ".pdf")

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=lower_bound,
            Operator=">",
        )

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=upper_bound,
            Operator="<",
        )

        self.reset_satellite()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=3,
            Operator=">",
        )

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

        # lamda = np.array(mtd[self.peaks].column("Wavelength"))
        # lamda_min = np.min(lamda)
        # lamda_max = np.max(lamda)

        # banks = np.array(mtd[self.peaks].column("BankName"))

        # spectrum_dict = {}

        # for i, peak in enumerate(mtd[self.peaks]):
        #     h, k, l = [int(val) for val in peak.getIntHKL()]
        #     m, n, p = [int(val) for val in peak.getIntMNP()]

        #     run = int(peak.getRunNumber())
        #     key = (run, h, k, l, m, n, p)
        #     raw_intens, raw_sig = self.norm_dict[key]

        #     bank = banks[i]

        #     intens = peak.getIntensity()
        #     sig = peak.getSigmaIntensity()

        #     lamda = peak.getWavelength()
        #     two_theta = peak.getScattering()

        #     L = 0.5 * lamda**4 / np.sin(0.5 * two_theta) ** 2

        #     norm = raw_intens / intens
        #     err = np.abs(norm) * np.sqrt((raw_sig / raw_intens)**2 + (sig / intens)**2)

        #     items = spectrum_dict.get(bank)
        #     if items is None:
        #         items = [], [], [], []
        #     items[0].append(lamda)
        #     items[1].append(norm / L)
        #     items[2].append(err / L)
        #     items[3].append(i)
        #     spectrum_dict[bank] = items

        # with PdfPages(filename + "_spec.pdf") as pdf:

        #     for bank in spectrum_dict.keys():

        #         x, y, e, ind = spectrum_dict[bank]

        #         x, y, e = np.array(x), np.array(y), np.array(e)

        #         ind = np.array(ind)

        #         sort = np.argsort(x)

        #         x, y, e = x[sort], y[sort], e[sort]

        #         ind = ind[sort]

        #         y_hat = lowess(y, x, frac=0.15, delta=0.1, it=10, return_sorted=False)

        #         diff = y - y_hat

        #         spread = 3 * np.nanmedian(np.abs(diff))

        #         mask = np.abs(diff) < spread

        #         fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        #         ax.set_xlabel("$\lambda$ [$\AA$]")
        #         ax.errorbar(x[mask], y[mask], fmt=".", color="C0")
        #         ax.errorbar(x[~mask], y[~mask], fmt="x", color="C0")
        #         ax.plot(x, y_hat, color="C1")
        #         ax.plot(x[mask], diff[mask], ".", color="C2")
        #         ax.plot(x[~mask], diff[~mask], "x", color="C2")
        #         ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        #         ax.axhline(y=-spread, color='k', linestyle=':', linewidth=1)
        #         ax.axhline(y=+spread, color='k', linestyle=':', linewidth=1)
        #         ax.minorticks_on()
        #         ax.set_xlim(lamda_min, lamda_max)
        #         ax.set_ylabel("Spectrum")
        #         ax.set_title(bank)

        #         pdf.savefig(fig, dpi=100, bbox_inches=None)
        #         plt.close(fig)

        #         for i in ind[~mask].tolist():
        #             peak = mtd[self.peaks].getPeak(i)
        #             peak.setSigmaIntensity(peak.getIntensity())

        # FilterPeaks(
        #     InputWorkspace=self.peaks,
        #     OutputWorkspace=self.peaks,
        #     FilterVariable="Signal/Noise",
        #     FilterValue=3,
        #     Operator=">",
        # )

        self.rescale_intensities()

    def merge_intensities(self, name=None, fit_dict=None):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        CreatePeaksWorkspace(
            NumberOfPeaks=0,
            OutputWorkspace=peaks + "_lean",
            OutputType="LeanElasticPeak",
        )

        CopySample(
            InputWorkspace=peaks,
            OutputWorkspace=peaks + "_lean",
            CopyName=False,
            CopyEnvironment=False,
        )

        peaks_lean = mtd[peaks + "_lean"]

        ol = mtd[peaks].sample().getOrientedLattice()
        mod_HKL = ol.getModHKL().copy()

        fit_dict = {}
        ellipsoid_dict = {}

        for peak in mtd[peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getPeakNumber())
            key = (run, h, k, l, m, n, p)

            intens = peak.getIntensity()
            sigma = peak.getSigmaIntensity()
            lamda = peak.getWavelength()
            Tbar = peak.getAbsorptionWeightedPathLength() * 1e8  # Ang

            N, vol, pk_data, pk_norm, bkg_data, bkg_norm = self.info_dict[key]

            key = (h, k, l, m, n, p)

            items = fit_dict.get(key)
            if items is None:
                items = [], [], [], [], [], [], [], [], [], []
            items[0].append(N)
            items[1].append(vol)
            items[2].append(pk_data)
            items[3].append(pk_norm)
            items[4].append(bkg_data)
            items[5].append(bkg_norm)
            items[6].append(lamda)
            items[7].append(Tbar)
            items[8].append(intens)
            items[9].append(sigma)
            fit_dict[key] = items

            key = (run, h, k, l, m, n, p)

            c, S_inv, min_bounds, max_bounds = self.shape_dict[key]

            key = (h, k, l, m, n, p)

            items = ellipsoid_dict.get(key)
            if items is None:
                items = [], [], [], []
            items[0].append(c)
            items[1].append(S_inv)
            items[2].append(min_bounds)
            items[3].append(max_bounds)
            ellipsoid_dict[key] = items

        for key in fit_dict.keys():
            items = fit_dict.get(key)
            items = [np.array(item) for item in items]
            fit_dict[key] = items

        for key in ellipsoid_dict.keys():
            items = ellipsoid_dict.get(key)
            mins = np.min(items[2], axis=0)
            maxs = np.max(items[3], axis=0)
            items = np.array(items[0]), np.array(items[1]), mins, maxs
            ellipsoid_dict[key] = items

        F2, Q, y = [], [], []

        for key in fit_dict.keys():
            h, k, l, m, n, p = key
            hkl = np.array([h, k, l]) + np.dot(mod_HKL, [m, n, p])
            peak = peaks_lean.createPeakHKL(hkl.tolist())
            peak.setIntHKL(V3D(h, k, l))
            peak.setIntMNP(V3D(m, n, p))

            d = peak.getDSpacing()

            items = fit_dict[key]
            (
                N,
                d3x,
                pk_data,
                pk_norm,
                bkg_data,
                bkg_norm,
                lamda,
                Tbar,
                I,
                sigma,
            ) = items

            norm = np.nansum(pk_norm)
            data = np.nansum(pk_data)

            volume = N * d3x

            w = pk_norm / N

            vol = np.nansum(volume * w) / np.nansum(w)

            b = np.nansum(bkg_data) / np.nansum(bkg_norm)
            b_err = np.sqrt(np.nansum(bkg_data)) / np.nansum(bkg_norm)

            wl = np.nansum(lamda * w) / np.nansum(w)
            wpl = np.nansum(Tbar * w) / np.nansum(w)

            intens = self.scale * (data / norm - b) * vol
            sig_ext = self.scale * np.sqrt(data / norm**2 + b_err**2) * vol

            sig_int = np.sqrt(np.nanmean((I - intens) ** 2))

            if sig_int > 0 and sig_ext > 0 and intens > 0:
                Q.append(2 * np.pi / d)
                F2.append(intens)
                y.append(sig_int / sig_ext)

            peak.setIntensity(intens)
            peak.setSigmaIntensity(sig_ext)
            peak.setBinCount(sig_int)
            peak.setWavelength(wl)
            peak.setAbsorptionWeightedPathLength(wpl * 1e-8)
            peaks_lean.addPeak(peak)

        # F2, Q, y = np.array(F2), np.array(Q), np.array(y)

        # A = np.column_stack(
        #     [
        #         np.log(F2) ** 2,
        #         Q**2,
        #         np.log(F2) * Q,
        #         np.log(F2),
        #         Q,
        #         np.ones_like(y),
        #     ]
        # )
        # x, *_ = np.linalg.lstsq(A, y)

        # y_fit = np.dot(A, x)

        filename = os.path.splitext(self.filename)[0] + app + "_merge"

        # vmin, vmax = [np.min(y), np.max(y)] if len(y) > 1 else [0.1, 10]

        # label = "$\sigma_\mathrm{int}/\sigma_\mathrm{ext}$"

        # fig, ax = plt.subplots(1, 2, sharey=True, layout="constrained")
        # ax[0].minorticks_on()
        # ax[1].minorticks_on()
        # ax[0].set_yscale("log")
        # ax[1].set_yscale("log")
        # im = ax[0].scatter(Q, F2, c=y, vmin=vmin, vmax=vmax, norm="log")
        # ax[1].scatter(Q, F2, c=y_fit, vmin=vmin, vmax=vmax, norm="log")
        # ax[0].set_xlabel("$|Q|$ [$\AA^{-1}]$")
        # ax[1].set_xlabel("$|Q|$ [$\AA^{-1}]$")
        # ax[0].set_ylabel("$F^2$ [arb. unit]")
        # cb = fig.colorbar(im, ax=ax, label=label)
        # cb.minorticks_on()
        # fig.savefig(filename + ".pdf")

        # self.x = x

        FilterPeaks(
            InputWorkspace=peaks + "_lean",
            OutputWorkspace=peaks + "_lean",
            FilterVariable="Signal/Noise",
            FilterValue=3,
            Operator=">",
        )

        for col in ["h", "k", "l", "DSpacing"]:
            SortPeaksWorkspace(
                InputWorkspace=peaks + "_lean",
                OutputWorkspace=peaks + "_lean",
                ColumnNameToSortBy=col,
                SortAscending=False,
            )

        SaveReflections(
            InputWorkspace=peaks + "_lean",
            Filename=filename + "_jana.int",
            Format="Jana",
        )

        SaveReflections(
            InputWorkspace=peaks + "_lean",
            Filename=filename + "_fullprof.int",
            Format="Fullprof",
        )

    def reset_satellite(self):
        mod_mnp = []
        mod_hkl = []
        for peak in mtd[self.peaks]:
            hkl = peak.getHKL()
            int_hkl = peak.getIntHKL()
            int_mnp = peak.getIntMNP()
            if int_mnp.norm2() > 0:
                mod_mnp.append(np.array(int_mnp))
                mod_hkl.append(np.array(hkl - int_hkl))

        ol = mtd[self.peaks].sample().getOrientedLattice()

        if len(mod_mnp) > 0:
            mod_vec = np.linalg.pinv(mod_mnp) @ np.array(mod_hkl)

            ol.setModVec1(V3D(*mod_vec[0]))
            ol.setModVec2(V3D(*mod_vec[1]))
            ol.setModVec3(V3D(*mod_vec[2]))

            ol.setModUB(ol.getUB() @ ol.getModHKL())

            max_order = ol.getMaxOrder()

            self.max_order = max_order if max_order > 0 else 1
            self.modUB = ol.getModUB().copy()
            self.modHKL = ol.getModHKL().copy()

            ol.setMaxOrder(self.max_order)

        else:
            self.max_order = 0
            self.modUB = np.zeros((3, 3))
            self.modHKL = np.zeros((3, 3))

            ol.setMaxOrder(0)

            ol.setModVec1(V3D(0, 0, 0))
            ol.setModVec2(V3D(0, 0, 0))
            ol.setModVec3(V3D(0, 0, 0))

            ol.setModUB(self.modUB)

        print(self.max_order)

    def save_peaks(self, name=None, fit_dict=None):
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

        self.merge_intensities(name, fit_dict)

        # for peak in mtd[peaks]:
        # I = peak.getIntensity()
        # Q = 2 * np.pi / peak.getDSpacing()
        # sig = peak.getSigmaIntensity()
        # diff = peak.getBinCount()
        # A = [np.log(I) ** 2, Q**2, np.log(I) * Q, np.log(I), Q, 1]
        # sig_est = np.dot(A, self.x) * sig
        # I if diff >= 2 * sig and diff > 0 else sig
        # peak.setSigmaIntensity(sig)

        _, indices = np.unique(
            mtd[peaks].column("BankName"), return_inverse=True
        )

        for i, peak in zip(indices.tolist(), mtd[peaks]):
            peak.setRunNumber(1)

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
            ApplyAnvredCorrections=False,
            SortBy="RunNumber",
        )

        self.refine_UB(peaks)

        SaveIsawUB(InputWorkspace=peaks, Filename=filename + ".mat")

        self.resort_hkl(peaks, filename + ".hkl")

        self.calculate_statistics(peaks, filename + "_symm.txt")

        if self.max_order > 0:
            nuclear = peaks + "_nuc"
            satellite = peaks + "_sat"

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=nuclear,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator="=",
            )

            nuc_ol = mtd[nuclear].sample().getOrientedLattice()
            nuc_ol.setMaxOrder(0)
            nuc_ol.setModVec1(V3D(0, 0, 0))
            nuc_ol.setModVec2(V3D(0, 0, 0))
            nuc_ol.setModVec3(V3D(0, 0, 0))
            nuc_ol.setModUB(np.zeros((3, 3)))

            SaveHKL(
                InputWorkspace=nuclear,
                Filename=filename + "_nuc.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(InputWorkspace=nuclear, Filename=filename + "_nuc.mat")

            self.resort_hkl(nuclear, filename + "_nuc.hkl")

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=satellite,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator=">",
            )

            sat_ol = mtd[satellite].sample().getOrientedLattice()
            sat_ol.setMaxOrder(self.max_order)
            sat_ol.setModVec1(V3D(*self.modHKL[:, 0]))
            sat_ol.setModVec2(V3D(*self.modHKL[:, 1]))
            sat_ol.setModVec3(V3D(*self.modHKL[:, 2]))
            sat_ol.setModUB(self.modUB)

            SaveHKL(
                InputWorkspace=satellite,
                Filename=filename + "_sat.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(
                InputWorkspace=satellite, Filename=filename + "_sat.mat"
            )

            self.resort_hkl(satellite, filename + "_sat.hkl")

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
        self.point_groups = [point_group]

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

    def resort_hkl(self, peaks, filename):
        ol = mtd[peaks].sample().getOrientedLattice()

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
                m.append(columns[3] if max_order > 0 else 0)
                n.append(columns[4] if max_order > 0 else 0)
                p.append(columns[5] if max_order > 0 else 0)

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
        "-w", "--wobble", action="store_true", help="Off-centering correction"
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

    if args.wobble:
        WobbleCorrection("peaks", filename=args.filename)

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

    for workspace, parameters in zip(prune.workspaces, prune.parameters):
        peaks.save_peaks(workspace, parameters)


if __name__ == "__main__":
    main()
