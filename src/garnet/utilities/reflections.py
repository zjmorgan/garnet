import os
import sys

from mantid.simpleapi import (
    LoadNexus,
    FilterPeaks,
    SortPeaksWorkspace,
    CombinePeaksWorkspaces,
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

from mantid.geometry import PointGroupFactory
from mantid import config

config["Q.convention"] = "Crystallography"

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from concurrent.futures import ProcessPoolExecutor

import argparse


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
        assert "PeaksWorkspace" in str(type(peaks))

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

        assert not np.isclose(np.dot(u_vector, v_vector), 0)

        self.u_vector = u_vector
        self.v_vector = v_vector

        if shape == "sphere":
            assert type(params) is float
        elif shape == "cylinder":
            assert len(params) == 2
        else:
            assert len(params) == 3

        self.shape = shape
        self.params = params

        if filename is not None:
            assert type(filename) is str
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.set_shape()
        self.create_sample_orientations()

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
                self.params / 200, alpha, beta, gamma
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
                self.params[0] / 200, self.params[1] / 100, alpha, beta, gamma
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
                self.params[0] / 100,
                self.params[1] / 100,
                self.params[2] / 100,
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

        for i, run in enumerate(self.runs):
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
            filename = os.path.splitext(self.filename) + "_abs.txt"

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

    def create_sample_orientations(self):
        filename = os.path.splitext(self.filename) + "_abs.pdf"

        with PdfPages(filename) as pdf:
            for R, run in zip(self.Rs, self.runs):
                hkl = np.eye(3)
                q = np.matmul(self.UB, hkl)

                reciprocal_lattice = np.matmul(R, q)

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

                ax.set_title("run #{}".format(run))
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


class PrunePeaks:
    def __init__(self, peaks):
        assert "PeaksWorkspace" in str(type(peaks))

        self.peaks = peaks

        self.remove_non_integrated()
        self.remove_off_centered()

        self.models = ["type I gaussian", "type I lorentzian", "type II"]

        self.spherical_extinction()
        self.extract_info()

        self.n_iter = 7

        V = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert V > 0

        self.V = V

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

        for model in self.models:
            if "gaussian" in model:
                file_ext = "secondary_extinction_gaussian_sphere.csv"
            elif "lorentzian" in model:
                file_ext = "secondary_extinction_lorentzian_sphere.csv"
            else:
                file_ext = "primary_extinction_sphere.csv"

            data = np.loadtxt(
                file_ext, skiprows=1, delimiter=",", usecols=np.arange(91)
            )
            theta = np.loadtxt(file_ext, delimiter=",", max_rows=1)

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

        xs = (
            a**2
            / V**2
            * lamda**3
            * g
            / np.sin(two_theta)
            * (Tbar * 1e8)
            * F2
        )

        return xs

    def extinction_xp(self, r2, F2, lamda, V):
        a = 1e-4  # Ang

        xp = a**2 / V**2 * lamda**2 * r2 * F2

        return xp

    def extinction_correction(self, param, F2, two_theta, lamda, Tbar, model):
        c1, c2 = self.f1[model](two_theta), self.f2[model](two_theta)

        V = self.V

        if model == "type II":
            xp = self.extinction_xp(param, F2, lamda, V)
            yp = 1 / (1 + c1 * xp) ** c2
            return yp
        else:
            xs = self.extinction_xs(param, F2, two_theta, lamda, Tbar, V)
            ys = 1 / (1 + c1 * xs) * c2
            return ys

    def scale_factor(self, y_hat, y, e):
        s = np.nanmedian(y / y_hat)

        return s

    def cost(self, x, model):
        wr = []

        for key in self.peaks_dict.keys():
            I, sig, lamda, two_theta, Tbar, k = self.peaks_dict[key]
            diff = self.residual(x, I, sig, two_theta, lamda, Tbar, k, model)
            wr += diff.tolist()

        wr = np.array(wr)

        return wr[np.isfinite(wr)]

    def residual(self, x, I, sig, two_theta, lamda, Tbar, k, model):
        (p,) = x[0]

        I0 = np.copy(k)

        y = self.extinction_correction(p, I0, two_theta, lamda, Tbar, model)

        s = self.scale_factor(y, I, sig)

        return (s * y - I) / sig

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
            Tbar = peak.getAbsorptionWeightedPathLength()

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
            peaks_dict[key] = *items, np.nanmedian(items[0])

        self.peaks_dict = peaks_dict

    def extiction_prune(self):
        for model in self.models:
            app = "_{}.pdf".format(model).replace(" ", "_")

            filename = os.path.splitext(self.filename) + app

            with PdfPages(filename) as pdf:
                I_max = 0
                for key in self.peaks_dict.keys():
                    items = list(self.peaks_dict.get(key))
                    I = np.nanmedian(items[0])
                    if I > I_max:
                        I_max = I
                    items[-1] = I
                    self.peaks_dict[key] = items

                for i_iter in range(self.n_iter):
                    sol = scipy.optimize.least_squares(
                        self.cost,
                        x0=(1 / I_max),
                        bounds=([0], [np.inf]),
                        args=(model),
                        loss="soft_l1",
                    )

                    ext_dict = {}

                    scales, I0s = [], []

                    for key in self.peaks_dict.keys():
                        I, sig, lamda, two_theta, Tbar, k = self.peaks_dict[
                            key
                        ]

                        (p,) = sol.x[0]

                        I0 = np.copy(k)

                        y = self.extinction_correction(
                            p, I0, two_theta, lamda, Tbar, model
                        )

                        s = self.scale_factor(y, I, sig)

                        self.peaks_dict[key][-1] = s

                        ext_dict[key] = s, F2, y

                        scales.append(s)
                        I0s.append(I0)

                    scales = np.array(scales)
                    I0s = np.array(I0s)

                    mask = (I0s > 0) & (scales > 0)

                    fig, ax = plt.subplots(1, 1, layout="constrained")
                    ax.plot(scales[mask], I0s[mask], ".")
                    ax.minorticks_on()
                    ax.set_xlabel("scale $s$")
                    ax.set_ylabel("$I_0$")
                    # ax.set_xscale('log')
                    # ax.set_yscale('log')
                    pdf.savefig()


class SaveLoadPeaks:
    def __init__(self, peaks, filename, scale=None):
        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        if scale is not None:
            assert scale > 0

        self.scale = scale

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

    def load_peaks(self):
        LoadNexus(Filename=self.filename, OutputWorkspace=self.peaks)

    def save_peaks(self, name=None):
        self.rescale_intensities()

        if name is not None:
            app = "_{}".format(name).replace(" ", "_")
        else:
            app = ""

        filename = os.path.splitext(self.filename) + app

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=3,
            Operator=">",
        )

        hkl = [
            mtd[self.peaks].column(mtd[self.peaks].getColumnNames().index(col))
            for col in ["h", "k", "l"]
        ]
        s = [
            1 / mtd[self.peaks].sample().getOrientedLattice().d(h, k, l)
            for h, k, l in zip(*hkl)
        ]

        hkls = np.round(np.column_stack([*hkl, s]) * 1000, 1).astype(int)
        sort = np.lexsort(hkls.T).tolist()
        for no, i in enumerate(sort):
            peak = mtd["prune"].getPeak(i)
            peak.setPeakNumber(no)
            peak.setRunNumber(1)

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            ColumnNameToSortBy="PeakNumber",
            SortAscending=True,
            OutputWorkspace=self.peaks,
        )

        SaveHKL(
            InputWorkspace=self.peaks,
            Filename=filename + ".hkl",
            DirectionCosines=True,
        )

        SaveIsawUB(InputWorkspace=self.peaks, Filename=filename + ".mat")


def main():
    parser = argparse.ArgumentParser(description="Corrections for integration")

    parser.add_argument(
        "filename", type=str, help="Peaks Workspace", required=True
    )

    parser.add_argument(
        "-w", "--wobble", action="store_false", help="Off-centering correction"
    )

    parser.add_argument(
        "-f",
        "--formula",
        type=bool,
        default="Yb3 Al5 O12",
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

    args = parser.parse_args()

    # if args.verbose:
    #     print("Verbose mode enabled.")
    # print(f"Hello, {args.name}!")


if __name__ == "__main__":
    main()

# FilterPeaks(
#     InputWorkspace="peaks",
#     OutputWorkspace="peaks",
#     FilterVariable="h^2+k^2+l^2",
#     FilterValue=0,
#     Operator=">",
# )

# SortPeaksWorkspace(
#     InputWorkspace="peaks",
#     OutputWorkspace="peaks",
#     ColumnNameToSortBy="RunNumber",
#     SortAscending=False,
# )

# SortPeaksWorkspace(
#     InputWorkspace="peaks",
#     OutputWorkspace="peaks",
#     ColumnNameToSortBy="Intens",
#     SortAscending=False,
# )


# AbsorptionCorrection(
#     "peaks", "Yb3 Al5 O12", 8, [1, 0, 0], [0, 1, 0], 0.3, "sphere"
# )

# CloneWorkspace(InputWorkspace='peaks',
#                OutputWorkspace='peaks_corr')

# StatisticsOfPeaksWorkspace(
#     InputWorkspace="peaks_corr",
#     PointGroup=point_group,
#     OutputWorkspace="stats",
#     EquivalentIntensities="Median",
#     SigmaCritical=3,
#     WeightedZScore=False,
# )
