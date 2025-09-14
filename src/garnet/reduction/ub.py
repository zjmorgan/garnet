from mantid.simpleapi import (
    SelectCellWithForm,
    SelectCellOfType,
    ShowPossibleCells,
    TransformHKL,
    CalculatePeaksHKL,
    IndexPeaks,
    FindUBUsingFFT,
    FindUBUsingLatticeParameters,
    FindUBUsingIndexedPeaks,
    OptimizeLatticeForCellType,
    CalculateUMatrix,
    HasUB,
    SetUB,
    LoadIsawUB,
    SaveIsawUB,
    CopySample,
    CreateEmptyTableWorkspace,
    FilterPeaks,
    DeleteWorkspace,
    mtd,
)

from mantid.geometry import PointGroupFactory, UnitCell

import json

import numpy as np

from scipy.spatial.transform import Rotation

import scipy.spatial
import scipy.optimize
import scipy.linalg

lattice_group = {
    "Triclinic": "-1",
    "Monoclinic": "2/m",
    "Orthorhombic": "mmm",
    "Tetragonal": "4/mmm",
    "Rhombohedral": "-3m",
    "Hexagonal": "6/mmm",
    "Cubic": "m-3m",
}

centering_matrices = {
    "P": np.eye(3),
    "A": np.array([[2, 0, 0], [0, 1, 1], [0, 1, -1]]) / 2,
    "B": np.array([[1, 0, 1], [0, 2, 0], [1, 0, -1]]) / 2,
    "C": np.array([[1, 1, 0], [1, -1, 0], [0, 0, 2]]) / 2,
    "I": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2,
    "F": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2,
    "R": np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3,
}


class UBModel:
    def __init__(self, peaks):
        """
        Tools for working with peaks and UB.

        Parameters
        ----------
        peaks : str
            Table of peaks.

        """

        self.peaks = mtd[peaks]

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        if mtd.doesExist(self.peaks.getName()):
            if hasattr(self.peaks, "sample"):
                ol = self.peaks.sample().getOrientedLattice()
                a, b, c = ol.a(), ol.b(), ol.c()
                alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

                self.a, self.b, self.c = a, b, c
                self.alpha, self.beta, self.gamma = alpha, beta, gamma

                return a, b, c, alpha, beta, gamma

    def get_max_d_spacing(self):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=self.peaks):
            if hasattr(self.peaks, "sample"):
                ol = self.peaks.sample().getOrientedLattice()

            return 1 / min([ol.astar(), ol.bstar(), ol.cstar()])

    def has_UB(self):
        """
        Check if peaks table has a UB determined.

        """

        return HasUB(Workspace=self.peaks)

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def load_UB(self, filename, run_number=None):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.
        run_number : str, optional
            Run number to replace starred expression.

        """

        LoadIsawUB(
            InputWorkspace=self.peaks,
            Filename=filename.replace("*", str(run_number)),
        )

    def determine_UB_with_primitive_cell(self, min_d, max_d, tol=0.15):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.15.

        """

        FindUBUsingFFT(
            PeaksWorkspace=self.peaks,
            MinD=min_d,
            MaxD=max_d,
            Tolerance=tol,
            DegreesPerStep=1,
        )

    def determine_UB_with_lattice_parameters(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        tol=0.2,
    ):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(
            PeaksWorkspace=self.peaks,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Tolerance=tol,
            FixParameters=True,
            NumInitial=50,
            Iterations=5,
        )

    def convert_conventional_to_primitive(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        centering,
    ):
        uc = UnitCell(a, b, c, alpha, beta, gamma)

        G = uc.getG()

        P = centering_matrices[centering]

        Gp = P.T @ G @ P

        uc.recalculateFromGstar(np.linalg.inv(Gp))

        return uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

    def calculate_transform_extents(self, centering):
        P = centering_matrices[centering]

        return np.linalg.inv(P).T

    def transform_primitive_to_conventional(self, centering):
        self.transform_lattice(self.calculate_transform_extents(centering))

    def get_primitive_cell_length_range(self, centering):
        const = self.get_lattice_parameters()
        const = self.convert_conventional_to_primitive(*const, centering)

        d_min = 0.9 * np.min(const[:3])
        d_max = 1.1 * np.max(const[:3])

        return d_min, d_max

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
        )

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.

        +----------------+---------------+----------------------+
        | Lattice system | Lengths       | Angles               |
        +================+===============+======================+
        | Cubic          | :math:`a=b=c` | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Hexagonal      | :math:`a=b`   | :math:`α=β=90,γ=120` |
        +----------------+---------------+----------------------+
        | Rhombohedral   | :math:`a=b=c` | :math:`α=β=γ`        |
        +----------------+---------------+----------------------+
        | Tetragonal     | :math:`a=b`   | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Orthorhombic   | None          | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Monoclinic     | None          | :math:`α=γ=90`       |
        +----------------+---------------+----------------------+
        | Triclinic      | None          | None                 |
        +----------------+---------------+----------------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        OptimizeLatticeForCellType(
            PeaksWorkspace=self.peaks, CellType=cell, Apply=True, Tolerance=tol
        )

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        CalculateUMatrix(
            PeaksWorkspace=self.peaks,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        SelectCellWithForm(
            PeaksWorkspace=self.peaks,
            FormNumber=number,
            Apply=True,
            Tolerance=tol,
        )

    def select_type(self, cell, centering, tol=0.1):
        """
        Transform to conventional cell using cell and centering type.

        Parameters
        ----------
        cell : str
            Cell type.
        centering : str
            Centering type.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        SelectCellOfType(
            PeaksWorkspace=self.peaks,
            CellType=cell if centering != "R" else "Rhombohedral",
            Centering=centering,
            Apply=True,
            Tolerance=tol,
        )

    def possible_conventional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form results.

        """

        result = ShowPossibleCells(
            PeaksWorkspace=self.peaks,
            MaxScalarError=max_error,
            AllowPermutations=permutations,
            BestOnly=False,
        )

        vals = [json.loads(cell) for cell in result.Cells]

        return vals

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ",".join(["{},{},{}".format(*row) for row in transform])

        TransformHKL(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=False,
        )

    def generate_lattice_transforms(self, cell):
        """
        Obtain possible transforms compatabile with a unit cell lattice.

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transform[name] = T.tolist()

        return {key: transform[key] for key in sorted(transform.keys())}

    def index_peaks(
        self,
        tol=0.15,
        sat_tol=None,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol
        save_info = True if max_order > 0 else False

        indexing = IndexPeaks(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
            RoundHKLs=True,
            CommonUBForAll=True,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            SaveModulationInfo=save_info,
        )

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """

        CalculatePeaksHKL(PeaksWorkspace=self.peaks, OverWrite=True)

    def copy_UB(self, workspace):
        """
        Copy UB to another workspace.

        Parameters
        ----------
        workspace : float
            Target workspace to copy the UB to.

        """

        CopySample(
            InputWorkspace=self.peaks,
            OutputWorkspace=workspace,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )


class Optimization:
    def __init__(self, peaks, tol=0.1):
        """
        Optimize lattice and orientation using nonlinear least squares.

        Parameters
        ----------
        peaks : str
            Name of peaks workspace to perform constrained UB optimization.
        tol : float
            Indexing tolerance for optimization.

        """

        self.peaks = peaks

        ub_inv = np.linalg.inv(self.get_UB()) / (2 * np.pi)

        Qs, hkls = [], []

        for pk in mtd[peaks]:
            hkl = np.array(pk.getHKL())
            Q = np.array(pk.getQSampleFrame())

            mod_Q = np.linalg.norm(Q)

            if mod_Q > 0:
                diff_hkl = np.abs(hkl - np.dot(ub_inv, Q))

                if (diff_hkl < tol).all():
                    hkls.append(hkl)
                    Qs.append(Q)

        self.Q, self.hkl = np.array(Qs), np.array(hkls)

        self.min_req = True if len(self.hkl) > 3 else False

    def get_UB(self):
        """
        Current UB matrux.

        Returns
        -------
        UB : 2d-array
            UB-matrix.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            return ol.getUB()

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            a, b, c = ol.a(), ol.b(), ol.c()
            alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

            self.a, self.b, self.c = a, b, c
            self.alpha, self.beta, self.gamma = alpha, beta, gamma

            return a, b, c, alpha, beta, gamma

    def get_orientation_angles(self):
        """
        Current orientation angles.

        Returns
        -------
        phi : float
            Rotation axis azimuthal angle in radians.
        theta : float
            Rotation axis polar angle in radians.
        omega : float
            Rotation angle in radians.

        """

        if mtd.doesExist(self.peaks):
            U = mtd[self.peaks].sample().getOrientedLattice().getU()

            omega = np.arccos((np.trace(U) - 1) / 2)

            val, vec = np.linalg.eig(U)

            ux, uy, uz = vec[:, np.argwhere(np.isclose(val, 1))[0][0]].real

            theta = np.arccos(uz)
            phi = np.arctan2(uy, ux)

            return phi, theta, omega

    def U_matrix(self, phi, theta, omega):
        u0 = np.cos(phi) * np.sin(theta)
        u1 = np.sin(phi) * np.sin(theta)
        u2 = np.cos(theta)

        w = omega * np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array(
            [
                [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
            ]
        )

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def fixed(self, x):
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        return (a, b, c, alpha, beta, gamma, *x)

    def cubic(self, x):
        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):
        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):
        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):
        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):
        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, hkl, Q, fun, W=np.eye(3)):
        """
        Optimization residual function.

        Parameters
        ----------
        x : list
            Parameters.
        hkl : list
            Miller indices.
        Q : list
            Q-sample vectors.
        fun : function
            Lattice constraint function.
        W: 3x3-array
            Weight matrix

        Returns
        -------
        residual : list
            Least squares residuals.

        """

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        wr = (np.einsum("ij,lj->li", UB, hkl) * 2 * np.pi - Q) @ W.T

        return wr.flatten()

    def whiten_weight_matrix(self, Q):
        sigma = np.cov(Q.T)
        L = np.linalg.cholesky(sigma)
        W = np.linalg.inv(L)
        return W

    def optimize_lattice(self, cell):
        """
        Refine the orientation and lattice parameters under constraints.

        Parameters
        ----------
        cell : str
            Lattice centering to constrain paramters.

        """

        if mtd.doesExist(self.peaks) and self.min_req:
            a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

            phi, theta, omega = self.get_orientation_angles()

            fun_dict = {
                "Fixed": self.fixed,
                "Cubic": self.cubic,
                "Rhombohedral": self.rhombohedral,
                "Tetragonal": self.tetragonal,
                "Hexagonal": self.hexagonal,
                "Orthorhombic": self.orthorhombic,
                "Monoclinic": self.monoclinic,
                "Triclinic": self.triclinic,
            }

            x0_dict = {
                "Fixed": (),
                "Cubic": (a,),
                "Rhombohedral": (a, alpha),
                "Tetragonal": (a, c),
                "Hexagonal": (a, c),
                "Orthorhombic": (a, b, c),
                "Monoclinic": (a, b, c, beta),
                "Triclinic": (a, b, c, alpha, beta, gamma),
            }

            fun = fun_dict[cell]
            x0 = x0_dict[cell]

            W = self.whiten_weight_matrix(self.Q)

            x0 += (phi, theta, omega)
            args = (self.hkl, self.Q, fun, W)

            sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

            a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

            B = self.B_matrix(a, b, c, alpha, beta, gamma)
            U = self.U_matrix(phi, theta, omega)

            UB = np.dot(U, B)

            J = sol.jac
            inv_cov = J.T.dot(J)

            cov = (
                np.linalg.inv(inv_cov)
                if np.linalg.det(inv_cov) > 0
                else np.zeros((3, 3))
            )

            chi2dof = np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
            cov *= chi2dof

            sig = np.sqrt(np.diagonal(cov))

            sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

            if np.isclose(a, sig_a):
                sig_a = 0
            if np.isclose(b, sig_b):
                sig_b = 0
            if np.isclose(c, sig_c):
                sig_c = 0

            if np.isclose(alpha, sig_alpha):
                sig_alpha = 0
            if np.isclose(beta, sig_beta):
                sig_beta = 0
            if np.isclose(gamma, sig_gamma):
                sig_gamma = 0

            ol = mtd[self.peaks].sample().getOrientedLattice()
            ol.setUB(UB)
            ol.setModUB(UB @ ol.getModHKL())
            ol.setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)


class RefineSingleCrystalGoniometer:
    def __init__(self, peaks, tol=0.12, cell="Triclinic", n_iter=1):
        self.peaks = peaks

        for iter in range(n_iter):
            self.table = self.peaks + "_#{}".format(iter)

            CreateEmptyTableWorkspace(OutputWorkspace=self.table)

            mtd[self.table].addColumn("float", "Requested Omega")
            mtd[self.table].addColumn("float", "Refined Omega")

            mtd[self.table].addColumn("float", "Requested Chi")
            mtd[self.table].addColumn("float", "Refined Chi")

            mtd[self.table].addColumn("float", "Requested Phi")
            mtd[self.table].addColumn("float", "Refined Phi")

            ol = mtd[self.peaks].sample().getOrientedLattice()

            self.U = ol.getU().copy()

            self.a = ol.a()
            self.b = ol.b()
            self.c = ol.c()
            self.alpha = ol.alpha()
            self.beta = ol.beta()
            self.gamma = ol.gamma()

            self.peak_dict = {}

            runs = np.unique(mtd[self.peaks].column("RunNumber")).tolist()

            IndexPeaks(
                PeaksWorkspace=self.peaks, Tolerance=tol, CommonUBForAll=False
            )

            for i, run in enumerate(runs):
                FilterPeaks(
                    InputWorkspace=self.peaks,
                    FilterVariable="RunNumber",
                    FilterValue=run,
                    Operator="=",
                    OutputWorkspace="_tmp",
                )

                Q = np.array(mtd["_tmp"].column("QLab"))
                hkl = np.array(mtd["_tmp"].column("IntHKL"))

                mask = hkl.any(axis=1)

                R = mtd["_tmp"].getPeak(0).getGoniometerMatrix().copy()

                omega, chi, phi = (
                    Rotation.from_matrix(R)
                    .as_euler("YZY", degrees=True)
                    .tolist()
                )

                self.peak_dict[run] = (omega, chi, phi), Q[mask], hkl[mask]

                DeleteWorkspace(Workspace="_tmp")

            self.optimize_lattice(cell)

    def calculate_goniometer(self, omega, chi, phi):
        return Rotation.from_euler(
            "YZY", [omega, chi, phi], degrees=True
        ).as_matrix()

    def get_orientation_angles(self):
        """
        Current orientation angles.

        Returns
        -------
        phi : float
            Rotation axis azimuthal angle in radians.
        theta : float
            Rotation axis polar angle in radians.
        omega : float
            Rotation angle in radians.

        """

        omega = np.arccos((np.trace(self.U) - 1) / 2)

        val, vec = np.linalg.eig(self.U)

        ux, uy, uz = vec[:, np.argwhere(np.isclose(val, 1))[0][0]].real

        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)

        return phi, theta, omega

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        return a, b, c, alpha, beta, gamma

    def U_matrix(self, phi, theta, omega):
        u0 = np.cos(phi) * np.sin(theta)
        u1 = np.sin(phi) * np.sin(theta)
        u2 = np.cos(theta)

        w = omega * np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array(
            [
                [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
            ]
        )

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def fixed(self, x):
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        return (a, b, c, alpha, beta, gamma, *x)

    def cubic(self, x):
        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):
        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):
        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):
        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):
        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, peak_dict, func):
        """
        Optimization residual function.

        Parameters
        ----------
        x : list
            Parameters.
        peak_dict : dictionary
            Goniometer angles, Q-lab vectors, Miller indices.            .
        func : function
            Lattice constraint function.

        Returns
        -------
        residual : list
            Least squares residuals.

        """

        a, b, c, alpha, beta, gamma, phi, theta, omega, *params = func(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        # ub_inv = np.linalg.inv(2 * np.pi * UB)

        params = np.array(params).reshape(-1, 3)

        diff = []

        for i, run in enumerate(peak_dict.keys()):
            (omega, chi, phi), Q, hkl = peak_dict[run]
            omega_off, chi_off, phi_off = params[i]
            R = self.calculate_goniometer(
                omega + omega_off, chi + chi_off, phi + phi_off
            )
            # hkl = np.einsum("ij,lj->li", ub_inv @ R.T, Q)
            # int_hkl = np.round(hkl)
            # diff += (hkl - int_hkl).flatten().tolist()
            diff += (
                (np.einsum("ij,lj->li", R @ UB, hkl) * 2 * np.pi - Q)
                .flatten()
                .tolist()
            )

        return diff + params.flatten().tolist()

    def optimize_lattice(self, cell):
        """
        Refine the orientation and lattice parameters under constraints.

        Parameters
        ----------
        cell : str
            Lattice centering to constrain paramters.

        """

        a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

        phi, theta, omega = self.get_orientation_angles()

        fun_dict = {
            "Fixed": self.fixed,
            "Cubic": self.cubic,
            "Rhombohedral": self.rhombohedral,
            "Tetragonal": self.tetragonal,
            "Hexagonal": self.hexagonal,
            "Orthorhombic": self.orthorhombic,
            "Monoclinic": self.monoclinic,
            "Triclinic": self.triclinic,
        }

        x0_dict = {
            "Fixed": (),
            "Cubic": (a,),
            "Rhombohedral": (a, alpha),
            "Tetragonal": (a, c),
            "Hexagonal": (a, c),
            "Orthorhombic": (a, b, c),
            "Monoclinic": (a, b, c, beta),
            "Triclinic": (a, b, c, alpha, beta, gamma),
        }

        fun = fun_dict[cell]
        x0 = x0_dict[cell]

        n = 3 * len(self.peak_dict.keys())

        x0 += (phi, theta, omega) + (0,) * n
        args = (self.peak_dict, fun)

        sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

        a, b, c, alpha, beta, gamma, phi, theta, omega, *params = fun(sol.x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        params = np.array(params).reshape(-1, 3)

        peak_dict = {}
        for i, run in enumerate(self.peak_dict.keys()):
            (omega, chi, phi), Q, hkl = self.peak_dict[run]
            omega_off, chi_off, phi_off = params[i]
            omega_prime, chi_prime, phi_prime = (
                omega + omega_off,
                chi + chi_off,
                phi + phi_off,
            )
            mtd[self.table].addRow(
                [omega, omega_prime, chi, chi_prime, phi, phi_prime]
            )
            R = self.calculate_goniometer(omega_prime, chi_prime, phi_prime)
            peak_dict[run] = R

        for peak in mtd[self.peaks]:
            run = peak.getRunNumber()
            peak.setGoniometerMatrix(peak_dict[run])

        UB = np.dot(U, B)

        J = sol.jac
        cov = np.linalg.inv(J.T.dot(J))

        chi2dof = np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
        cov *= chi2dof

        sig = np.sqrt(np.diagonal(cov))

        sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

        if np.isclose(a, sig_a):
            sig_a = 0
        if np.isclose(b, sig_b):
            sig_b = 0
        if np.isclose(c, sig_c):
            sig_c = 0

        if np.isclose(alpha, sig_alpha):
            sig_alpha = 0
        if np.isclose(beta, sig_beta):
            sig_beta = 0
        if np.isclose(gamma, sig_gamma):
            sig_gamma = 0

        ol = mtd[self.peaks].sample().getOrientedLattice()
        ol.setUB(UB)
        ol.setModUB(UB @ ol.getModHKL())
        ol.setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)


class Reorient:
    def __init__(self, peaks, crystal_system="Triclinic", lattice_system=None):
        if lattice_system is None:
            lattice_system = crystal_system

        self.peaks = peaks

        ol = mtd[self.peaks].sample().getOrientedLattice()

        self.U = ol.getU().copy()
        self.B = ol.getB().copy()

        self.a = ol.a()
        self.b = ol.b()
        self.c = ol.c()
        self.alpha = ol.alpha()
        self.beta = ol.beta()
        self.gamma = ol.gamma()

        transforms = self.cell_symmetry_matrices(
            crystal_system, lattice_system
        )

        self.minimize(transforms)

    def cell_symmetry_matrices(self, crystal_system, lattice_system):
        if crystal_system == "Cubic":
            symbol = "m-3m"
        elif crystal_system == "Hexagonal":
            symbol = "6/mmm"
        elif crystal_system == "Tetragonal":
            symbol = "4/mmm"
        elif crystal_system == "Trigonal":
            if lattice_system == "Rhombohedral":
                symbol = "-3m r"
            elif lattice_system == "Hexagonal":
                symbol = "-3m"
        elif crystal_system == "Orthorhombic":
            symbol = "mmm"
        elif crystal_system == "Monoclinic":
            symbol = "2/m"
        elif crystal_system == "Triclinic":
            symbol = "-1"

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transforms = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transforms[name] = T

        return transforms

    def minimize(self, transforms, tol=0.12):
        cost, T = -np.inf, np.eye(3)
        for order, M in transforms.items():
            UBp = self.U @ self.B @ np.linalg.inv(M)
            trace = np.trace(UBp)
            if trace > cost:
                cost = trace
                T = M.copy()

        hkl_trans = ",".join(9 * ["{}"]).format(*T.flatten())

        TransformHKL(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=False,
        )


class FindUB:
    def __init__(self, peaks, a, b, c, alpha, beta, gamma, tol=0.12):
        self.peaks = peaks

        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        t = np.linspace(0, np.pi, 2048)
        cdf = (t - np.sin(t)) / np.pi

        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

        s_vecs = np.array(mtd[self.peaks].column("QSample")) / (2 * np.pi)

        n_restarts = 3
        n_size = 100

        best_result = None
        best_index = 0
        for seed in range(n_restarts):
            subset_idx = np.random.choice(
                len(s_vecs), size=n_size, replace=False
            )
            self.s = s_vecs[subset_idx].copy()

            x = scipy.optimize.differential_evolution(
                self.objective,
                [(0, 1), (0, 1), (0, 1)],
                popsize=1000,
                polish=False,
                workers=-1,
            ).x

            UB = self.get_UB(x)

            SetUB(Workspace=self.peaks, UB=UB)

            result = IndexPeaks(
                PeaksWorkspace=self.peaks, Tolerance=tol, CommonUBForAll=False
            )

            n_index = result[0]

            if n_index > best_index:
                best_result = x
                best_index = n_index

        UB = self.get_UB(best_result)

        SetUB(Workspace=self.peaks, UB=UB)

        IndexPeaks(
            PeaksWorkspace=self.peaks, Tolerance=tol, CommonUBForAll=False
        )

    def get_UB(self, x):
        G = self.metric_tensor(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )
        Gstar = np.linalg.inv(G)

        B = scipy.linalg.cholesky(Gstar, lower=False)
        U = self.orientation_matrix(*x)

        return U @ B

    def objective(self, x):
        """
        Objective function.

        Parameters
        ----------
        x : array
            Refineable parameters.

        Returns
        -------
        neg_ind : int
            Negative number of peaks indexed.

        """

        params = np.reshape(x, (-1, 3))

        results = [self.cost(param) for param in params]

        return np.array(results)

    def cost(self, x):
        UB = self.get_UB(x)
        UB_inv = np.linalg.inv(UB)

        hkl = np.einsum("ij,kj->ik", UB_inv, self.s)

        s = np.einsum("ij,j...->i...", UB, hkl)
        s = np.linalg.norm(s, axis=0)

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        dist = np.einsum("ij,j...->i...", UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        return np.sum(dist**2)

    def metric_tensor(self, a, b, c, alpha, beta, gamma):
        """
        Calculate the metric tensor :math:`G`.

        Returns
        -------
        G : 2d-array
            3x3 matrix of lattice parameter info for Cartesian transforms.

        """

        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)

        g11 = a**2
        g22 = b**2
        g33 = c**2
        g12 = a * b * np.cos(gamma)
        g13 = c * a * np.cos(beta)
        g23 = b * c * np.cos(alpha)

        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])

        return G

    def orientation_matrix(self, u0, u1, u2):
        """
        The sample orientation matrix :math:`U`.

        Parameters
        ----------
        u0, u1, u2 : float
            Rotation paramters.

        Returns
        -------
        U : 2d-array
            3x3 sample orientation matrix.

        """

        theta = np.arccos(1 - 2 * u0)
        phi = 2 * np.pi * (u1 - 0.5)

        w = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

        omega = self._angle(u2)

        U = scipy.spatial.transform.Rotation.from_rotvec(omega * w).as_matrix()

        return U
