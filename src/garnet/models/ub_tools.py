from mantid.simpleapi import (SelectCellWithForm,
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
                              LoadIsawUB,
                              SaveIsawUB,
                              mtd)

from mantid.geometry import PointGroupFactory

from mantid.utils.logging  import capture_logs

import numpy as np

axes_group = {'Triclinic': '-1',
              'Monoclinic': '2/m',
              'Orthorhombic': 'mmm',
              'Tetragonal': '4/mmm',
              'Rhombohedral': '-3m',
              'Hexagonal': '6/mmm',
              'Cubic': 'm-3m'}

class UBTools():

    def __init__(self, peaks_ws):

        self.peaks_ws = mtd[peaks_ws]

    def has_ub(self, filename):

        return HasUB(Workspace=self.peaks_ws)

    def save_ub(self, filename):

        SaveIsawUB(InputWorkspace=self.peaks_ws, Filename=filename)

    def load_ub(self, filename):

        LoadIsawUB(InputWorkspace=self.peaks_ws, Filename=filename)

    def find_ub_niggli_cell(self, min_d, max_d, tol=0.1):

        FindUBUsingFFT(PeaksWorkspace=self.peak_ws, 
                       Tolerance=tol,
                       MinD=min_d,
                       MaxD=max_d)

    def find_ub_lattice_parameters(self, a, 
                                         b,
                                         c,
                                         alpha,
                                         beta,
                                         gamma,
                                         tol=0.1):

        FindUBUsingLatticeParameters(PeaksWorkspace=self.peaks_ws, 
                                     a=a,
                                     b=b,
                                     c=c,
                                     alpha=alpha,
                                     beta=beta,
                                     gamma=gamma,
                                     Tolerance=tol)

    def find_ub_indexed_peaks(self, tol=0.1, sat_tol=0.1):

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(PeaksWorkspace=self.peaks_ws,
                                Tolerance=tol,
                                ToleranceForSatellite=tol_for_sat)

    def constrain_lattice_parameters(self, cell, tol=0.1):

        OptimizeLatticeForCellType(PeaksWorkspace=self.peaks_ws,
                                   CellType=cell,
                                   Apply=True,
                                   Tolerance=tol)

    def determine_orientation(self, a, b, c, alpha, beta, gamma):

        CalculateUMatrix(PeaksWorkspace=self.peaks_ws, 
                         a=a,
                         b=b,
                         c=c,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma)

    def select_cell(self, number, tol=0.1):
        
        SelectCellWithForm(PeaksWorkspace=self.peaks_ws,
                           FormNumber=number,
                           Apply=True,
                           Tolerance=tol)

    def possible_cells(self, max_error=0.2, permutations=True):

        with capture_logs(level='notice') as logs:

            ShowPossibleCells(PeaksWorkspace=self.peaks_ws,
                              MaxScalarError=max_error,
                              AllowPermuations=permutations,
                              BestOnly=False)

            vals = logs.getvalue()
            vals = [val for val in vals.split('\n') if val.startswith('Form')]

    def transform_cell(self, transform, tol=0.1):

        hkl_trans = ','.join(['{},{},{}'.format(*row) for row in transform])
        
        TransformHKL(PeaksWorkspace=self.peaks_ws,
                     Tolerance=tol,
                     HKLTransform=hkl_trans)

    def generate_transform_cell(self, cell):

        symbol = axes_group[cell]        

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = '{}: '.format(symop.getOrder())+symop.getIdentifier()
                transform[name] = T.tolist()

        return {key:transform[key] for key in sorted(transform.keys())}

    def index_peaks(self, tol=0.1,
                          sat_tol=None,
                          mod_vec_1=[0,0,0],
                          mod_vec_2=[0,0,0],
                          mod_vec_3=[0,0,0],
                          max_order=0,
                          cross_terms=False):

        tol_for_sat = sat_tol if sat_tol is not None else tol

        indexing = IndexPeaks(PeaksWorkspace=self.peaks_ws,
                              Tolerance=tol,
                              ToleranceForSatellite=tol_for_sat,
                              RoundHKLs=True,
                              CommonUBForAll=True,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True)

        return indexing

    def calculate_hkl(self):
    
        CalculatePeaksHKL(PeaksWorkspace=self.peaks_ws,
                          OverWrite=True)
