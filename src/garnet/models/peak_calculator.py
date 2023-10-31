from mantid.kernel import V3D
from mantid.simpleapi import (CreatePeaksWorkspace,
                              DeleteTableRows,
                              AddPeakHKL,
                              HasUB,
                              SetUB,
                              mtd)

class PeakCalculator():

    def __init__(self, ref_ws=None):

        CreatePeaksWorkspace(InstrumentWorkspace=ref_ws,
                             NumberOfPeaks=0,
                             OutputType='LeanElasticPeak',
                             OutputWorkspace='peak_calc_ws')

        if not HasUB(Workspace='peak_calc_ws'):
            self.update_lattice_parameters(2, 1, 1, 90, 90, 90)

        AddPeakHKL(Workspace='peak_calc_ws', HKL=V3D(1,0,0))
        AddPeakHKL(Workspace='peak_calc_ws', HKL=V3D(0,1,0))

    def update_lattice_parameters(self, a, b, c, alpha, beta, gamma):

        SetUB(Workspace='peak_calc_ws', 
              a=a, 
              b=b, 
              c=c, 
              alpha=alpha,
              beta=beta,
              gamma=gamma)

    def calculate(self, hkl_1, hkl_2):

        DeleteTableRows(TableWorkspace='peak_calc_ws', Rows=1)
        DeleteTableRows(TableWorkspace='peak_calc_ws', Rows=0)

        AddPeakHKL(Workspace='peak_calc_ws', HKL=V3D(*hkl_1))
        AddPeakHKL(Workspace='peak_calc_ws', HKL=V3D(*hkl_2))

        peak_1 = mtd['peak_calc_ws'].getPeak(0)
        peak_2 = mtd['peak_calc_ws'].getPeak(1)

        ol = mtd['peak_calc_ws'].sample().getOrientedLattice()

        d_1 = peak_1.getDSpacing()
        d_2 = peak_2.getDSpacing()
        phi_12 = ol.recAngle(*hkl_1, *hkl_2)

        return d_1, d_2, phi_12