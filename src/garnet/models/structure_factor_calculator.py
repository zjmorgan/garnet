from mantid.kernel import V3D
from mantid.geometry import (CrystalStructure,
                             ReflectionGenerator,
                             ReflectionConditionFilter)
from mantid.simpleapi import (CreatePeaksWorkspace,
                              DeleteTableRows,
                              AddPeakHKL,
                              LoadCIF,
                              SetUB,
                              mtd)

import numpy as np

class StructureFactorCalculatorModel():

    def __init__(self, ref_ws=None):

        CreatePeaksWorkspace(InstrumentWorkspace=ref_ws,
                             NumberOfPeaks=0,
                             OutputType='LeanElasticPeak',
                             OutputWorkspace='struct_fact_ws')

    def load_CIF(self, filename):

        LoadCIF(Workspace='struct_fact_ws',
                InputFile=filename)

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        unit_cell = cryst_struct.getUnitCell()

        self.update_lattice_parameters(unit_cell.a(),
                                       unit_cell.b(),
                                       unit_cell.c(),
                                       unit_cell.alpha(),
                                       unit_cell.beta(),
                                       unit_cell.gamma())

    def set_crystal_structure(self, lattice_params, space_group, scatterers):

        line = ' '.join(['{}']*6)

        constants = line.format(*lattice_params)

        atom_info = ';'.join([line.format(*s) for s in scatterers])

        cs = CrystalStructure(constants, space_group, atom_info)

        mtd['struct_fact_ws'].sample().setCrystalStructure(cs)

        self.update_lattice_parameters(*lattice_params)

    def update_lattice_parameters(self, a, b, c, alpha, beta, gamma):

        SetUB(Workspace='struct_fact_ws',
              a=a,
              b=b,
              c=c,
              alpha=alpha,
              beta=beta,
              gamma=gamma)

    def generate_F2(self, d_min=0.7):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        generator = ReflectionGenerator(cryst_struct)

        sf_filt = ReflectionConditionFilter.StructureFactor

        unit_cell = cryst_struct.getUnitCell()

        d_max = np.max([unit_cell.a(), unit_cell.b(), unit_cell.c()])

        hkls = generator.getUniqueHKLsUsingFilter(d_min, d_max, sf_filt)

        ds = generator.getDValues(hkls)

        F2s = generator.getFsSquared(hkls)

        # for peak_row in range(mtd['struct_fact_ws'].getNumberPeaks()-1,-1,-1):
        #     DeleteTableRows(TableWorkspace='struct_fact_ws', Rows=peak_row)

        # for peak_row, (hkl, F2) in enumerate(zip(hkls, F2s)):
        #     AddPeakHKL(Workspace='struct_fact_ws', HKL=V3D(*hkl))
        #     peak = mtd['struct_fact_ws'].getPeak(peak_row)
        #     peak.setIntensity(F2)

        return hkls, ds, F2s

    def get_point_group(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        return cryst_struct.getSpaceGroup().getPointGroup().getHMSymbol()

    def get_space_group(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        return cryst_struct.getSpaceGroup().getHMSymbol()


