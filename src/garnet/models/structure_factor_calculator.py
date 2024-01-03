from mantid.kernel import V3D

from mantid.geometry import (CrystalStructure,
                             ReflectionGenerator,
                             ReflectionConditionFilter,
                             SpaceGroup,
                             PointGroup,
                             PointGroupFactory,
                             SpaceGroupFactory)

from mantid.simpleapi import (CreatePeaksWorkspace,
                              DeleteTableRows,
                              AddPeakHKL,
                              LoadCIF,
                              SetUB,
                              mtd)

import numpy as np
import scipy.linalg

class StructureFactorCalculatorModel():

    def __init__(self, ref_ws=None):

        CreatePeaksWorkspace(InstrumentWorkspace=ref_ws,
                             NumberOfPeaks=0,
                             OutputType='LeanElasticPeak',
                             OutputWorkspace='struct_fact_ws')

    def generate_space_groups_from_crystal_system(self, system):

        pg_system = getattr(PointGroup.CrystalSystem, system)
        pgs = list(PointGroupFactory.getPointGroupSymbols(pg_system))
        pgs = [PointGroupFactory.createPointGroup(pg) for pg in pgs]
        sgs = [SpaceGroupFactory.getSpaceGroupsForPointGroup(pg) for pg in pgs]
        sgs = [sg for sg_list in sgs for sg in sg_list]
        sgs = [SpaceGroupFactory.createSpaceGroup(sg) for sg in sgs]

        nos = np.unique([sg.getNumber() for sg in sgs]).tolist()

        space_group = []
        for no in nos:
            symbol = SpaceGroupFactory.subscribedSpaceGroupSymbols(no)[0]
            space_group.append('{}: {}'.format(no,symbol))

        return space_group

    def generate_settings_from_space_group(self, sg):

        no, symbol = sg.split(': ')

        return list(SpaceGroupFactory.subscribedSpaceGroupSymbols(int(no)))

    def load_CIF(self, filename):

        LoadCIF(Workspace='struct_fact_ws', InputFile=filename)

        params = self.get_lattice_constants()

        self.update_lattice_parameters(*params)

    def set_crystal_structure(self, params, space_group, scatterers):

        line = ' '.join(['{}']*6)

        constants = line.format(*params)

        atom_info = ';'.join([line.format(*s) for s in scatterers])

        cs = CrystalStructure(constants, space_group, atom_info)

        mtd['struct_fact_ws'].sample().setCrystalStructure(cs)

        self.update_lattice_parameters(*params)

    def update_lattice_parameters(self, a, b, c, alpha, beta, gamma):

        SetUB(Workspace='struct_fact_ws',
              a=a,
              b=b,
              c=c,
              alpha=alpha,
              beta=beta,
              gamma=gamma)

        self.B = mtd['struct_fact_ws'].sample().getOrientedLattice().getB()

    def generate_F2(self, d_min=0.7):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        generator = ReflectionGenerator(cryst_struct)

        sf_filt = ReflectionConditionFilter.StructureFactor

        unit_cell = cryst_struct.getUnitCell()

        d_max = np.max([unit_cell.a(), unit_cell.b(), unit_cell.c()])

        hkls = generator.getUniqueHKLsUsingFilter(d_min, d_max, sf_filt)

        ds = generator.getDValues(hkls)

        F2s = generator.getFsSquared(hkls)

        for peak_row in range(mtd['struct_fact_ws'].getNumberPeaks()-1,-1,-1):
            DeleteTableRows(TableWorkspace='struct_fact_ws', Rows=peak_row)

        for peak_row, (hkl, F2) in enumerate(zip(hkls, F2s)):
            AddPeakHKL(Workspace='struct_fact_ws', HKL=V3D(*hkl))
            peak = mtd['struct_fact_ws'].getPeak(peak_row)
            peak.setIntensity(F2)

        return hkls, ds, F2s

    def get_crystal_system(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getCrystalSystem().name

    def get_lattice_system(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getLatticeSystem().name

    def get_point_group_name(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getName()

    def get_space_group(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        no = sg.getNumber()
        symbol = SpaceGroupFactory.subscribedSpaceGroupSymbols(no)[0]

        return '{}: {}'.format(no,symbol)

    def get_setting(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        return cryst_struct.getSpaceGroup().getHMSymbol()

    def get_lattice_constants(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        uc = cryst_struct.getUnitCell()

        params = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

        return params

    def get_scatterers(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        return [atm.split(' ') for atm in list(cryst_struct.getScatterers())]

    def generate_atom_positions(self):

        scatterers = self.get_scatterers()        

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        A = self.get_unit_cell_transform()
        
        atom_dict = {}

        for ind, scatterer in enumerate(scatterers):

            atom, x, y, z, occ, U = scatterer

            xyz = np.array(sg.getEquivalentPositions([x,y,z]))

            r_xyz = np.einsum('ij,kj->ki', A, xyz).tolist()
            r_occ = np.full(len(xyz), float(occ)).tolist()
            r_ind = np.full(len(xyz), ind).tolist()

            if atom_dict.get(atom) is None:
                atom_dict[atom] = r_xyz, r_occ, r_ind
            else:
                R_xyz, R_occ, R_ind = atom_dict[atom]
                R_xyz += r_xyz
                R_occ += r_occ
                R_ind += r_ind
                atom_dict[atom] = R_xyz, R_occ, R_ind

        return atom_dict

    def get_unit_cell_transform(self):

        cryst_struct = mtd['struct_fact_ws'].sample().getCrystalStructure()

        uc = cryst_struct.getUnitCell()

        G = uc.getG()

        A = scipy.linalg.cholesky(G, lower=False)

        return A

    def get_transform(self):

        A = self.get_unit_cell_transform()

        t = A.copy()
        t /= np.max(t, axis=1)

        return t

    def ab_star_axes(self):

        if self.B is not None:

            return np.dot(self.B, [0,0,1]), np.dot(self.B, [1,0,0])

    def bc_star_axes(self):

        if self.B is not None:

            return np.dot(self.B, [1,0,0]), np.dot(self.B, [0,1,0])

    def ca_star_axes(self):

        if self.B is not None:

            return np.dot(self.B, [0,1,0]), np.dot(self.B, [0,0,1])

    def ab_axes(self):

        if self.B is not None:

            return np.cross(*self.bc_star_axes()), \
                   np.cross(*self.ca_star_axes())

    def bc_axes(self):

        if self.B is not None:

            return np.cross(*self.ca_star_axes()), \
                   np.cross(*self.ab_star_axes())

    def ca_axes(self):

        if self.B is not None:

            return np.cross(*self.ab_star_axes()), \
                   np.cross(*self.bc_star_axes())

    def get_vector(self, axes_type, ind):

        if self.B is not None:

            if axes_type == '[hkl]':
                matrix = self.B
            else:
                matrix = np.cross(np.dot(self.B, np.roll(np.eye(3),2,1)).T,
                                  np.dot(self.B, np.roll(np.eye(3),1,1)).T).T

            vec = np.dot(matrix, ind)

            return vec

    def constrain_parameters(self):

        params = np.array([False]*6)

        lattice_system = self.get_lattice_system()

        if lattice_system == 'Cubic':
            params[1:6] = True
        elif lattice_system == 'Rhombohedral':
            params[1:3] = True
            params[4:6] = True
        elif lattice_system == 'Hexagonal' or lattice_system == 'Tetragonal':
            params[2] = True
            params[3:6] = True
        elif lattice_system == 'Orthorhombic':
            params[3:6] = True
        elif lattice_system == 'Monoclinic':
            pg_name = self.get_point_group_name()
            if 'unique axis b' in pg_name:
                params[3] = True
                params[5] = True
            else:
                params[3:4] = True

        return params.tolist()