import os
import unittest

from garnet.models.crystal_structure_tools import CrystalStructureModel

class test_CrystalStructureModel(unittest.TestCase):

    def setUp(self):

        pass

    def tearDown(self):

        pass

    def test_silicon(self):

        struct_fact_calc = CrystalStructureModel()

        lattice_params = [5.431]*3+[90]*3

        space_group = 'F d -3 m'

        scatterers = [['Si',0,0,0,1,0.05]]

        struct_fact_calc.set_crystal_structure(lattice_params,
                                               space_group,
                                               scatterers)

        hkls, ds, F2s = struct_fact_calc.generate_F2()

        self.assertEqual(struct_fact_calc.get_space_group(), space_group)
        self.assertEqual(struct_fact_calc.get_point_group(), 'm-3m')

    # def test_T4_lysozyme(self):

    #     struct_fact_calc = StructureFactorCalculator()
    #     struct_fact_calc.load_CIF(os.path.abspath('tests/data/5vnq.cif'))

    #     hkls, ds, F2s = struct_fact_calc.generate_F2(d_min=2)

    #     self.assertEqual(len(F2s), 1000)
    #     self.assertEqual(struct_fact_calc.get_space_group(), 'P 32 2 1')
