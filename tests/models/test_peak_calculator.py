import unittest

from garnet.models.peak_calculator import PeakCalculator

class test_PeakCalculator(unittest.TestCase):

    def setUp(self):

        pass

    def tearDown(self):

        pass

    def test_lattice(self):

        peak_calc = PeakCalculator()
        peak_calc.update_lattice_parameters(5, 6, 7, 80, 90, 100)

        d_1, d_2, phi_12 = peak_calc.calculate([1,2,3], [4,1,3])       

        self.assertAlmostEqual(d_1, 1.781, 3)
        self.assertAlmostEqual(d_2, 1.066, 3)
        self.assertAlmostEqual(phi_12, 37.892, 3)
