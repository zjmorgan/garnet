import os
import unittest

import numpy as np

from garnet.models.coverage_optimizer import CoverageOptimizerModel

class test_CoverageOptimizerModel(unittest.TestCase):

    def setUp(self):

        pass

    def tearDown(self):

        pass

    def test_garnet(self):

        inst_name = 'TOPAZ'

        axes = ['{},0,1,0,1', '135,0,0,1,1', '{},0,1,0,1']

        limits = [(-180,180), None, (-180,180)]

        UB = np.array([[-0.11589006, -0.09516246,  0.10667678],
                       [ 0.03385979,  0.1151471 ,  0.13950266],
                       [-0.13888608,  0.1074783 , -0.05500369]])

        wl_limits = [0.3, 3.5]

        d_min = 0.5

        point_group = 'm-3m'
        refl_cond = 'Body centred'

        garnet = CoverageOptimizerModel()
        garnet.initialize_parameters(inst_name, axes, limits, UB,
                                     wl_limits, point_group, refl_cond, d_min)

        garnet.initialize_settings(6, 4, 'garnet', n_proc=6)
        garnet.optimize_settings(10)
        
        self.assertGreater(self.garnet.best[0], self.garnet.best[-1])

if __name__ == "__main__":
    unittest.main()