import numpy as np

from garnet.views.parallel_utilities import WorkerThread

class CoverageOptimizer:

    def __init__(self, view, model):

        self.view = view
        self.model = model

        self.view.initialize_button.clicked.connect(self.initialize)
        self.view.optimize_button.clicked.connect(self.optimize)

        self.p = None
        self.init = False

    def process_finished(self):

        best, worst = self.model.get_coverage()

        best = [0]+best
        worst = [0]+worst
        generation = np.arange(len(best))

        self.view.update_plots(generation, best, worst)
        self.p = None
        self.init = True

    def optimize(self):

        if self.init:

            for _ in range(self.view.get_generations()):

                self.model.optimize_settings(1, self.view.get_elite_rate(),
                                                self.view.get_mutation_rate())

                best, worst = self.model.get_coverage()

                best = [0]+best
                worst = [0]+worst
                generation = np.arange(len(best))

                self.view.update_plots(generation, best, worst)

                #self.view.update_table(self.model.get_settings())

    def initialize(self):

        inst_name = 'TOPAZ'

        axes = ['{},0,1,0,1', '135,0,0,1,1', '{},0,1,0,1']

        limits = [(-180,180), None, (-180,180)]

        UB = np.array([[-0.11589006, -0.09516246,  0.10667678],
                       [ 0.03385979,  0.1151471 ,  0.13950266],
                       [-0.13888608,  0.1074783 , -0.05500369]])

        wl_limits = [0.4, 3.5]

        d_min = 0.5

        point_group = 'm-3m'
        refl_cond = 'Body centred'

        self.model.initialize_parameters(inst_name, axes, limits, UB,
                                         wl_limits, point_group,
                                         refl_cond, d_min)

        args = [self.view.get_settings(),
                self.view.get_individuals(), 'garnet', '/tmp',  5]

        self.view.generate_table(['omega','chi','phi'],
                                 self.view.get_settings())

        if self.p is None:
            self.p = WorkerThread()
            self.p.setup_function(self.model.initialize_settings, args)
            self.p.finished.connect(self.process_finished)
            self.p.start()
