from mantid.simpleapi import (CreatePeaksWorkspace,
                              CombinePeaksWorkspaces,
                              PredictPeaks,
                              FilterPeaks,
                              StatisticsOfPeaksWorkspace,
                              CloneWorkspace,
                              DeleteWorkspace,
                              RenameWorkspaces,
                              LoadEmptyInstrument,
                              LoadNexus,
                              SaveNexus,
                              AddSampleLog,
                              SetGoniometer,
                              SetUB,
                              mtd)

import os
import numpy as np

from garnet.models.parallel_task_utilities import ParallelTasks

class CoverageOptimizerModel(ParallelTasks):

    def __init__(self):

        super().__init__(None, None)

    def initialize_parameters(self, inst_name, axes, limits, UB,
                                    wl_limits, point_group, refl_cond, d_min):

        self.inst_name = inst_name
        self.axes = axes
        self.limits = limits
        self.UB = UB
        self.wl_limits = wl_limits
        self.point_group = point_group
        self.refl_cond = refl_cond
        self.d_min = d_min
        self.d_max = 1/np.linalg.norm(UB, axis=0).min()

        self.best = []
        self.worst = []

        self.settings = None

    def get_coverage(self):

        return self.best, self.worst

    def get_settings(self):

        return self.settings

    def initialize_settings(self, n_orient, n_indiv, outname,
                                  outdir='/tmp/', n_proc=2):

        self.n_orient = n_orient
        self.n_indiv = n_indiv

        fname = os.path.join(outdir, outname+'_ind{}.nxs')
        fnames = [fname.format(i_indiv) for i_indiv in range(n_indiv)]

        plan = outname+'_ind{}'
        plans = [plan.format(i_indiv) for i_indiv in range(n_indiv)]

        self.function = self._generation
        self.args = (n_orient, self.inst_name, self.axes, self.limits,
                     self.UB, self.wl_limits, self.refl_cond,
                     self.d_min, self.d_max)

        self.run_tasks(fnames, n_proc)

        fitness = []
        for fname, plan in zip(fnames, plans):
            LoadNexus(Filename=fname, OutputWorkspace=plan)
            fitness.append(self._coverage(plan))

        self.fitness = np.array(fitness)
        self.fnames = np.array(fnames)
        self.plans = np.array(plans)
        self.plan = outname+'_ind{}'

        ranking = np.argsort(self.fitness)

        cov_min = self.fitness[ranking[0]]
        cov_max = self.fitness[ranking[-1]]

        self.best.append(cov_max)
        self.worst.append(cov_min)

    def _coverage(self, peaks_ws):

        StatisticsOfPeaksWorkspace(InputWorkspace=peaks_ws,
                                   OutputWorkspace=peaks_ws+'_sorted',
                                   StatisticsTable=peaks_ws+'_stats',
                                   EquivalentsWorkspace=peaks_ws+'_equiv',
                                   PointGroup=self.point_group,
                                   LatticeCentering=self.refl_cond)

        stats = mtd[peaks_ws+'_stats'].row(0)

        return stats['Data Completeness']

    def _load_instrument(self, inst_name):

        if not mtd.doesExist(inst_name):

            LoadEmptyInstrument(InstrumentName=inst_name,
                                OutputWorkspace=inst_name)

    def _set_UB(self, ws, UB):

        SetUB(Workspace=ws, UB=UB)

    def _generation(self, fnames, n_orient, inst_name, axes, limits, UB,
                          wl_limits, refl_cond, d_min, d_max, proc=1):

        if not mtd.doesExist(inst_name):

            self._load_instrument(inst_name)
            self._set_UB(inst_name, UB)

        for fname in fnames:

            CreatePeaksWorkspace(InstrumentWorkspace=inst_name,
                                 NumberOfPeaks=0,
                                 OutputWorkspace='gen_peaks')

            self._set_UB('gen_peaks', UB)

            for i_orient in range(n_orient):

                self._generate_setting(inst_name, i_orient, axes, limits, UB,
                                       wl_limits, refl_cond, d_min, d_max)

                CombinePeaksWorkspaces(LHSWorkspace='gen_peaks',
                                       RHSWorkspace='gen_peaks_ws',
                                       OutputWorkspace='gen_peaks')

                run = mtd['gen_peaks_ws'].run()

                for key in run.keys():

                    angle = run[key].value[0]

                    AddSampleLog(Workspace='gen_peaks',
                                 LogName='{}_{}'.format(key,i_orient),
                                 LogText='{}'.format(angle))

                DeleteWorkspace(Workspace='gen_peaks_ws')

            SaveNexus(InputWorkspace='gen_peaks', Filename=fname)

    def _generate_setting(self, inst_name, run, axes, limits, UB,
                                wl_limits, refl_cond, d_min, d_max):

        if not mtd.doesExist(inst_name):

            self._load_instrument(inst_name)
            self._set_UB(inst_name, UB)

        ax = []
        for axis, limit in zip(axes, limits):
            if limit is not None:
                angle = limit[0]+(limit[1]-limit[0])*np.random.random()
                ax.append(axis.format(round(angle,1)))
            else:
                ax.append(axis)
        for _ in range(6-len(axes)):
            ax.append(None)

        SetGoniometer(Workspace=inst_name,
                      Axis0=ax[0],
                      Axis1=ax[1],
                      Axis2=ax[2],
                      Axis3=ax[3],
                      Axis4=ax[4],
                      Axis5=ax[5])

        PredictPeaks(InputWorkspace=inst_name,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max,
                     WavelengthMin=wl_limits[0],
                     WavelengthMax=wl_limits[1],
                     ReflectionCondition=refl_cond,
                     OutputWorkspace='gen_peaks_ws')

        for peak in mtd['gen_peaks_ws']:
            peak.setRunNumber(run)
            peak.setIntensity(1)
            peak.setSigmaIntensity(peak.getIntensity())

    def optimize_settings(self, n_gener, elite_rate=20, mutation_rate=20):

        n_elites = int(round(elite_rate/100*self.n_indiv))

        ranking = np.argsort(self.fitness)

        for _ in range(n_gener):

            elites = self.plans[ranking[-n_elites:]]

            fraction = self.fitness/np.sum(self.fitness)

            selections = []
            while len(selections) < (self.n_indiv-n_elites+1) // 2:
                choices = np.random.choice(self.plans, size=2,
                                           p=fraction, replace=False)
                selections.append(choices)

            self._crossover(elites, selections)

            self._mutation(mutation_rate)

            self.fitness = []
            for plan in self.plans:
                self.fitness.append(self._coverage(plan))

            ranking = np.argsort(self.fitness)

            cov_min = self.fitness[ranking[0]]
            cov_max = self.fitness[ranking[-1]]

            self.best.append(cov_max)
            self.worst.append(cov_min)

    def _mutation(self, mutation_rate):

        for i_indiv, plan in enumerate(self.plans):
            for i_orient in range(self.n_orient):
                if 100*np.random.random() < mutation_rate:

                    FilterPeaks(InputWorkspace=plan,
                                FilterVariable='RunNumber',
                                FilterValue=i_orient,
                                Operator='!=',
                                OutputWorkspace=plan)

                    self._generate_setting(self.inst_name, i_orient,
                                           self.axes, self.limits, self.UB,
                                           self.wl_limits, self.refl_cond,
                                           self.d_min, self.d_max)

                    CombinePeaksWorkspaces(LHSWorkspace=plan,
                                           RHSWorkspace='gen_peaks_ws',
                                           OutputWorkspace=plan)


                    run = mtd['gen_peaks_ws'].run()

                    for key in run.keys():

                        angle = run[key].value[0]

                        AddSampleLog(Workspace=plan,
                                     LogName='{}_{}'.format(key,i_orient),
                                     LogText='{}'.format(angle))

                    DeleteWorkspace(Workspace='gen_peaks_ws')

    def _crossover(self, elites, selections):

        i_indiv = 0

        next_plan = 'next_'+self.plan
        next_plans = []

        for best in elites:

            plan = next_plan.format(i_indiv)

            CloneWorkspace(InputWorkspace=best,
                           OutputWorkspace=plan)

            next_plans.append(plan)

            i_indiv += 1

        for recombinations in selections:

            k = np.random.randint(self.n_orient)

            for parents in [recombinations, recombinations[::-1]]:

                if i_indiv < self.n_indiv:

                    plan = next_plan.format(i_indiv)

                    FilterPeaks(InputWorkspace=parents[0],
                                FilterVariable='RunNumber',
                                FilterValue=k,
                                Operator='<',
                                OutputWorkspace='cross_peaks_ws0')

                    FilterPeaks(InputWorkspace=parents[1],
                                FilterVariable='RunNumber',
                                FilterValue=k,
                                Operator='>=',
                                OutputWorkspace='cross_peaks_ws1')

                    CombinePeaksWorkspaces(LHSWorkspace='cross_peaks_ws0',
                                           RHSWorkspace='cross_peaks_ws1',
                                           OutputWorkspace=plan)

                    DeleteWorkspace(Workspace='cross_peaks_ws0')
                    DeleteWorkspace(Workspace='cross_peaks_ws1')

                    next_plans.append(plan)

                    i_indiv += 1

        RenameWorkspaces(InputWorkspaces=next_plans,
                         WorkspaceNames=self.plans)