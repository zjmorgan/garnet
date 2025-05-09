import sys
import os

import multiprocess as multiprocessing

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, ".."))
sys.path.append(directory)

from mantid import config

config["Q.convention"] = "Crystallography"

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parallel import ParallelTasks
from garnet.reduction.integration import Integration
from garnet.reduction.normalization import Normalization
from garnet.reduction.parametrization import Parametrization
from garnet.reduction.data import DataModel
from garnet.config.instruments import beamlines

inst_dict = {
    "corelli": "CORELLI",
    "bl9": "CORELLI",
    "topaz": "TOPAZ",
    "bl12": "TOPAZ",
    "mandi": "MANDI",
    "bl11b": "MANDI",
    "snap": "SNAP",
    "bl3": "SNAP",
    "demand": "DEMAND",
    "hb3a": "DEMAND",
    "wand2": "WAND²",
    "hb2c": "WAND²",
}

reduction_types = {
    "temp": None,
    "int": "Integration",
    "norm": "Normalization",
    "param": "Parameterization",
}

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    filename, reduction, arg = sys.argv[1], sys.argv[2], sys.argv[3]

    assert reduction in reduction_types.keys()

    if arg.isdigit():
        n_proc = int(arg)
    else:
        instrument = inst_dict[arg.lower()]
        assert filename.endswith(".yaml")

    rp = ReductionPlan()

    if reduction == "temp":
        rp.generate_plan(instrument)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            rp.plan.pop("OutputName", None)
            rp.plan.pop("OutputPath", None)
            rp.save_plan(filename)

    else:
        rp.load_plan(filename)

        by_run = True

        if reduction == "norm":
            func = Normalization.normalize_parallel
            comb = Normalization.combine_parallel
            inst = Normalization(rp.plan)
        elif reduction == "param":
            func = Parametrization.parametrize_parallel
            comb = Parametrization.combine_parallel
            inst = Parametrization(rp.plan)
        else:
            func = Integration.integrate_parallel
            comb = Integration.combine_parallel
            inst = Integration(rp.plan)

        for key in reduction_types.keys():
            if key != reduction and key != "temp":
                if rp.plan.get(reduction_types[key]) is not None:
                    rp.plan.pop(reduction_types[key])

        inst.create_directories()

        data = DataModel(beamlines[rp.plan["Instrument"]])
        data.update_raw_path(rp.plan)

        # if reduction == "int":
        #     profile_fit = rp.plan["Integration"].get("ProfileFit")
        #     if profile_fit is None:
        #         profile_fit = True
        #     by_run = False
        #     if data.laue and not profile_fit:
        #         by_run = True

        if by_run:
            pt = ParallelTasks(func, comb)

            n_runs = len(rp.plan["Runs"])

            max_proc = min(os.cpu_count(), n_runs)

            if n_proc > max_proc:
                n_proc = max_proc

            pt.run_tasks(rp.plan, n_proc)

        else:
            max_proc = os.cpu_count()

            if n_proc > max_proc:
                n_proc = max_proc

            inst.integrate(n_proc)

        fname = filename.replace(".yaml", "_" + reduction + ".json")

        fname = os.path.join(inst.get_output_path(), os.path.basename(fname))

        rp.save_plan(fname)
