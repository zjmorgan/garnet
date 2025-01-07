import time
from tqdm import tqdm

import numpy as np
from garnet.reduction.parallel import ParallelTasks


def func(plan, runs, proc):

    result = []

    for run in tqdm(runs, desc="Task {}".format(proc), position=proc):
        time.sleep(1)
        result.append(run)

    return result


def test_parallel():

    vals = np.arange(20).tolist()

    plan = {"Runs": vals}

    pt = ParallelTasks(func)
    pt.run_tasks(plan, 4)
    assert np.sort(np.concatenate(pt.results)).tolist() == vals


if __name__ == "__main__":
    test_parallel()
