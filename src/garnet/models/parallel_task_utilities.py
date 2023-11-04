import multiprocessing
import numpy as np

class ParallelTasks:

    def __init__(self, function, args):

        self.function = function
        self.args = args

    def run_tasks(self, values, n_proc):

        split = [split.tolist() for split in np.array_split(values, n_proc)]

        join_args = [(s, *self.args, proc) for proc, s in enumerate(split)]

        multiprocessing.set_start_method('spawn', force=True)    
        with multiprocessing.get_context('spawn').Pool(n_proc) as pool:
            pool.starmap(self.function, join_args)
            pool.close()
            pool.join()
