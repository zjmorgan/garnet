import os
import sys
import traceback

import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from mantid import config

np.seterr(all="ignore", invalid="ignore")

multiprocessing.set_start_method("spawn", force=True)

config["Q.convention"] = "Crystallography"
config.setLogLevel(2, quiet=False)


class ParallelTasks:
    def __init__(self, function, combine=None):
        self.function = function
        self.combine = combine
        self.results = None

    def run_tasks(self, plan, n_proc):
        """
        Run parallel tasks with processing pool.

        Parameters
        ----------
        plan : dict
            Data reduction plan split over each process.
        n_proc : int
            Number of processes.

        """

        runs = plan["Runs"]

        pool = multiprocessing.Pool(processes=n_proc)

        def terminate_pool(e):
            print(e)
            pool.terminate()

        split = [split.tolist() for split in np.array_split(runs, n_proc)]

        join_args = [(plan, s, proc) for proc, s in enumerate(split)]

        config["MultiThreaded.MaxCores"] == "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TBB_THREAD_ENABLED"] = "0"

        try:
            result = pool.starmap_async(
                self.safe_function_wrapper,
                join_args,
                error_callback=terminate_pool,
            )
            self.results = result.get()
        except Exception as e:
            print("Exception in pool: {}".format(e))
            traceback.print_exc()
            pool.terminate()
            sys.exit()

        pool.close()
        pool.join()

        config["MultiThreaded.MaxCores"] == "4"
        os.environ.pop("OPENBLAS_NUM_THREADS")
        os.environ.pop("MKL_NUM_THREADS")
        os.environ.pop("NUMEXPR_NUM_THREADS")
        os.environ.pop("OMP_NUM_THREADS")
        os.environ.pop("TBB_THREAD_ENABLED")

        if self.combine is not None:
            self.combine(plan, self.results)

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print("Exception in worker function: {}".format(e))
            traceback.print_exc()
            raise


class ParallelProcessor:
    def __init__(self, n_proc=1):
        self.n_proc = n_proc

    def process_dict(self, data, func):
        self.function = func
        if self.n_proc > 1:
            with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
                futures = [
                    executor.submit(self.safe_function_wrapper, kv)
                    for kv in data.items()
                ]
                results = {}
                for future in as_completed(futures):
                    try:
                        key, value = future.result()
                        results[key] = value
                    except Exception as e:
                        print("Exception in pool: {}".format(e))
                        traceback.print_exc()
        else:
            results = {k: func((k, v)) for k, v in data.items()}
        return results

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print("Exception in worker function: {}".format(e))
            traceback.print_exc()
            raise
