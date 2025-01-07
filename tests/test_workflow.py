import os
import tempfile
import subprocess

import garnet.workflow


def test_template():

    script = os.path.abspath("./src/garnet/workflow.py")
    command = ["python", script]

    with tempfile.TemporaryDirectory() as tmpdir:

        os.chdir(tmpdir)

        for inst in garnet.workflow.inst_dict.keys():

            fname = inst + ".yaml"
            subprocess.run(command + [fname, "temp", inst])

            assert os.path.exists(os.path.join(tmpdir, fname))
