from __future__ import absolute_import


def check_for_mpi_relaunch(argv):
    if argv[1] != "--mpi-relaunch":
        return

    from pickle import loads
    f, args, kwargs = loads(argv[2])

    f(*args, **kwargs)
    import sys
    sys.exit()


def run_with_mpi_ranks(py_script, ranks, callable_, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}

    import sys
    import os
    newenv = os.environ.copy()
    newenv["PYTOOLS_RUN_WITHIN_MPI"] = "1"

    from pickle import dumps
    callable_and_args = dumps((callable_, args, kwargs))

    from subprocess import check_call
    check_call(["mpirun", "-np", str(ranks),
        sys.executable, py_script, "--mpi-relaunch", callable_and_args],
        env=newenv)
