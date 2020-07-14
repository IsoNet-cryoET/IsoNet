"""See pytools.prefork for this module's reason for being."""
from __future__ import absolute_import

import mpi4py.rc  # pylint:disable=import-error
mpi4py.rc.initialize = False

from mpi4py.MPI import *  # noqa pylint:disable=wildcard-import,wrong-import-position

import pytools.prefork  # pylint:disable=wrong-import-position
pytools.prefork.enable_prefork()


if Is_initialized():  # noqa pylint:disable=undefined-variable
    raise RuntimeError("MPI already initialized before MPI wrapper import")


def InitWithAutoFinalize(*args, **kwargs):  # noqa
    result = Init(*args, **kwargs)  # noqa pylint:disable=undefined-variable
    import atexit
    atexit.register(Finalize)  # noqa pylint:disable=undefined-variable
    return result
