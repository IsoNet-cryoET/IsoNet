"""OpenMPI, once intialized, prohibits forking. This helper module
allows the forking of *one* helper child process before OpenMPI
initializaton that can do the forking for the fork-challenged
parent process.

Since none of this is MPI-specific, it got parked in pytools.
"""
from __future__ import absolute_import


class ExecError(OSError):
    pass


class DirectForker(object):
    def __init__(self):
        self.apids = {}
        self.count = 0

    @staticmethod
    def call(cmdline, cwd=None):
        from subprocess import call as spcall

        try:
            return spcall(cmdline, cwd=cwd)
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                            % (" ".join(cmdline), e))

    def call_async(self, cmdline, cwd=None):
        from subprocess import Popen

        try:
            self.count += 1

            proc = Popen(cmdline, cwd=cwd)
            self.apids[self.count] = proc

            return self.count
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                             % (" ".join(cmdline), e))

    @staticmethod
    def call_capture_output(cmdline, cwd=None, error_on_nonzero=True):
        from subprocess import Popen, PIPE

        try:
            popen = Popen(cmdline, cwd=cwd, stdin=PIPE, stdout=PIPE,
                          stderr=PIPE)
            stdout_data, stderr_data = popen.communicate()

            if error_on_nonzero and popen.returncode:
                raise ExecError("status %d invoking '%s': %s"
                                % (popen.returncode, " ".join(cmdline),
                                   stderr_data.decode("utf-8", errors="replace")))

            return popen.returncode, stdout_data, stderr_data
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                            % (" ".join(cmdline), e))

    def wait(self, aid):
        proc = self.apids.pop(aid)
        retc = proc.wait()

        return retc

    def waitall(self):
        rets = {}

        for aid in list(self.apids):
            rets[aid] = self.wait(aid)

        return rets


def _send_packet(sock, data):
    from struct import pack
    from six.moves.cPickle import dumps

    packet = dumps(data)

    sock.sendall(pack("I", len(packet)))
    sock.sendall(packet)


def _recv_packet(sock, who="Process", partner="other end"):
    from struct import calcsize, unpack
    size_bytes_size = calcsize("I")
    size_bytes = sock.recv(size_bytes_size)

    if len(size_bytes) < size_bytes_size:
        from warnings import warn
        warn("%s exiting upon apparent death of %s" % (who, partner))

        raise SystemExit

    size, = unpack("I", size_bytes)

    packet = b""
    while len(packet) < size:
        packet += sock.recv(size)

    from six.moves.cPickle import loads
    return loads(packet)


def _fork_server(sock):
    # Ignore keyboard interrupts, we'll get notified by the parent.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Construct a local DirectForker to do the dirty work
    df = DirectForker()

    funcs = {
        "call": df.call,
        "call_async": df.call_async,
        "call_capture_output": df.call_capture_output,
        "wait": df.wait,
        "waitall": df.waitall
    }

    try:
        while True:
            func_name, args, kwargs = _recv_packet(
                sock, who="Prefork server", partner="parent"
            )

            if func_name == "quit":
                df.waitall()
                _send_packet(sock, ("ok", None))
                break
            else:
                try:
                    result = funcs[func_name](*args, **kwargs)
                # FIXME: Is catching all exceptions the right course of action?
                except Exception as e:  # pylint:disable=broad-except
                    _send_packet(sock, ("exception", e))
                else:
                    _send_packet(sock, ("ok", result))
    finally:
        sock.close()

    import os
    os._exit(0)  # pylint:disable=protected-access


class IndirectForker(object):
    def __init__(self, server_pid, sock):
        self.server_pid = server_pid
        self.socket = sock

        import atexit
        atexit.register(self._quit)

    def _remote_invoke(self, name, *args, **kwargs):
        _send_packet(self.socket, (name, args, kwargs))
        status, result = _recv_packet(
            self.socket, who="Prefork client", partner="prefork server"
        )

        if status == "exception":
            raise result

        assert status == "ok"
        return result

    def _quit(self):
        self._remote_invoke("quit")

        from os import waitpid
        waitpid(self.server_pid, 0)

    def call(self, cmdline, cwd=None):
        return self._remote_invoke("call", cmdline, cwd)

    def call_async(self, cmdline, cwd=None):
        return self._remote_invoke("call_async", cmdline, cwd)

    def call_capture_output(self, cmdline, cwd=None, error_on_nonzero=True):
        return self._remote_invoke("call_capture_output", cmdline, cwd,
                                   error_on_nonzero)

    def wait(self, aid):
        return self._remote_invoke("wait", aid)

    def waitall(self):
        return self._remote_invoke("waitall")


forker = DirectForker()


def enable_prefork():
    global forker  # pylint:disable=global-statement

    if isinstance(forker, IndirectForker):
        return

    from socket import socketpair
    s_parent, s_child = socketpair()

    from os import fork
    fork_res = fork()

    # Child
    if fork_res == 0:
        s_parent.close()
        _fork_server(s_child)
    # Parent
    else:
        s_child.close()
        forker = IndirectForker(fork_res, s_parent)


def call(cmdline, cwd=None):
    return forker.call(cmdline, cwd)


def call_async(cmdline, cwd=None):
    return forker.call_async(cmdline, cwd)


def call_capture_output(cmdline, cwd=None, error_on_nonzero=True):
    return forker.call_capture_output(cmdline, cwd, error_on_nonzero)


def wait(aid):
    return forker.wait(aid)


def waitall():
    return forker.waitall()
