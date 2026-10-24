"""Blocks and logs every connection that is not on this machine (test helper for A13).

Python imports this file at start-up when its folder is on PYTHONPATH, and so does every Python
subprocess the program starts. A connection to an address other than loopback, a unix socket or a
name lookup other than localhost is refused and written to the file named by SOCKET_LOG.
"""

import json
import os
import socket

LOG = os.environ.get("SOCKET_LOG")
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo
LOCAL = {"127.0.0.1", "::1", "localhost", "", None}


def _note(kind, target):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "kind": kind, "target": str(target)}) + "\n")


def _local(address):
    if isinstance(address, (str, bytes)):  # a unix socket path
        return True
    return address[0] in LOCAL


def connect(self, address):
    if not _local(address):
        _note("connect", address)
        raise OSError("outbound connections are blocked")
    return _real_connect(self, address)


def connect_ex(self, address):
    if not _local(address):
        _note("connect_ex", address)
        return 111
    return _real_connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    if host not in LOCAL:
        _note("getaddrinfo", host)
        raise OSError("name lookups are blocked")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
