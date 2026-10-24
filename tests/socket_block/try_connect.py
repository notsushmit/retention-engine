"""A control: tries to reach a documentation address (192.0.2.1); the block must log it."""

import socket

try:
    socket.create_connection(("192.0.2.1", 80), timeout=1)
except OSError as e:
    print("refused:", e)
