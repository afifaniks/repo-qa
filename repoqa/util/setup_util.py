import socket
import sys


def setup():
    host_name = socket.gethostname()

    if host_name == "hestia":
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
