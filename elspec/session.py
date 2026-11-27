"""
The session file: the saved ion set and global parameters, stored in the
working directory so each data directory keeps its own.
"""
import json
import os.path

from .model import IonModel


def load_session(fname):
    """
    Returns (global parameters dict, list of IonModel). Both are empty if
    the file does not exist.
    """
    if not os.path.exists(fname):
        return {}, []
    with open(fname) as F:
        session = json.load(F)
    ions = [IonModel.from_dict(d) for d in session.get("ions", [])]
    return session.get("globals", {}), ions


def write_session(fname, global_params, ions):
    """
    Write the session (atomically, so an interrupted write can't corrupt it)
    """
    session = {
        "globals": global_params,
        "ions": [im.to_dict() for im in ions],
    }
    tmp = fname + ".tmp"
    with open(tmp, "w") as F:
        json.dump(session, F, indent=2)
    os.replace(tmp, fname)
