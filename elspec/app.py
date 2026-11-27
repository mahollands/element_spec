"""
Command line handling and program start-up
"""
import argparse
import sys

from .constants import DEFAULT_SESSION, LOGA_INIT
from .data import load_spectra
from .linelist import available_ions
from .params import GLOBAL_PARAMS, ION_PARAMS
from .session import load_session


def get_args():
    parser = argparse.ArgumentParser(
        prog="element_spec",
        description="Interactive LTE line identification. Parameters given "
        "here are only starting values (overriding the session file): all can "
        "be changed in the window.",
    )
    parser.add_argument(
        "fnames",
        type=str,
        nargs="*",
        help="Text file(s) with columns of wavelength [Å] and flux (further "
        "columns are ignored). Omit to show the models alone.",
    )
    parser.add_argument(
        "--wave",
        type=str,
        default=None,
        choices=["air", "vac"],
        help=r"Initial wavelength convention (air/vac) for the models, to "
        "match the data (default air)",
    )
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"Session file storing saved ions and global parameters "
        f"(default {DEFAULT_SESSION})",
    )
    parser.add_argument(
        "--ion",
        default="CaII",
        help="Initial ion (default CaII)",
    )
    parser.add_argument(
        "--emission",
        action="store_true",
        help="Start in emission mode",
    )
    parser.add_argument(
        "--logA",
        type=float,
        default=None,
        help=f"Initial log10 abundance (default: the saved value if the ion "
        f"is saved, otherwise {LOGA_INIT})",
    )
    # Defaults of None, so that values from the session file are only
    # overridden when given explicitly.
    for spec in GLOBAL_PARAMS + ION_PARAMS:
        parser.add_argument(
            f"--{spec.name}",
            type=int if spec.is_int else float,
            default=None,
            help=f"{spec.tooltip} (default {spec.default})",
        )
    return parser.parse_args()


def initial_values(args, session_globals):
    """
    Starting values for the GUI parameters. Global parameters come from the
    command line, then the session file, then defaults. Per-ion parameters
    (logA and ION_PARAMS) are None unless given on the command line: the GUI
    then uses the saved values if the starting ion is saved, else defaults.
    """
    initial = {
        "wave": args.wave or session_globals.get("wave", "air"),
        "ion": args.ion,
        "emission": args.emission,
        "logA": args.logA,
    }
    for spec in GLOBAL_PARAMS:
        cli_value = getattr(args, spec.name)
        if cli_value is not None:
            initial[spec.name] = cli_value
        else:
            initial[spec.name] = session_globals.get(spec.name, spec.default)
    for spec in ION_PARAMS:
        initial[spec.name] = getattr(args, spec.name)
    return initial


def main():
    args = get_args()
    session_globals, saved = load_session(args.session)
    initial = initial_values(args, session_globals)

    if args.ion not in available_ions():
        sys.exit(f"No atomic data for {args.ion}\nAvailable Ions:\n"
                 + ", ".join(available_ions()))
    try:
        S = load_spectra(args.fnames)
    except (FileNotFoundError, ValueError) as err:
        sys.exit(str(err))

    # Imported here so that errors above (and --help) don't wait for Qt
    from .gui import run_gui
    return run_gui(S, bool(args.fnames), initial, saved, args.session)
