#!/usr/bin/env python
import argparse
import glob
from functools import reduce, partial
import operator
import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from numba import vectorize, float64

try:
    from mh.spectra import Spectrum, vac_to_air
except ImportError:
    from spectra import Spectrum, vac_to_air

# If this program is run via a softlink, realpath resolves this, and so the
# data associated with this program can be located.
INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))

rcParams["text.usetex"] = False

###############################################################################
# Arg parse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fnames",
        type=str,
        help=r"File with spectral data (type None if no data)",
    )
    parser.add_argument(
        "El",
        type=str,
        help=r"Element to display",
    ),
    parser.add_argument(
        "Teff",
        type=float,
        help=r"Effective temperature [K]",
    )
    parser.add_argument(
        "Au",
        type=float,
        help=r"Abundance in arbitrary units",
    )
    parser.add_argument(
        "-rv",
        type=float,
        default=0.0, help=r"Radial velocity [km/s]",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help=r"Rescale model by factor",
    )
    parser.add_argument(
        "--res",
        type=float,
        default=2.0,
        help=r"Model resolution [Å] (default=2.0)",
    )
    parser.add_argument(
        "-wl",
        type=float,
        default=0.5,
        help=r"Lorentzian width [Å] at a reference wavelength of 5000 Å (default=0.5)",
    )
    parser.add_argument(
        "-gb",
        type=float,
        default=-1.0,
        help=r"Gaussian blur data [Å]",
    )
    parser.add_argument(
        "-dx",
        type=float,
        default=0.1,
        help=r"Wavelength sampling for model spectrum [Å]",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help=r"model: use if the input a model (doesn't have errors)",
    )
    parser.add_argument(
        "--emission",
        action="store_true",
        help=r"emission: use to specify emission lines",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=500,
        help=r"N strongest lines used",
    )
    parser.add_argument(
        "--wave",
        type=str,
        default="air",
        choices=["air", "vac"],
        help=r"Wavelengths (air/vac)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=r"Write 'model' to disk",
    )
    parser.add_argument(
        "--noread",
        dest="read",
        action="store_false",
        help="Ignore disk models",
    )
    args = parser.parse_args()
    if args.model:
        args.wave = "vac"
    if args.wl < 3*args.dx:
        raise ValueError("Lines will be undersampled: Increase wl/decrease dx")
    return args

###############################################################################
# methods


@vectorize([float64(float64, float64, float64)], cache=True)
def lorentzian(x, x0, w):
    """
    Unit-normed Lorentzian profile. w=FWHM
    """
    xx = 2 * (x - x0) / w
    if abs(xx) > 20:
        return 0
    return 1 / (np.pi * w * (1 + xx * xx))


# @jit(nopython=True, cache=True)
def line_profile(x, linedata, wl):
    """
    Creates line profile for a single line
    """
    V = lorentzian(x, linedata.lam, wl*(linedata.lam/5000)**2)
    return linedata.strength * V


def model_abs(p, x, Linedata):
    """
    Creates absorption profile from combination of lines
    """
    A, wl = p
    kappa = sum(line_profile(x, linedata, wl) for linedata in Linedata.itertuples())
    return np.exp(-A * kappa)


def model_em(p, x, Linedata):
    """
    Creates emission profile from combination of lines
    """
    A, wl = p
    LL = sum(line_profile(x, linedata, wl) for linedata in Linedata.itertuples())
    return A * LL


def create_model(args, Linedata):
    """
    Generate model with lines from specified Ion at specified Teff
    """
    xm = np.arange(*S.x01, args.dx)
    model_fun = model_em if args.emission else model_abs
    ym = model_fun((args.Au, args.wl), xm, Linedata)
    M = Spectrum(xm, ym, 0, wave=args.wave, y_unit="")
    M = M.convolve_gaussian(args.res)
    M += 1e-300  # Needed to deal with numerical issues with very strong lines after convolution
    return M

###############################################################################

def load_spec(fname, args):
    """
    Loads an input spectrum
    """
    if not args.model:
        return Spectrum.from_txt(fname, wave=args.wave)
    if fname.endswith(".dk"):
        S = Spectrum.from_dk(fname)
    else:
        S = Spectrum.from_txt(fname, model=True)
    S.e = np.abs(S.y / 100)
    return S


def load_spectra(args):
    """
    Loads the input spectra
    """
    if args.fnames == "None":
        x = np.arange(1000, 10000, 0.5)
        S = Spectrum(x, 0, 0)
    else:
        SS = [load_spec(f, args) for f in args.fnames.split(",")]
        S = Spectrum.join(SS, sort=True)
        if args.gb > 0:
            S = S.convolve_gaussian(args.gb)
    return S


def load_previous_models(M, args):
    """
    Load and combine previously generated absorption
    profiles into a total profile
    """
    flist_abs, flist_em = glob.glob("LTE*[0-9].npy"), glob.glob("LTE*emission.npy")
    el_prefix = f"LTE-{args.El}-"

    if args.emission:
        flist_em = [f for f in flist_em if not f.startswith(el_prefix)]
    else:
        flist_abs = [f for f in flist_abs if not f.startswith(el_prefix))
    if not (flist_abs or flist_em):
        return None

    # Load absorption and emission profiles
    load_npy = partial(Spectrum.from_npy, wave=args.wave, y_unit="")
    MMr_abs = map(load_npy, flist_abs)
    MMr_em = map(load_npy, flist_em)

    # combime abs and em profiles
    unit_spec = Spectrum(M.x, 1, 0, wave=args.wave, y_unit="")
    Mr = reduce(operator.mul, MMr_abs, unit_spec) + sum(MMr_em)
    return Mr * args.scale


def setup_Linedata(args):
    Linedata = np.load(f"{INSTALL_DIR}/linelist.rec.npy")
    Linedata = pd.DataFrame(Linedata)

    all_ions = np.unique(Linedata["ion"])
    ionmatch = Linedata["ion"] == args.El.encode()
    Linedata = Linedata[ionmatch]
    if Linedata.empty:
        print(f"Could not find atomic data for {args.El}")
        print("Available Ions:")
        print(*[ion.decode() for ion in all_ions])
        sys.exit()

    # Apply velocity and air shifts to atomic data
    beta_v = args.rv / 2.998e5
    Linedata["lam"] *= np.sqrt((1 + beta_v) / (1 - beta_v))
    if args.wave == "air":
        Linedata["lam"] = vac_to_air(Linedata["lam"])

    # Only use lines in data range
    validwave = (Linedata["lam"] > S.x[0]) & (Linedata["lam"] < S.x[-1])
    Linedata = Linedata[validwave]
    if Linedata.empty:
        sys.exit("No {} lines in the range {:.1f} -- {:.1f}A".format(args.El, *S.x01))

    beta = 1.43877688 / args.Teff # 1/(kb*T) in cm
    dE = 1e8 / Linedata["lam"]
    # Only use N strongest lines (and set strength column)
    if args.emission:
        E_hi = Linedata["E_low"] + dE
        boltz = np.exp(-beta * E_hi)
        A_ki = 10 ** Linedata["loggf"] * (5000/Linedata["lam"])**3
        Linedata["strength"] = A_ki * boltz
    else:
        boltz = np.exp(-beta * Linedata["E_low"])
        gf = 10 ** (Linedata["loggf"])
        stim = -np.expm1(-beta*dE)
        Linedata["strength"] = gf * boltz * stim * (Linedata["lam"]/5000)**2
    Linedata.sort_values("strength", ascending=False, inplace=True)
    Linedata = Linedata[: args.N]

    return Linedata


def make_figure(S, M, args):
    M *= args.scale
    plt.figure(figsize=(12, 6))
    S.plot(c="grey", drawstyle="steps-mid", zorder=1)
    if args.emission:
        (1 + M).plot("r-", zorder=3)
    else:
        M.plot("r-", zorder=3)
    if args.read:
        Mr = load_previous_models(M, args)
        if Mr is not None:
            Mr.plot("C0-", zorder=2)
    plt.xlim(*S.x01)
    plt.ylim(0, 1.2 * args.scale)
    plt.xlabel(r"Wavelength [Å]")
    plt.ylabel(r"Normalised flux")
    plt.tight_layout()
    plt.show()


###############################################################################

if __name__ == "__main__":
    args = get_args()
    S = load_spectra(args)
    Linedata = setup_Linedata(args)
    M = create_model(args, Linedata)
    if args.write:
        outname = f"LTE-{args.El}-{args.Teff:.0f}"
        outname += "_emission.npy" if args.emission else ".npy"
        M.write(outname)
    else:
        make_figure(S, M, args)
