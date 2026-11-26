#!/usr/bin/env python
"""
Tool for interactively identifying elements within a spectrum. The abundance is
set with a slider (log10 scale). After the figure is closed, the user is
prompted whether to write the model (at the final slider abundance) to disk.
"""
import argparse
import glob
from functools import reduce, partial
import operator
import os.path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import Slider
import pandas as pd
from numba import vectorize, float64

try:
    from mh.spectra import Spectrum, vac_to_air, logarange
except ImportError:
    from spectra import Spectrum, vac_to_air, logarange

# If this program is run via a softlink, realpath resolves this, and so the
# data associated with this program can be located.
INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))

rcParams["text.usetex"] = False

LOGA_MIN, LOGA_MAX, LOGA_INIT = -5.0, 10.0, 0.0
LIVE_UPDATE_MAX_TIME = 0.2  # [s] slower models only update on slider release

###############################################################################
# Arg parse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fnames",
        type=str,
        nargs="*",
        help=r"File(s) with spectral data (omit to show the model alone)",
    )
    parser.add_argument(
        "Ion",
        type=str,
        help=r"Ion to display",
    )
    parser.add_argument(
        "Teff",
        type=float,
        help=r"Effective temperature [K]",
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
        default=5.0,
        help=r"Log-wavelength sampling for model spectrum [km/s]",
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
        "--noread",
        dest="read",
        action="store_false",
        help="Ignore disk models",
    )
    args = parser.parse_args()
    if args.model:
        args.wave = "vac"
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
    return 2 / (np.pi * w * (1 + xx * xx))


def line_profile(x, linedata, wl):
    """
    Creates line profile for a single line
    """
    V = lorentzian(x, linedata.lam, wl*(linedata.lam/5000)**2)
    return linedata.strength * V


def compute_opacity(S, args, Linedata):
    """
    Sum of line profiles (unit abundance) on the model wavelength grid. This is
    independent of abundance, so is only computed once.
    """
    min_line_width = args.wl * (S.x[0]/5000)**2
    dx = min(args.dx, 2.998e5 * min_line_width / (2*S.x[0]))
    xm = logarange(*S.x01, 2.998e5/dx)
    kappa = sum(line_profile(xm, linedata, args.wl) for linedata in Linedata.itertuples())
    return xm, kappa


class ModelBuilder:
    """
    Builds the convolved model spectrum for a given abundance from the
    precomputed opacity.
    """
    def __init__(self, xm, kappa, args):
        self.xm = xm
        self.kappa = kappa
        self.args = args
        if args.emission:
            # Emission is linear in abundance, so convolve once and rescale
            self._em_unit = self._convolve(kappa)

    def _convolve(self, y):
        M = Spectrum(self.xm, y, 0, wave=self.args.wave, y_unit="")
        return M.convolve_gaussian(self.args.res)

    def __call__(self, logA):
        A = 10**logA
        if self.args.emission:
            M = self._em_unit * A
        else:
            M = self._convolve(np.exp(-A * self.kappa))
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
    if not args.fnames:
        x = np.linspace(1000, 10000, 2)
        S = Spectrum(x, 0, 0)
    else:
        missing = [f for f in args.fnames if not os.path.isfile(f)]
        if missing:
            sys.exit("Could not find file(s): " + ", ".join(missing))
        SS = [load_spec(f, args) for f in args.fnames]
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
    el_prefix = f"LTE-{args.Ion}-"

    if args.emission:
        flist_em = [f for f in flist_em if not f.startswith(el_prefix)]
    else:
        flist_abs = [f for f in flist_abs if not f.startswith(el_prefix)]
    if not (flist_abs or flist_em):
        return None

    # Load absorption and emission profiles
    load_npy = partial(Spectrum.from_npy, wave=args.wave, y_unit="")
    MM_prev_abs = map(load_npy, flist_abs)
    MM_prev_em = map(load_npy, flist_em)

    # combime abs and em profiles
    unit_spec = Spectrum(M.x, 1, 0, wave=args.wave, y_unit="")
    M_prev = reduce(operator.mul, MM_prev_abs, unit_spec) + sum(MM_prev_em)
    return M_prev * args.scale


def setup_Linedata(S, args):
    Linedata = np.load(f"{INSTALL_DIR}/linelist.rec.npy")
    Linedata = pd.DataFrame(Linedata)

    all_ions = np.unique(Linedata["ion"])
    ionmatch = Linedata["ion"] == args.Ion.encode()
    Linedata = Linedata[ionmatch]
    if Linedata.empty:
        ions = ', '.join(ion.decode() for ion in all_ions)
        sys.exit(f"No atomic data for {args.Ion}\nAvailable Ions:\n{ions}")


    # Apply velocity and air shifts to atomic data
    beta_v = args.rv / 2.998e5
    Linedata["dE"] = 1e8 / Linedata["lam"]
    Linedata["lam"] *= np.sqrt((1 + beta_v) / (1 - beta_v))
    if args.wave == "air":
        Linedata["lam"] = vac_to_air(Linedata["lam"])

    # Only use lines in data range
    validwave = (Linedata["lam"] > S.x[0]) & (Linedata["lam"] < S.x[-1])
    Linedata = Linedata[validwave]
    if Linedata.empty:
        sys.exit("No {} lines in the range {:.1f} -- {:.1f}A".format(args.Ion, *S.x01))

    beta = 1.43877688 / args.Teff # 1/(kb*T) in cm
    # Only use N strongest lines (and set strength column)
    if args.emission:
        E_hi = Linedata["E_low"] + Linedata["dE"]
        boltz = np.exp(-beta * E_hi)
        A_ki = 10 ** Linedata["loggf"] * (5000/Linedata["lam"])**3
        Linedata["strength"] = A_ki * boltz
    else:
        boltz = np.exp(-beta * Linedata["E_low"])
        gf = 10 ** (Linedata["loggf"])
        stim = -np.expm1(-beta*Linedata["dE"])
        Linedata["strength"] = gf * boltz * stim * (Linedata["lam"]/5000)**2
    Linedata.sort_values("strength", ascending=False, inplace=True)
    Linedata = Linedata[: args.N]

    return Linedata


def display_y(M, args):
    """
    y-values of the model as plotted (scaled, and offset for emission)
    """
    y = M.y * args.scale
    return 1 + y if args.emission else y


def interactive_figure(S, build_model, args):
    """
    Plot the data and model with an abundance slider. Returns the final
    log10 abundance when the figure is closed.
    """
    t0 = time.perf_counter()
    M = build_model(LOGA_INIT)
    build_time = time.perf_counter() - t0

    fig =plt.figure(f"{args.Ion} {args.Teff:.0f}K", figsize=(12, 6.5))
    ax = fig.add_axes([0.07, 0.17, 0.90, 0.78])
    ax_slider = fig.add_axes([0.12, 0.04, 0.76, 0.03])

    plt.sca(ax)
    S.plot(c="grey", drawstyle="steps-mid", zorder=1)
    (model_line,) = ax.plot(M.x, display_y(M, args), "r-", zorder=3)
    if args.read:
        M_prev = load_previous_models(M, args)
        if M_prev is not None:
            ax.plot(M_prev.x, M_prev.y, "C0-", zorder=2)
    ax.set_xlim(*S.x01)
    ax.set_ylim(0, 1.2 * args.scale)
    ax.set_xlabel(r"Wavelength [Å]")
    ax.set_ylabel(r"Normalised flux")

    slider = Slider(
        ax_slider,
        r"log$_{10}$(A)",
        LOGA_MIN,
        LOGA_MAX,
        valinit=LOGA_INIT,
        valstep=0.05,
    )

    def update(logA):
        model_line.set_ydata(display_y(build_model(logA), args))
        fig.canvas.draw_idle()

    if build_time < LIVE_UPDATE_MAX_TIME:
        # Fast enough to redraw continuously while dragging
        slider.on_changed(update)
    else:
        # Slow models (e.g. large wavelength ranges) only update on release,
        # otherwise dragging queues up many expensive recomputations.
        shown = {"logA": slider.val}

        def on_release(event):
            if slider.val != shown["logA"]:
                shown["logA"] = slider.val
                update(slider.val)

        fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show()
    return slider.val


def prompt_write(build_model, logA, args):
    """
    Ask whether to save the model at the final abundance
    """
    outname = f"LTE-{args.Ion}-{args.Teff:.0f}"
    outname += "_emission.npy" if args.emission else ".npy"
    exists = " (overwrites existing file)" if os.path.exists(outname) else ""
    try:
        reply = input(f"Write model with log10(A)={logA:.2f} to {outname}{exists}? [y/N] ")
    except EOFError:
        reply = ""
    if reply.strip().lower() in ("y", "yes"):
        build_model(logA).write(outname)
        print(f"Wrote {outname}")


###############################################################################

if __name__ == "__main__":
    args = get_args()
    S = load_spectra(args)
    Linedata = setup_Linedata(S, args)
    xm, kappa = compute_opacity(S, args, Linedata)
    build_model = ModelBuilder(xm, kappa, args)
    logA = interactive_figure(S, build_model, args)
    prompt_write(build_model, logA, args)
