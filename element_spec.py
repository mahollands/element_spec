#!/usr/bin/python3
import argparse
import glob
from functools import reduce, partial
import operator
import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import vectorize, float64
try:
    from mh.spectra import *
except ImportError:
    from spectra import *

#If this program is run via a softlink, realpath resolves this, and so the
#data associated with this program can be located.
INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))

#..............................................................................
# Arg parse

parser = argparse.ArgumentParser()
parser.add_argument("fnames", type=str, \
    help=r"File with spectral data")
parser.add_argument("El", type=str, \
    help=r"Element to display")
parser.add_argument("Teff", type=float, \
    help=r"Effective temperature [K]")
parser.add_argument("Au", type=float, \
    help=r"Abundance in arbitrary units")
parser.add_argument("-rv", type=float, default=0., \
    help=r"Radial velocity [km/s]")
parser.add_argument("-s", "--scale", type=float, default=1., \
    help=r"Rescale model by factor")
parser.add_argument("--res", type=float, default=2.0, \
    help=r"Model resolution [\AA] (default=2.0)")
parser.add_argument("-wl", type=float, default=0.5, \
    help=r"Lorentzian width [\AA] (default=0.5)")
parser.add_argument("-gb", type=float, default=-1.0, \
    help=r"Gaussian blur data [\AA]")
parser.add_argument("--norm", type=str, default="BB", choices=["BB", "unit"], \
    help=r"normalisation: BB (def), unit")
parser.add_argument("--model", action="store_const", const=True, \
    help=r"model: use if the input a model (doesn't have errors)")
parser.add_argument("--emission", action="store_const", const=True, \
    help=r"emission: use to specify emission lines")
parser.add_argument("-N", type=int, default=500, \
    help=r"N strongest lines used")
parser.add_argument("--wave", type=str, default="air", choices=["air", "vac"], \
    help=r"Wavelengths (air/vac)")
parser.add_argument("--write", action="store_const", const=True, \
    help=r"Write 'model' to disk")
parser.add_argument("--noread", dest="read", action="store_const", const=False, \
    default=True, help="Ignore disk models")
args = parser.parse_args()

#1/(kb*T) in cm-1
beta = 1/(0.695*args.Teff)
load_npy = partial(spec_from_npy, wave=args.wave, y_unit="")

#.............................................................................
# methods

@vectorize([float64(float64, float64, float64)], cache=True)
def lorentzian(x, x0, w):
    """
    Unit-normed Lorentzian profile. w=FWHM
    """
    return 1/(np.pi*w*(1+(2*(x-x0)/w)**2))

#@jit(nopython=True, cache=True)
def line_profile(x, linedata, wl):
    """
    Creates line profile for a single line
    """
    V = lorentzian(x, linedata.lam, wl)
    return linedata.strength * V

def model_abs(p, x):
    """
    Creates absorption profile from combination of lines
    """
    A, wl = p
    LL = sum(line_profile(x, linedata, wl) for linedata in Linedata.itertuples())
    return np.exp(-A*LL)

def model_em(p, x):
    """
    Creates emission profile from combination of lines
    """
    A, wl = p
    LL = sum(line_profile(x, linedata, wl) for linedata in Linedata.itertuples())
    return A*LL

def normalise(M, S, args):
    """
    Normalise absorption profile against input spectrum
    in some specified way
    """
    if args.norm == "BB":
        BB = Black_body(M.x, args.Teff, args.wave, S.x_unit, S.y_unit)
        M *= BB
        if args.model:
            M = args.scale*M.scale_model_to_model(S)
        else:
            M = args.scale*M.scale_model(S)
    elif args.norm == "unit":
        M *= args.scale
    else:
        raise ValueError
    return M

#.............................................................................

#Load spectrum
def load_spec(fname, args):
    """
    Loads the input spectrum
    """
    try:
        if args.model:
            M = model_from_dk(fname) if fname.endswith(".dk") else model_from_txt(fname)
            M.e = np.abs(M.y/100)
            return M
        return spec_from_txt(fname, wave=args.wave)
    except IOError:
        print(f"Could not find file: {fname}")
        sys.exit()
    except ValueError:
        print(f"Could not parse file: {fname}")
        sys.exit()

def load_previous_models(S, M, args):
    """
    Load and combine previously generated absorption
    profiles into a total profile
    """
    flist_abs, flist_em = glob.glob("LTE*[0-9].npy"), glob.glob("LTE*emission.npy")
    if (len(flist_abs), len(flist_em)) == (0, 0):
        return None
        
    el_prefix = f"LTE-{args.El}-"
    #Load absorption and emission profiles
    if args.emission:
        MMr_abs = map(load_npy, flist_abs)
        MMr_em = (load_npy(f) for f in flist_em if not f.startswith(el_prefix))
    else:
        MMr_abs = (load_npy(f) for f in flist_abs if not f.startswith(el_prefix))
        MMr_em = map(load_npy, flist_em)
        
    #combime abs and em profiles
    Mr = reduce(operator.mul, MMr_abs, 1) + sum(MMr_em)

    assert len(Mr) == len(M), "Length of loaded models does not match data"
    return normalise(Mr, S, args)

#...........................................................

if args.model:
    args.wave = 'vac'

SS = [load_spec(f, args) for f in args.fnames.split(',')]
S = join_spectra(SS, sort=True)

if args.gb > 0.:
    S = S.convolve_gaussian(args.gb)

#Create linelist
Linedata = np.load(f"{INSTALL_DIR}/linelist.rec.npy")
Linedata = pd.DataFrame(Linedata)
if args.emission:
    Linedata['E_hi'] = 1e8/Linedata['lam'] + Linedata['E_low']
    Linedata['A_ki'] = 1e8 * 10**Linedata['loggf'] / Linedata['lam']**2
all_ions = np.unique(Linedata['ion'])
ionmatch = Linedata['ion'] == args.El.encode()
Linedata = Linedata[ionmatch]
if len(Linedata) == 0:
    print(f"Could not find atomic data for {args.El}")
    print("Available Ions:")
    print(*[ion.decode() for ion in all_ions])
    sys.exit()

#Only use lines in data range
validwave = (Linedata['lam'] > S.x[0]) & (Linedata['lam'] < S.x[-1])
Linedata = Linedata[validwave]
if len(Linedata) == 0:
    print("No {} lines in the range {:.1f} -- {:.1f}A".format(args.El, *S.x01))
    sys.exit()

#Only use N strongest lines (and set strength column)
if args.emission:
    boltz = np.exp(-beta*Linedata['E_hi'])
    A_ki = Linedata['A_ki']
    linestrength = A_ki * boltz
else:
    boltz = np.exp(-beta*Linedata['E_low'])
    gf = 10**(Linedata['loggf'])
    linestrength = gf * boltz
Linedata['strength'] = linestrength
Linedata.sort_values('strength', ascending=False, inplace=True)
Linedata = Linedata[:args.N]

#Apply redshift to atomic data
beta_v = args.rv / 2.998e5
Linedata['lam'] *= np.sqrt((1+beta_v)/(1-beta_v))

#Change to air wavelengths
if args.wave == "air":
    Linedata['lam'] = vac_to_air(Linedata['lam'])

#Generate model with lines from specified Ion at specified Teff
xm = np.arange(*S.x01, 0.1)
model_fun = model_em if args.emission else model_abs
ym = model_fun((args.Au, args.wl), xm)
M = Spectrum(xm, ym, np.zeros_like(xm), wave=args.wave, y_unit="")
M = M.convolve_gaussian(args.res)
M += 1E-300 #Needed to deal with numerical issues with very strong lines after convolution

if args.write:
    outname = f"LTE-{args.El}-{args.Teff:.0f}"
    outname += "_emission.npy" if args.emission else ".npy"
    M.write(outname)
else:
    #Make plot
    M = normalise(M, S, args)
    plt.figure(figsize=(12, 6))
    S.plot(c='grey', drawstyle='steps-mid', zorder=1)
    if args.emission:
        (1+M).plot('r-', zorder=3)
    else:
        M.plot('r-', zorder=3)
    if args.read:
        Mr = load_previous_models(S, M, args)
        if Mr is not None:
            Mr.plot('C0-', zorder=2)
    plt.xlim(*S.x01)
    plt.ylim(0, 1.2*np.percentile(S.y, 99))
    plt.xlabel(r"Wavelength [\AA]")
    plt.ylabel(r"Normalised flux")
    plt.tight_layout()
    plt.show()
