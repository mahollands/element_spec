#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from mh.spectra import *
from mh.spectra.misc import black_body
import argparse
import glob
from functools import reduce
import operator
from numba import jit, vectorize, float64
import os.path

#If this program is run via a softlink, realpath resolves this, and so the
#data associated with this program can be located.
INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))

#..............................................................................
# Arg parse

parser = argparse.ArgumentParser()
parser.add_argument("fnames", type=str, \
  help="File with spectral data")
parser.add_argument("El", type=str, \
  help="Element to display")
parser.add_argument("Teff", type=float, \
  help="Effective temperature [K]")
parser.add_argument("Au", type=float, \
  help="Abundance in arbitrary units")
parser.add_argument("-rv", type=float, default=0., \
  help="Radial velocity [km/s]")
parser.add_argument("-s", "--scale", type=float, default=1., \
  help="Rescale model by factor")
parser.add_argument("--res", type=float, default=2.0, \
  help="Model resolution [\AA] (default=2.0)")
parser.add_argument("-wl", type=float, default=0.5, \
  help="Lorentzian width [\AA] (default=0.5)")
parser.add_argument("-gb", type=float, default=-1.0, \
  help="Gaussian blur data [\AA]")
parser.add_argument("--norm", type=str, default="BB", choices=["BB","unit"], \
  help="normalisation: BB (def), unit")
parser.add_argument("--model", action="store_const", const=True, \
  help="model: use if the input a model (doesn't have errors)")
parser.add_argument("-N", type=int, default=500, \
  help="N strongest lines used")
parser.add_argument("--wave", type=str, default="air", choices=["air","vac"], \
  help="Wavelengths (air/vac)")
parser.add_argument("--write", action="store_const", const=True, \
  help="Write 'model' to disk")
parser.add_argument("--noread", dest="read", action="store_const", const=False, default=True, \
  help="Ignore disk models")
args = parser.parse_args()

beta = 1/(0.695*args.Teff)

#.............................................................................
# methods

@vectorize([float64(float64, float64, float64)], cache=True)
def lorentzian(x, x0, w):
  """
  Unit-normed Lorentzian profile. w=FWHM
  """
  return 1/(np.pi*w*(1+(0.5*(x-x0)/w)**2))

@jit(nopython=True, cache=True)
def line_profile(x, linedata, wl):
  """
  Creates line profile for a single line
  """
  boltz = math.exp(-beta*linedata['E_low'])

  gf = 10**(linedata['loggf'])
  V = lorentzian(x, linedata['lambda'], wl)
  return gf * boltz * V

def model(p, x):
  """
  Creates absorption profile from combination of lines
  """
  A, wl = p
  LL = sum(line_profile(x, linedata, wl) for linedata in Linedata)
  return np.exp(-A*LL)

def normalise(M, S, args):
  if args.norm == "BB":
    BB = Black_body(M.x, args.Teff, args.wave, S.x_unit, S.y_unit)
    M *= BB
    #M *= Black_body(M.x, args.Teff, args.wave, S.x_unit, S.y_unit)
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
  try:
    if args.model:
      skip = 55 if fname.endswith(".dk") else 0
      M = model_from_txt(fname, skiprows=skip)
      M.e = np.abs(M.y/100)
      return M
    else:
      return spec_from_txt(fname, wave=args.wave)
  except IOError:
    print("Could not find file: {}".format(fname))
    exit()
  except ValueError:
    print("Could not parse file: {}".format(fname))
    exit()

def load_previous_models(S, M, args):
  flist = glob.glob("LTE*.npy")
  if len(flist) == 0:
    return None
  else:
    MMr = [spec_from_npy(fname, args.wave, y_unit="") for fname in flist \
      if not fname.startswith("LTE-{}-".format(args.El))] #ignore current element
    Mr = reduce(operator.mul, MMr)
    assert len(Mr) == len(M), "Length of loaded models does not match data"
    Mr = normalise(Mr, S, args)
    return Mr

#...........................................................

if args.model:
  args.wave = 'vac'

SS = [load_spec(f, args) for f in args.fnames.split(',')]
S = join_spectra(SS, sort=True)

if args.gb > 0.:
  S = S.convolve_gaussian(args.gb)

#Create linelist
Linedata = np.load(f"{INSTALL_DIR}/linelist.rec.npy")
all_ions = np.unique(Linedata['ion'])
ionmatch = Linedata['ion'] == args.El.encode()
Linedata = Linedata[ionmatch]
if len(Linedata) == 0:
  print("Could not find atomic data for {}".format(args.El))
  print("Available Ions:")
  print(*[ion.decode() for ion in all_ions])
  exit()

#Change to air wavelengths
if args.wave == "air":
  Linedata['lambda'] = vac_to_air(Linedata['lambda'])

#Only use lines in data range
validwave = (Linedata['lambda'] > S.x[0]) & (Linedata['lambda'] < S.x[-1])
Linedata = Linedata[validwave]
if len(Linedata) == 0:
  print("No {} lines in the range {:.1f} -- {:.1f}A".format(args.El, S.x[0], S.x[-1]))
  exit()

#Only use N strongest lines
boltz = np.exp(-beta*Linedata['E_low'])
gf = 10**(Linedata['loggf'])
linestrength = gf * boltz
strongest = np.argsort(linestrength)[-args.N:]
Linedata = Linedata[strongest]

#Generate model with lines from specified Ion at specified Teff
xm = np.arange(S.x[0], S.x[-1], 0.1)
ym = model((args.Au, args.wl), xm)
M = Spectrum(xm, ym, np.zeros_like(xm), wave=args.wave, y_unit="")
M.apply_redshift(args.rv)
M = M.convolve_gaussian(args.res)
M += 1E-300 #Needed to deal with numerical issues with very strong lines after convolution

if args.write:
  M.write("LTE-{}-{:.0f}.npy".format(args.El, args.Teff))
else:
  #Make plot
  M = normalise(M, S, args)
  plt.figure(figsize=(12, 6))
  S.plot(c='grey', drawstyle='steps-mid', zorder=1)
  M.plot('r-', zorder=3)
  if args.read:
    Mr = load_previous_models(S, M, args)
    if Mr is not None:
      Mr.plot('C0-', zorder=2)
  plt.xlim(S.x[0], S.x[-1])
  plt.ylim(0, 1.2*np.percentile(S.y, 99))
  plt.xlabel("Wavelength [\AA]")
  plt.ylabel("Normalised flux")
  plt.tight_layout()
  plt.show()
