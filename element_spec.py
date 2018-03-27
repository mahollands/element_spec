#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from mh.spectra import *
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
import argparse

INSTALL_DIR = "/home/astro/phujdu/Software/element_spec/"

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
parser.add_argument("-wl", type=float, default=2.0, \
  help="Lorentzian width [\AA] (default=2.0)")
parser.add_argument("-gb", type=float, default=-1.0, \
  help="Gaussian blur spectrum [\AA]")
parser.add_argument("--norm", type=str, default="BB", choices=["BB","unit"], \
  help="normalisation: BB (def), unit")
parser.add_argument("--model", type=bool, default=False, \
  help="model: True/False (is the input a model)")
parser.add_argument("-N", type=int, default=500, \
  help="N strongest lines used")
parser.add_argument("--wave", type=str, default="air", choices=["air","vac"], \
  help="Wavelengths (air/vac)")
args = parser.parse_args()

beta = 1/(0.695*args.Teff)

#.............................................................................
# methods

def line_profile(x, linedata, res, wl):
  boltz = math.exp(-beta*linedata['E_low'])
  gf = 10**(linedata['loggf'])
  V = voigt(x, linedata['lambda'], res, wl)
  return  gf * boltz * V

def model(p, x):
  A, res, wl = p
  LL = sum(line_profile(x, linedata, res, wl) for linedata in Linedata)
  return np.exp(-A*LL)

#.............................................................................

#Load spectrum
def load_spec(fname):
  try:
    if args.model:
      if fname.endswith(".dk"):
        return model_from_txt(fname, skiprows=55)
      else:
        return model_from_txt(fname)
    else:
      return spec_from_txt(fname)
  except IOError:
    print("Could not find file: {}".format(fname))
    exit()
  except ValueError:
    print("Could not parse file: {}".format(fname))
    exit()

SS = [load_spec(f) for f in args.fnames.split(',')]
S = join_spectra(SS, sort=True)

if args.gb > 0.:
  S.convolve_gaussian(args.gb)

#Create linelist
Linedata = np.load(INSTALL_DIR+"linelist.rec.npy")
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
ym = model((args.Au, args.res, args.wl), xm)
M = Spectrum(xm, ym, np.zeros_like(xm))
M.apply_redshift(args.rv, args.wave)

#Normalisation
if args.norm == "BB":
  M *= black_body(xm, args.Teff)
  M = args.scale*M.scale_model(S)
elif args.norm == "unit":
  M *= args.scale

plt.figure(figsize=(12,6))
plt.plot(S.x, S.y, c='grey', drawstyle='steps-mid')
plt.plot(M.x, M.y, 'r-')
plt.xlim(S.x[0], S.x[-1])
plt.ylim(0, 1.2*np.percentile(S.y, 99))
plt.xlabel("Wavelength [\AA]")
plt.ylabel("Normalised flux")
plt.tight_layout()
plt.show()
