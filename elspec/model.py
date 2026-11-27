"""
Line opacity and model spectra.

A model is built in stages, each cached by IonModel and only recomputed when
the parameters it depends on change:
    lines    select_lines        ion, emission, N, Teff, rv, wave
    opacity  compute_opacity     + wl, dx
    builder  ModelBuilder        + res
    model    ModelBuilder(logA)  + logA
"""
import copy
from dataclasses import dataclass

import numpy as np
from numba import vectorize, float64

from .constants import C_KMS, LAMBDA_REF, LOGA_INIT
from .linelist import select_lines, ModelError
from .spectrum import Spectrum, logarange


@dataclass(frozen=True)
class GlobalParams:
    """
    Parameters shared by all ions that affect the models (scale only affects
    display, so is not included).
    """
    Teff: float
    rv: float
    res: float
    dx: float
    wave: str


# Lorentzian profiles are truncated beyond this many FWHM from line centre
LORENTZ_CUTOFF_FWHM = 10.0


@vectorize([float64(float64, float64, float64)], cache=True)
def lorentzian(x, x0, w):
    """
    Unit-normed Lorentzian profile. w=FWHM. Truncated beyond
    LORENTZ_CUTOFF_FWHM.
    """
    xx = 2 * (x - x0) / w
    if abs(xx) > 2 * LORENTZ_CUTOFF_FWHM:
        return 0
    return 2 / (np.pi * w * (1 + xx * xx))


def line_width(lam, wl):
    """
    Lorentzian FWHM at wavelength lam, for FWHM wl at LAMBDA_REF
    """
    return wl * (lam / LAMBDA_REF) ** 2


def compute_opacity(x01, lines, wl, dx):
    """
    Sum of line profiles (unit abundance) on a log-uniform wavelength grid
    spanning x01. The grid step is dx [km/s], reduced if needed so the
    narrowest lines (at the blue end) have at least 2 pixels per FWHM.
    Returns (wavelengths, opacity).

    Each profile is only evaluated over the pixels within its truncation
    window, rather than the whole grid. Lines are added in order, so the
    result is identical to summing full-grid profiles.
    """
    x0, x1 = x01
    dx = min(dx, C_KMS * line_width(x0, wl) / (2 * x0))
    xm = logarange(x0, x1, C_KMS / dx)

    lam = lines["lam"].to_numpy(dtype=float) if len(lines) else np.empty(0)
    strength = lines["strength"].to_numpy(dtype=float) if len(lines) else np.empty(0)
    width = line_width(lam, wl)
    half_window = LORENTZ_CUTOFF_FWHM * width
    # One pixel of margin either side: lorentzian() itself zeroes anything
    # beyond the cutoff, so the window only needs to cover it.
    i0 = np.maximum(np.searchsorted(xm, lam - half_window) - 1, 0)
    i1 = np.minimum(np.searchsorted(xm, lam + half_window, side="right") + 1, len(xm))

    kappa = np.zeros_like(xm)
    for lam_k, w_k, s_k, a, b in zip(lam, width, strength, i0, i1):
        kappa[a:b] += s_k * lorentzian(xm[a:b], lam_k, w_k)
    return xm, kappa


class ModelBuilder:
    """
    Builds the convolved model spectrum for a given abundance from the
    precomputed opacity.
    """
    def __init__(self, xm, kappa, res, emission=False):
        self.xm = xm
        self.kappa = kappa
        self.res = res
        self.emission = emission
        if emission:
            # Emission is linear in abundance, so convolve once and rescale
            self._em_unit = self._convolve(kappa)

    def _convolve(self, y):
        return Spectrum(self.xm, y).convolve_gaussian(self.res)

    def __call__(self, logA):
        """
        Model spectrum for abundance 10**logA: normalised flux for absorption,
        or the line emission (to be added to a continuum of 1) for emission.
        """
        A = 10**logA
        if self.emission:
            M = self._em_unit * A
        else:
            M = self._convolve(np.exp(-A * self.kappa))
        M += 1e-300  # Needed to deal with numerical issues with very strong lines after convolution
        return M


class IonModel:
    """
    Per-ion parameters, plus the model computed from them and the global
    parameters, with each stage of the calculation cached.
    """
    PARAMS = ("ion", "emission", "logA", "wl", "N")
    STAGES = ("lines", "opacity", "builder", "model")

    def __init__(self, ion, emission=False, logA=LOGA_INIT, wl=0.5, N=500,
                 include=True):
        self.ion = ion
        self.emission = emission
        self.logA = logA
        self.wl = wl
        self.N = N
        self.include = include  # Whether shown, for ions in the saved set
        self._keys = {}
        self._cache = {}

    @property
    def key(self):
        """
        Identifies the ion: an ion can be saved once each for absorption and
        emission.
        """
        return (self.ion, self.emission)

    @property
    def label(self):
        return self.ion + (" (em)" if self.emission else "")

    @property
    def n_lines(self):
        return len(self._cache.get("lines", ()))

    def to_dict(self):
        d = {name: getattr(self, name) for name in self.PARAMS}
        d["include"] = self.include
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def copy(self):
        """
        Copy sharing the (never modified in place) cached results
        """
        new = copy.copy(self)
        new._keys = dict(self._keys)
        new._cache = dict(self._cache)
        return new

    def adopt_cache(self, other):
        """
        Take over the cached results of another IonModel (typically a copy of
        this one that was computed elsewhere). The cache is keyed by the
        parameters used, so this is valid whatever this ion's parameters are.
        """
        self._keys = other._keys
        self._cache = other._cache

    def _stale(self, stage, key):
        """
        Whether a stage needs recomputing. If so, it and all later stages are
        invalidated.
        """
        if self._keys.get(stage) == key:
            return False
        for later in self.STAGES[self.STAGES.index(stage):]:
            self._keys.pop(later, None)
        self._keys[stage] = key
        return True

    def model(self, x01, g):
        """
        Model spectrum (unscaled) over wavelength range x01 for global
        parameters g (GlobalParams). x01 is assumed fixed for the lifetime of
        the IonModel. Raises ModelError if the model cannot be computed (e.g.
        no lines in range).
        """
        k_lines = (self.ion, self.emission, self.N, g.Teff, g.rv, g.wave)
        k_opacity = k_lines + (self.wl, g.dx)
        k_builder = k_opacity + (g.res,)
        k_model = k_builder + (self.logA,)

        if self._stale("lines", k_lines):
            try:
                self._cache["lines"] = select_lines(
                    self.ion, x01, g.Teff, g.rv, self.N, self.emission, g.wave)
                self._cache["error"] = None
            except ModelError as err:
                self._cache["lines"] = ()
                self._cache["error"] = err
        if self._cache["error"] is not None:
            raise self._cache["error"]
        if self._stale("opacity", k_opacity):
            self._cache["opacity"] = compute_opacity(
                x01, self._cache["lines"], self.wl, g.dx)
        if self._stale("builder", k_builder):
            self._cache["builder"] = ModelBuilder(
                *self._cache["opacity"], g.res, self.emission)
        if self._stale("model", k_model):
            self._cache["model"] = self._cache["builder"](self.logA)
        return self._cache["model"]


def combine_models(models, x):
    """
    Combine absorption models multiplicatively, and add emission models,
    interpolated onto wavelength grid x. Outside each model's range,
    absorption is 1 and emission 0. models is a list of (IonModel, Spectrum).
    """
    y = np.ones_like(x)
    for im, M in models:
        if not im.emission:
            y *= M.interp(x, left=1.0, right=1.0)
    for im, M in models:
        if im.emission:
            y += M.interp(x, left=0.0, right=0.0)
    return y
