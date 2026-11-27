"""
Atomic line data, and selecting/weighting the lines of an ion
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .constants import LINELIST_PATH, C_KMS, HC_OVER_K, LAMBDA_REF
from .spectrum import vac_to_air


class ModelError(Exception):
    """
    An ion's model cannot be computed
    """


class UnknownIonError(ModelError):
    pass


class NoLinesError(ModelError):
    pass


@lru_cache(maxsize=1)
def load_linelist():
    """
    The full line list (columns: ion, lam [Å, vacuum], E_low [cm-1], loggf),
    loaded once.
    """
    return pd.DataFrame(np.load(LINELIST_PATH))


@lru_cache(maxsize=1)
def available_ions():
    return [ion.decode() for ion in np.unique(load_linelist()["ion"])]


def select_lines(ion, x01, Teff, rv, N, emission=False, wave="air"):
    """
    Lines of an ion within the wavelength range x01, shifted by radial
    velocity rv [km/s] and converted to air wavelengths if requested. A
    'strength' column is added (LTE, at temperature Teff), and only the N
    strongest lines are returned, strongest first.

    Absorption strength ∝ gf λ² exp(-E_low/kT) (1 - exp(-hν/kT))
    Emission strength   ∝ gf λ⁻³ exp(-E_up/kT)
    with λ in units of LAMBDA_REF.
    """
    linelist = load_linelist()
    lines = linelist[linelist["ion"] == ion.encode()].copy()
    if lines.empty:
        raise UnknownIonError(f"No atomic data for {ion}")

    # Transition energy [cm-1], from the (vacuum) rest wavelength
    lines["dE"] = 1e8 / lines["lam"]

    # Apply velocity and air shifts
    beta_v = rv / C_KMS
    lines["lam"] *= np.sqrt((1 + beta_v) / (1 - beta_v))
    if wave == "air":
        lines["lam"] = vac_to_air(lines["lam"])

    # Only use lines in data range
    x0, x1 = x01
    lines = lines[(lines["lam"] > x0) & (lines["lam"] < x1)]
    if lines.empty:
        raise NoLinesError(f"No {ion} lines in the range {x0:.1f} -- {x1:.1f}A")

    beta = HC_OVER_K / Teff  # 1/(k_B T) [cm]
    gf = 10 ** lines["loggf"]
    lam_rel = lines["lam"] / LAMBDA_REF
    if emission:
        E_hi = lines["E_low"] + lines["dE"]
        lines["strength"] = gf * (LAMBDA_REF / lines["lam"])**3 * np.exp(-beta * E_hi)
    else:
        boltz = np.exp(-beta * lines["E_low"])
        stim = -np.expm1(-beta * lines["dE"])
        lines["strength"] = gf * boltz * stim * lam_rel**2
    lines = lines.sort_values("strength", ascending=False)
    return lines[:N]
