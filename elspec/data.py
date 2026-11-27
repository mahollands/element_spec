"""
Loading the observed spectra
"""
import os.path

import numpy as np

from .constants import NO_DATA_RANGE
from .spectrum import Spectrum


def load_spectra(fnames):
    """
    Loads and joins the input spectra (text files, see Spectrum.from_txt).
    With no files, returns an empty spectrum spanning NO_DATA_RANGE (so models
    can be shown alone). Raises FileNotFoundError listing any missing files,
    or ValueError if a file cannot be read.
    """
    if not fnames:
        return Spectrum(np.array(NO_DATA_RANGE), 0.0)
    missing = [f for f in fnames if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError("Could not find file(s): " + ", ".join(missing))
    spectra = []
    for fname in fnames:
        try:
            spectra.append(Spectrum.from_txt(fname))
        except ValueError as err:
            raise ValueError(f"Could not read {fname}: {err}") from err
    return Spectrum.join(spectra)
