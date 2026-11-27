"""
Definitions of the numeric parameters that can be set from the command line
and the GUI. Global parameters apply to every ion; per-ion parameters belong
to each ion separately.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str  # Attribute name (also the --option name)
    label: str  # GUI label
    default: float
    minimum: float
    maximum: float
    step: float
    decimals: int | None  # None for integer parameters
    tooltip: str

    @property
    def is_int(self):
        return self.decimals is None


GLOBAL_PARAMS = [
    ParamSpec("Teff", "Teff [K]", 10000.0, 1000.0, 200000.0, 500.0, 0,
              "Effective temperature"),
    ParamSpec("rv", "RV [km/s]", 0.0, -5000.0, 5000.0, 5.0, 1,
              "Radial velocity"),
    ParamSpec("res", "Resolution [Å]", 2.0, 0.01, 100.0, 0.5, 2,
              "Instrumental resolution (Gaussian FWHM) of the models"),
    ParamSpec("dx", "Grid step [km/s]", 5.0, 0.1, 100.0, 1.0, 1,
              "Maximum log-wavelength step of the model grid (reduced "
              "automatically if lines would be undersampled)"),
    ParamSpec("scale", "Scale", 1.0, 0.01, 100.0, 0.05, 2,
              "Rescale models by this factor"),
]

ION_PARAMS = [
    ParamSpec("wl", "Lorentz FWHM [Å]", 0.5, 0.01, 100.0, 0.1, 2,
              "Lorentzian FWHM at 5000 Å (scales with λ²)"),
    ParamSpec("N", "N lines", 500, 1, 100000, 100, None,
              "Number of strongest lines used"),
]

# Wavelength conventions: (GUI label, value)
WAVE_OPTIONS = [("Air", "air"), ("Vacuum", "vac")]
