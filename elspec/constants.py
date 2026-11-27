"""
Physical constants, file locations and fixed settings
"""
import os.path

# The package lives next to the line list. realpath resolves softlinks, so
# this works when the program is run via a link from elsewhere.
INSTALL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LINELIST_PATH = os.path.join(INSTALL_DIR, "linelist.rec.npy")

C_KMS = 2.998e5  # Speed of light [km/s]
HC_OVER_K = 1.43877688  # hc/k_B [cm K], i.e. 1/(k_B T) [cm] = HC_OVER_K / T
LAMBDA_REF = 5000.0  # Reference wavelength [Å] for line widths and strengths
EV_TO_INV_CM = 8065.54  # Energy conversion used for the line list [cm-1/eV]

# Wavelength range [Å] used when no data are given
NO_DATA_RANGE = (1000.0, 10000.0)

# Abundance slider (log10 scale)
LOGA_MIN, LOGA_MAX, LOGA_INIT, LOGA_STEP = -5.0, 10.0, 0.0, 0.05

DEFAULT_SESSION = "element_spec_session.json"
