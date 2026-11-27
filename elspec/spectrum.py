"""
A minimal spectrum class, and wavelength utilities.

Units are assumed throughout: wavelengths in Å, and fluxes in whatever units
the data file uses (normalised fluxes are expected for comparison with the
models).
"""
import math
import warnings

import numpy as np

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


class Spectrum:
    """
    Wavelengths x [Å] and fluxes y. y may be given as a scalar, which is
    broadcast to the length of x.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.broadcast_to(np.asarray(y, dtype=float), self.x.shape).copy()

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return f"Spectrum({len(self)} points, {self.x01[0]:.1f}-{self.x01[1]:.1f} Å)"

    @property
    def x01(self):
        """
        The lowest and highest wavelengths
        """
        x0, x1 = self.x[0], self.x[-1]
        return (x0, x1) if x1 > x0 else (x1, x0)

    # .........................................................................
    # Arithmetic with numbers or arrays (not other spectra)

    def __mul__(self, other):
        return Spectrum(self.x, self.y * np.asarray(other))

    __rmul__ = __mul__

    def __add__(self, other):
        return Spectrum(self.x, self.y + np.asarray(other))

    __radd__ = __add__

    # .........................................................................
    # Construction

    @classmethod
    def from_txt(cls, fname):
        """
        Load a spectrum from a text file with columns of wavelength [Å] and
        flux (further columns, e.g. uncertainties, are ignored). Columns may
        be separated by whitespace, or commas for .csv files. Blank lines,
        comments (from #), and any non-numeric header lines before the data
        are skipped. Rows are sorted by wavelength, and rows with non-finite
        wavelength or flux removed. Raises ValueError if the file cannot be
        interpreted.
        """
        delimiter = "," if fname.lower().endswith(".csv") else None
        try:
            # Fast path for plain numeric files (empty files are reported below)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = np.loadtxt(fname, delimiter=delimiter, comments="#", ndmin=2)
        except ValueError:
            # Headers or ragged rows
            data = _read_numeric_rows(fname, delimiter)
        if len(data) == 0:
            raise ValueError("no numeric data found")
        if data.shape[1] < 2:
            raise ValueError("need at least two columns (wavelength, flux)")
        data = data[:, :2]
        data = data[np.all(np.isfinite(data), axis=1)]
        S = cls(data[:, 0], data[:, 1])
        return S[np.argsort(S.x, kind="stable")]

    @classmethod
    def join(cls, spectra):
        """
        Join spectra into one, sorted by wavelength
        """
        S = cls(
            np.hstack([S.x for S in spectra]),
            np.hstack([S.y for S in spectra]),
        )
        return S[np.argsort(S.x, kind="stable")]

    def __getitem__(self, key):
        """
        Sub-spectrum for an index array or slice
        """
        return Spectrum(self.x[key], self.y[key])

    # .........................................................................
    # Processing

    def interp(self, x, left=None, right=None):
        """
        Fluxes linearly interpolated onto wavelengths x. Outside the spectrum,
        left/right are used if given, otherwise the end values.
        """
        return np.interp(x, self.x, self.y, left=left, right=right)

    def convolve_gaussian(self, fwhm):
        """
        Spectrum convolved with a Gaussian of the given FWHM [Å]
        """
        return Spectrum(self.x, convolve_gaussian(self.x, self.y, fwhm))


# Gaussian convolution settings
_NSIG = 5.0  # Kernel truncated at this many standard deviations
_GRID_PER_SIGMA = 3  # Minimum uniform grid points per sigma (non-uniform x)
_MAX_OVERSAMPLE = 10  # Uniform grid at most this many times the input size
_FFT_KERNEL_THRESHOLD = 64  # Longer kernels use FFT convolution

_erf = np.vectorize(math.erf, otypes=[float])


def _read_numeric_rows(fname, delimiter):
    """
    Read a text file as rows of numbers, skipping blank lines, comments
    (from #) and non-numeric header lines before the data. Rows are truncated
    to the shortest row's length. Returns a 2D array.
    """
    rows = []
    with open(fname) as F:
        for lineno, line in enumerate(F, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                rows.append([float(v) for v in line.split(delimiter)])
            except ValueError:
                if rows:
                    raise ValueError(f"non-numeric value on line {lineno}")
                # Header line before the data
    if not rows:
        return np.empty((0, 0))
    ncols = min(len(row) for row in rows)
    return np.array([row[:ncols] for row in rows])


def convolve_gaussian(x, y, fwhm, nsig=_NSIG):
    """
    Convolve y(x) with a Gaussian of the given FWHM (in the same units as x),
    truncated at nsig standard deviations.

    If x is uniformly spaced, y is convolved directly. Otherwise the linear
    interpolant of y is averaged over the cells of a uniform grid (conserving
    its integral), whose spacing is the smaller of the smallest input pixel
    and sigma/_GRID_PER_SIGMA (but no finer than _MAX_OVERSAMPLE times the
    number of input pixels). This is convolved, then interpolated back to x.
    The ends are treated as continuing with their edge values, so nothing
    wraps around. x need not be sorted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same length")
    if len(x) < 2:
        raise ValueError("At least two points are required")
    sigma = abs(fwhm) * FWHM_TO_SIGMA
    if sigma == 0:
        return y.copy()

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    dxs = np.diff(xs)
    span = xs[-1] - xs[0]
    if span <= 0:
        raise ValueError("x values must not all be identical")
    dx_min = dxs[dxs > 0].min()

    if np.allclose(dxs, dx_min, rtol=1e-6, atol=0):
        # Already uniform: convolve in place, no interpolation errors
        yc = _convolve_uniform(ys, sigma / dx_min, nsig)
    else:
        # Grid must resolve both the input pixels and the kernel, so that
        # interpolating the result back onto x is accurate
        dx_grid = min(dx_min, sigma / _GRID_PER_SIGMA)
        dx_grid = max(dx_grid, span / (_MAX_OVERSAMPLE * len(xs)))
        n_grid = int(np.ceil(span / dx_grid)) + 1
        xg = np.linspace(xs[0], xs[-1], n_grid)
        dxg = xg[1] - xg[0]
        # Average (not point-sample) the input over each grid cell, so that
        # narrow features keep their area when moved onto the grid
        edges = np.linspace(xs[0] - dxg / 2, xs[-1] + dxg / 2, n_grid + 1)
        yg = np.diff(_cumulative_integral(xs, ys, edges)) / dxg
        ycg = _convolve_uniform(yg, sigma / dxg, nsig)
        # Convolved result is smooth on the grid scale, so sampling is fine
        yc = np.interp(xs, xg, ycg)

    result = np.empty_like(yc)
    result[order] = yc
    return result


def _gaussian_kernel(sigma_px, nsig):
    """
    Pixel-integrated Gaussian kernel with standard deviation sigma_px (in
    pixels), truncated at +/- nsig standard deviations. Integrating over each
    pixel (rather than sampling at pixel centres) keeps the kernel accurate
    even when sigma is comparable to, or smaller than, a pixel.
    """
    half = max(1, int(np.ceil(nsig * sigma_px)))
    edges = (np.arange(-half, half + 2) - 0.5) / (np.sqrt(2) * sigma_px)
    kernel = 0.5 * np.diff(_erf(edges))
    return kernel / kernel.sum()


def _convolve_uniform(y, sigma_px, nsig):
    """
    Convolve uniformly sampled y with a Gaussian of width sigma_px pixels.
    The ends are padded with the edge values so that the convolution neither
    wraps around nor drags the flux towards zero at the ends.
    """
    kernel = _gaussian_kernel(sigma_px, nsig)
    pad = len(kernel) // 2
    ypad = np.pad(y, pad, mode="edge")
    if len(kernel) <= _FFT_KERNEL_THRESHOLD:
        return np.convolve(ypad, kernel, mode="valid")
    # 'valid' convolution via FFT
    n_full = len(ypad) + len(kernel) - 1
    n_fft = 1 << int(np.ceil(np.log2(n_full)))
    full = np.fft.irfft(np.fft.rfft(ypad, n_fft) * np.fft.rfft(kernel, n_fft), n_fft)
    return full[len(kernel) - 1:len(ypad)]


def _cumulative_integral(x, y, u):
    """
    Exact integral, from x[0] to each of the points u, of the piecewise
    linear interpolant of y(x) (x sorted). Beyond the ends of x, y is
    extended with its edge values.
    """
    h = np.diff(x)
    C = np.concatenate(([0.0], np.cumsum(0.5 * h * (y[1:] + y[:-1]))))
    k = np.clip(np.searchsorted(x, u, side="right") - 1, 0, len(x) - 2)
    t = np.clip(u, x[0], x[-1]) - x[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(h[k] > 0, (y[k + 1] - y[k]) / h[k], 0.0)
    inside = C[k] + y[k] * t + 0.5 * slope * t * t
    below = y[0] * np.minimum(u - x[0], 0)
    above = y[-1] * np.maximum(u - x[-1], 0)
    return inside + below + above


def logarange(x0, x1, R):
    """
    Like np.arange, but with log-spaced points, with spacing such that
    R = x/dx.
    """
    return np.exp(np.arange(np.log(x0), np.log(x1), 1 / R))


def vac_to_air(lam):
    """
    Converts vacuum wavelengths [Å] to air wavelengths, as per VALD3
    documentation
    """
    lam = np.asarray(lam, dtype=float)
    s = 1e4 / lam
    n = 1.0000834254 + 0.02406147 / (130. - s * s) + 0.00015998 / (38.9 - s * s)
    return lam / n
