"""
Line identification aids: which of an ion's lines produce visible features
in its model, where to mark them, and descriptions of lines near a given
wavelength.
"""
import numpy as np

from .constants import EV_TO_INV_CM
from .model import line_width
from .spectrum import vac_to_air

# Peak height of a unit-area Gaussian is 1/(GAUSS_FWHM_AREA * FWHM)
GAUSS_FWHM_AREA = np.sqrt(np.pi / np.log(2)) / 2  # = 1.0645

MARK_MIN_DEPTH = 0.01  # Lines must be at least this deep (or high) to be marked
MAX_MARKS = 100  # At most this many lines are marked


def predicted_depth(lines, logA, wl, res, emission=False):
    """
    Estimated depth (or height, for emission) relative to the continuum of
    each line on its own, after instrumental broadening, for abundance
    10**logA, Lorentzian width wl (at LAMBDA_REF) and resolution res [Å].

    In the optically thin limit a line's equivalent width is W = A*strength
    [Å]. If the line is narrower than the resolution its depth is then about
    W divided by the width of the instrumental profile; if broader, it follows
    from the peak optical depth of the Lorentzian. The smaller of the two is
    used, with saturation (absorption only) limiting depths to below 1.
    Blends with other lines are ignored.
    """
    A = 10.0**logA
    strength = lines["strength"].to_numpy(dtype=float)
    lam = lines["lam"].to_numpy(dtype=float)
    W = A * strength
    tau_peak = 2 * W / (np.pi * line_width(lam, wl))
    with np.errstate(divide="ignore"):
        smoothed = W / (GAUSS_FWHM_AREA * res)
    if emission:
        return np.minimum(tau_peak, smoothed)
    return np.minimum(-np.expm1(-tau_peak), smoothed)


def visible_lines(lines, logA, wl, res, M, emission=False,
                  min_depth=MARK_MIN_DEPTH):
    """
    The lines expected to be visible, with a 'depth' column (height, for
    emission) added. M is the ion's (unscaled) model. lines must be
    strongest first (as from select_lines), and keep that order.

    The depth of each line is the smaller of its predicted_depth (an upper
    bound, which keeps weak lines in the wings of strong ones from being
    credited with the wing's depth) and the depth of the model at the line
    (which accounts for saturation). Lines shallower than min_depth are
    dropped.
    """
    predicted = predicted_depth(lines, logA, wl, res, emission)
    at_line = M.interp(lines["lam"].to_numpy(dtype=float))
    model_depth = at_line if emission else 1 - at_line
    depth = np.minimum(predicted, model_depth)
    keep = depth >= min_depth
    visible = lines[keep].copy()
    visible["depth"] = depth[keep]
    return visible


def mark_positions(visible, res, max_marks=MAX_MARKS):
    """
    Wavelengths at which to mark the visible lines (strongest first). Lines
    within res/2 of a stronger marked line are part of the same feature, so
    are not marked separately.
    """
    marks = []
    for lam in visible["lam"].to_numpy(dtype=float):
        if len(marks) >= max_marks:
            break
        if all(abs(lam - m) > res / 2 for m in marks):
            marks.append(lam)
    return np.array(marks)


def lines_near(visible, x, tolerance, max_lines=5):
    """
    The visible lines within tolerance [Å] of wavelength x, strongest first
    """
    near = np.abs(visible["lam"].to_numpy(dtype=float) - x) <= tolerance
    return visible[near][:max_lines]


def rest_wavelength(lines, wave):
    """
    Rest wavelengths (air or vacuum, as given by wave) of lines, from their
    transition energies (lam includes any radial velocity shift)
    """
    lam_vac = 1e8 / lines["dE"].to_numpy(dtype=float)
    return vac_to_air(lam_vac) if wave == "air" else lam_vac


def describe_lines(near, ion, wave, rv, emission=False):
    """
    Plain-text description of lines (rows of visible_lines, strongest first)
    for a tooltip. The depth of the feature they form is given once; when
    several lines share it, each one's strength relative to the strongest
    is shown.
    """
    rest = rest_wavelength(near, wave)
    feature = "height" if emission else "depth"
    rows = [f"{ion} ({wave} wavelengths), feature {feature} ≈ "
            f"{min(near['depth'].max(), 9.99):.0%}"]
    strongest = near["strength"].max()
    for (_, line), lam_rest in zip(near.iterrows(), rest):
        text = f"{line.lam:9.2f} Å"
        if rv != 0:
            text += f" (rest {lam_rest:.2f})"
        text += (f"   log gf = {line.loggf:+.2f}"
                 f"   E_low = {line.E_low / EV_TO_INV_CM:.2f} eV")
        if len(near) > 1:
            text += f"   rel. strength {line.strength / strongest:.2f}"
        rows.append(text)
    return "\n".join(rows)
