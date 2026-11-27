"""
Model computations as self-contained jobs, so that they can run in a
background thread while the GUI stays responsive.

A job works on copies of the IonModels, so the GUI can keep changing the
originals (e.g. as a slider is dragged) while it runs. When the job finishes,
the GUI hands the computed caches back to the originals with adopt_caches().
Caches are keyed by the parameters they were computed with, so adopting them
is safe even if the originals' parameters have since changed.
"""
import threading
import time
import traceback
from dataclasses import dataclass, field

from .linelist import ModelError


@dataclass
class ModelResult:
    current: object  # Copy of the current IonModel, as computed
    M: object  # Its model Spectrum, or None if it could not be computed
    current_error: str | None
    g: object = None  # The GlobalParams used
    combined: list = field(default_factory=list)  # (IonModel copy, Spectrum)
    skipped: list = field(default_factory=list)  # Labels of saved ions with no model
    elapsed: float = 0.0


class ModelJob:
    """
    Computes the current ion's model and those of the given saved ions, for
    wavelength range x01 and global parameters g. Call run() (in any thread),
    then once done is set, read result (or error).
    """
    def __init__(self, current, saved, x01, g):
        self._pairs = [(current, current.copy())]
        self._pairs += [(im, im.copy()) for im in saved]
        self.x01 = x01
        self.g = g
        self.done = threading.Event()
        self.result = None
        self.error = None  # Formatted traceback if the job failed unexpectedly

    def run(self):
        try:
            self.result = self._compute()
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.done.set()

    def _compute(self):
        t0 = time.perf_counter()
        (_, current), *saved = self._pairs
        try:
            M, current_error = current.model(self.x01, self.g), None
        except ModelError as err:
            M, current_error = None, str(err)
        result = ModelResult(current, M, current_error, self.g)
        for _, im in saved:
            try:
                result.combined.append((im, im.model(self.x01, self.g)))
            except ModelError:
                result.skipped.append(im.label)
        result.elapsed = time.perf_counter() - t0
        return result

    def adopt_caches(self):
        """
        Give the computed caches to the original IonModels (call from the
        thread that owns them, once the job is done)
        """
        for original, copy in self._pairs:
            original.adopt_cache(copy)
