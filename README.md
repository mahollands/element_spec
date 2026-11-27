# Element-spec
This tool is designed for quickly identifying which elements (or more
specifically which ions) are present in a spectrum, by comparing it with
simple LTE models that take seconds rather than the minutes/hours needed for a
full atmospheric model.

### Requirements
* Python 3.10 or later
* numpy, pandas, numba, matplotlib
* PyQt5, PyQt6 or PySide6

No installation is needed beyond these: clone the repository and run
`element_spec.py` (directly, or via a softlink to it from somewhere on your
`PATH`).

### Usage
```
element_spec spectrum.dat
```
Spectra are text files with columns of wavelength [Å] and (normalised) flux;
further columns (e.g. flux errors) are ignored. Columns are separated by
whitespace, or commas for `.csv` files, and comments (`#`) and header lines are
skipped. Multiple files can be given (e.g. `element_spec blue.dat red.dat`),
and are joined into a single spectrum. Omit the filename(s) to show models
alone over 1000--10000 Å.

Everything else is set in the window, which shows the data (grey), the model
for the ion currently being edited (red), and the combined model of all saved
ions (blue):

* **All ions**: air/vacuum wavelengths, Teff, radial velocity, resolution,
  model grid step and a display scale factor. Changing these updates every
  ion, saved or not.
* **Current ion**: the ion (type to search), absorption or emission, Lorentzian
  width and number of lines used, plus the abundance slider below the plot
  (log scale, in arbitrary units). *Save to ion set* adds the ion to the saved
  ions, or updates it if already saved. With *Mark lines* on, the current
  ion's visible lines are marked with ticks above the spectrum; hover near a
  feature for the wavelength (and rest wavelength if shifted), log gf and
  lower level energy of the lines forming it.
* **Saved ions**: tick to include each in the blue model; *Edit* (or
  double-click) loads one back for adjustment; *Remove* deletes it.

Saved ions and the global settings are kept in `element_spec_session.json` in
the current directory, so each data directory has its own and the next run
starts where you left off. Starting values can also be given on the command
line (e.g. `--ion FeII --Teff 7500 --rv 50`), overriding the session file; see
`element_spec --help`.

### Strategy
A good strategy is to start with the most obvious ions, e.g. HI or CaII, and
use that to establish the RV and Teff, saving each ion when it looks right.
Then work through the next most common ions in your spectrum to mark off the
more obvious lines. In the case of D(A)Z white dwarfs this might be Ca, Mg, Fe,
and Si where depending on the temperature, neutral and/or first singly ionised
lines should be checked. Then the more exotic ions (such as TiI/II, SrII etc)
can be checked against the remaining unidentified lines. Refining Teff or RV
later updates all the saved ions together.

### Code layout
`element_spec.py` is the executable (it can be run via a softlink from
anywhere). The code is in the `elspec` package beside it:

| Module | Contents |
|---|---|
| `constants.py` | Physical constants, paths, fixed settings |
| `params.py` | Definitions of the adjustable parameters (shared by command line and GUI) |
| `spectrum.py` | Minimal spectrum class (text file reading, Gaussian convolution, interpolation) and air/vacuum conversion |
| `data.py` | Loading the observed spectra |
| `linelist.py` | The atomic line list; selecting and weighting an ion's lines |
| `model.py` | Line opacity, model spectra, and per-ion caching of each stage |
| `jobs.py` | Model calculations as self-contained jobs (run in a background thread) |
| `lineid.py` | Which of an ion's lines are visible, where to mark them, tooltip text |
| `session.py` | Reading/writing the session file |
| `app.py` | Command line and start-up |
| `gui.py` | The Qt window |

The line list (`linelist.rec.npy`) must stay alongside `element_spec.py`.
