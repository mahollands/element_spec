"""
elspec: quick LTE line identification for (white dwarf) spectra.

Modules:
    constants  Physical constants, paths and fixed settings
    params     Definitions of the user-adjustable parameters
    spectrum   Minimal spectrum class, convolution, wavelength utilities
    data       Loading the observed spectra
    linelist   Atomic line data, and selecting/weighting lines for an ion
    model      Line opacity, model spectra, and per-ion model caching
    jobs       Model calculations as jobs, for running in the background
    lineid     Which lines are visible, where to mark them, line descriptions
    session    Saving/loading the ion set and global parameters
    app        Command line and start-up (entry point: elspec.app.main)
    gui        The Qt interface
"""
