#!/usr/bin/env python
"""
element_spec: interactive identification of atomic/ionic lines in (white
dwarf) spectra using simple LTE models. See README.md, or run with --help.

This is just the entry point; the code is in the elspec package alongside this
file. Python resolves softlinks when adding a script's directory to the import
path, so this can be run via a link from anywhere.
"""
import sys

from elspec.app import main

if __name__ == "__main__":
    sys.exit(main())
