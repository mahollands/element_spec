# Element-spec
This tool is designed for quickly identifying which elements are present in a spectrum.

### Requirements
* Usual data-analysis modules (numpy, scipy, matplotlib)
* mh.spectra -- another repo of mine for working with astrophysical spectra
* linelist.rec.npy -- this is not hosted here, but can be provided on request.

### Examples
Basic usage looks like
```
element_spec spectrum.dat FeI 7500 1
```
* Arg1: The filename containing the spectrum you are working with.
* Arg2: Ion name
* Arg3: Effective temperature in Kelvin
* Arg4: Abundance in arbitrary units, 1.0 is good starting point.

Other options exist depending what you want to do.
For instance, if your spectrum is redshifed by 50km/s, you can use
```
element_spec  spectrum.dat FeI 7500 1 -rv=50
```

If you think a model looks like a good approximation, you can save it for next time
```
element_spec  spectrum.dat FeI 7500 1 -rv=50 --write
```
This model will be saved in the current directory, and plotted in blue next time.
Multiple models will be combined together, so you can keep track of all lines that
you've identified.

Description of other options can be found with
```
element_spec --help
```
