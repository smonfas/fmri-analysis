# fmri analysis

A collection of tools for fMRI analysis, some are wrapper scripts for other tools some are implemented analysis algorithms. Layer fMRI, MP2RAGE processing, surface activation clustering, ROI generation, visualization, ...

Includes bash and python scripts. Some python scripts can be used from the command line like the bash scripts, e.g. add to path using 

`export PATH=$PATH:fmri_analysis/library`. 

Python scripts can also be used by adding the repository as a git submodule and then using for example 

`from fmri_analysis import layer_analysis`.
