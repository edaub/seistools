# seistools

A collection of software useful for fault mechanics research

## Overview

This is a collection of tools developed by Eric Daub in the course of doing fault mechanics
research. It includes tools for the following tasks:

* Reading lab experiment data from Chris Marone's lab (the `biax` module)
* Dealing with earthquake catalogs (the `catalog` module)
* Computing Coulomb stresses (the `coulomb` module)
* Performing numerical integration (the `integration` module)
* Generating fractal fault profiles (the `rough` module
* Reading files from the SRCMOD database (the `source` module)
* Computing stress changes using the boundary integral method (the `stress` module)
* Modeling friction using Shear Transformation Zone (STZ) Theory (the `stz` module)

The code is provided with little documentation or verification, though there are a few
unit tests of some of the functionality. These are provided in hopes that they might
be useful to other researchers. If you find any bugs or other issues, please contact
Eric Daub (edaub@turing.ac.uk).

## Installation

The code can be installed in the usual way by running `setup.py` in the shell:

    $ python setup.py install
  
## Tests

Running the unit tests requires `pytest`. You can run the tests from anywhere in the
`seistools` directory by entering `pytest` in the shell. The tests for the `coulomb`
module are currently broken.
