"""
seistools is a collection of python functions and classes for analyzing earthquake data
written by Eric Daub. The tools are a bit eclectic, but span earthquake catalogs,
finite source models, friction experiments, friction laws, fault mechanics, and numerical
integration.

The package requires numpy and scipy, and should work for Python 2 or 3.
"""

from . import biax
from . import rough
from . import catalog
from . import coulomb
from . import stress
from . import source
from . import integration
from . import stz
