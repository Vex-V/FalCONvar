"""`python -m ver3.media` -> the component's driver.

A module rather than running `driver.py` directly: importing the package
already loads `driver`, so `python -m ver3.media.driver` would execute it twice
and Python warns about exactly that.
"""

from .driver import main

raise SystemExit(main())
