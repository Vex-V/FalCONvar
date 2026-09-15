"""`python -m falconvar.video_rag.media` -> the component's driver.

A module rather than running `driver.py` directly: importing the package
already loads `driver`, so `python -m falconvar.video_rag.media.driver` would execute it twice
and Python warns about exactly that.
"""

from .driver import main

raise SystemExit(main())
