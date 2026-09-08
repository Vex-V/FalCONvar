"""HTTP in front of ver3.

    main.py     the routes
    service.py  the pipeline in terms a request can supply
    jobs.py     one background worker, in-memory job records

**The API calls components, never drivers.** A driver is argparse; importing
one to reach the work behind it is what `falconvar` had to undo. Every
component exposes `run(video_id, ...) -> Produced`, and `service.COMPONENTS`
is a dispatch table over exactly those -- which is why one route runs any of
them rather than there being a handler per stage.
"""
