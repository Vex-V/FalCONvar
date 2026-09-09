"""HTTP in front of the pipeline.

    main.py     the routes
    service.py  the pipeline in terms a request can supply
    jobs.py     one background worker, in-memory job records
    browse.py   read-only queries over the rows a run wrote

The API calls components, never drivers: a driver is argparse, and importing
one to reach the work behind it would make a server depend on a CLI. Every
component exposes `run(video_id, ...) -> Produced`, and `service.COMPONENTS`
is a dispatch table over exactly those -- which is why one route runs any of
them rather than there being a handler per stage.
"""
