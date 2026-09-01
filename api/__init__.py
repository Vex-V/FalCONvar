"""HTTP in front of the pipeline.

`main.py` holds the routes, `service.py` the pipeline in terms a request can
supply, `jobs.py` the single background worker that runs anything too slow to
answer in a request. Nothing here reaches past `service` into `ver2`, and
nothing in `ver2` knows this exists.
"""
