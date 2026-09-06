"""HTTP in front of the pipeline.

`main.py` holds the routes, `service.py` the pipeline in terms a request can
supply, `jobs.py` the single background worker that runs anything too slow to
answer in a request. Nothing here reaches past `service` into `falconvar`, and
nothing in `falconvar` knows this exists.
"""
