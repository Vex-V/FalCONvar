"""Video ingestion: source -> chunker -> samplers -> manifest.

    source/    probe, sequential read, decimation
    chunker/   media time -> chunk id
    samplers/  which decimated frames are worth describing
    driver.py  wires them together and writes the manifest
"""
