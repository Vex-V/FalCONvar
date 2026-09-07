"""ver3 -- a rebuild of the pipeline, assembled deliberately.

`falconvar/` works and is measured, but it grew by accretion: 132 files, and
several of them exist because something needed them once. This package is built
the other way round -- nothing arrives until something here needs it, and what
arrives is read, simplified and re-justified rather than moved.

The rule while this exists: **ver3 never imports falconvar.** Not because the
old code is wrong, but because an import is how "we should check whether this
is still needed" quietly becomes "it is still here". Code crosses over by being
read and rewritten, never by being referenced.

The name is transitional. A version in a package name is the thing the
`ver2 -> falconvar` rename just removed, so this directory is scaffolding: when
it is the pipeline, it takes the pipeline's name.
"""
