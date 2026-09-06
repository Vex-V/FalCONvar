"""Getting frames out of a video: probe it, read it, reduce its rate.

Everything above this layer works on Frame objects and never touches a
VideoCapture, so replacing the reader with a live source changes nothing
downstream.
"""

from .decimate import Decimator
from .fetch import FrameFetcher
from .probe import MAX_PLAUSIBLE_FPS, PROBE_FRAMES, UnusableSource, probe
from .reader import read_frames
from .types import Frame, SourceInfo

__all__ = [
    "MAX_PLAUSIBLE_FPS",
    "PROBE_FRAMES",
    "Decimator",
    "Frame",
    "FrameFetcher",
    "SourceInfo",
    "UnusableSource",
    "probe",
    "read_frames",
]
