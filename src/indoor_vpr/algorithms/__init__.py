"""Concrete VPR algorithm implementations only."""

from .dinov2 import DINOv2
from .rgb_histogram import RGBHistogram
from .anyloc import AnyLoc

__all__ = ["AnyLoc", "DINOv2", "RGBHistogram"]
