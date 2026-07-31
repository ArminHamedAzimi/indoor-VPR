"""Concrete VPR algorithm implementations only."""

from .dinov2 import DINOv2
from .rgb_histogram import RGBHistogram

__all__ = ["DINOv2", "RGBHistogram"]
