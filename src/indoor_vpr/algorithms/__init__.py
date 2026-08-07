"""Concrete VPR algorithm implementations only."""

from .dinov2 import DINOv2
from .cosplace import CosPlace
from .mixvpr import MixVPR
from .rgb_histogram import RGBHistogram
from .anyloc import AnyLoc

__all__ = ["AnyLoc", "CosPlace", "DINOv2", "MixVPR", "RGBHistogram"]
