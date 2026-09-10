"""
Score each person on VGGT depth: A inside the mask, B in a surround ring,
score = |A-B|/B. A view is flattened if any person is.
"""

from .human_flatten import HumanFlatten

__all__ = ["HumanFlatten"]
