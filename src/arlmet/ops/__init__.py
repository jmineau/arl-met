"""
Operations on ARL meteorology files.

These are the transforms a user applies to ARL files: cropping a spatial or
vertical subset (`extract_subset`), sampling fields at points (`sample_points`),
and joining files together (`concat`, `concat_by_time`). The data-model and
codec modules (`file`, `grid`, `record`, `packing`, ...) live one level up.
"""

from .concat import concat, concat_by_time
from .sample import sample_points
from .subset import extract_subset

__all__ = [
    "concat",
    "concat_by_time",
    "extract_subset",
    "sample_points",
]
