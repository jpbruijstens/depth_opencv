# my_vision/__init__.py
from .streaming import (
    depth_stream,
)

from .processing import (
    convert_to_grayscale,
    convert_to_hsv,
    apply_gaussian_blur,
    detect_edges,
    calc_histogram,
    apply_threshold,
    apply_morphology,
    apply_contours,
    draw_contours,
    draw_rectangle,
    draw_text,
    apply_mask,
    apply_erosion,
    apply_dilation,
    color_mask,
    filter_contours,
    bounding_box,
    depth_of_object,
    filter_frame,
)

from .sliders import (
    create_hsv_sliders,
    get_hsv_values,
)

__author__ = "Jeroen Bruijstens"
__version__ = "0.1.0"

