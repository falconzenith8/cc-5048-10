"""实景虚拟展品合成器（实景遮挡内核）。"""
from .calibration import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthCamera,
    decode_depth,
    register_depth_to_color,
)
from .composite import composite
from .dataset import (
    FAR_MM,
    NEAR_MM,
    REFERENCE_EXHIBIT,
    simulate_recording,
)
from .pipeline import MOVE_LIMITS, PipelineError, evaluate_frame

__all__ = [
    "CameraExtrinsics",
    "CameraIntrinsics",
    "DepthCamera",
    "MOVE_LIMITS",
    "PipelineError",
    "REFERENCE_EXHIBIT",
    "NEAR_MM",
    "FAR_MM",
    "composite",
    "decode_depth",
    "evaluate_frame",
    "register_depth_to_color",
    "simulate_recording",
]
