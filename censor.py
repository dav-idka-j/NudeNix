import numpy as np
import os
from typing import Dict, Any, List, Tuple

from config_loader import ConfigLoader
from interfaces.base_censor_drawing_context import BaseCensorDrawingContext
from detection_types import Detection
from statistics import CensorReportModule

class CensorProcessor:
    def __init__(self, config_loader: ConfigLoader, drawing_context: BaseCensorDrawingContext, stats=None):
        self.config_loader = config_loader
        self.drawing_context = drawing_context
        print(f"CensorProcessor initialized with config and drawing context.")

        if stats and stats.enabled:
            stats.register_module(CensorReportModule(), priority=40)

    def apply_censorship(self, frame_ndarray: np.ndarray, detections: List[Detection], frame_time: float = 0.0, stats=None) -> np.ndarray:
        """
        Applies censorship effects to the given frame based on detections and configured settings,
        delegating the actual drawing to the provided drawing_context.

        Args:
            frame_ndarray (np.ndarray): The video frame as a NumPy array (RGB).
                                        This frame will be passed to the drawing_context,
                                        which may modify it in-place or use it for instructions.
            detections (List[Detection]): A list of Detection objects.
            frame_time (float): The timestamp of the current frame (for debugging/logging).
            stats (PerformanceStatistics, optional): Statistics object to record metrics.

        Returns:
            np.ndarray: The (potentially modified) frame_ndarray.
        """
        for detection in detections:
            label = detection.label()
            score = detection.score()

            # Get effective settings for this label (merging all_labels defaults with overrides)
            effective_settings = self.config_loader.get_label_settings(label)

            # Apply general detection filters
            if not effective_settings.get('enabled', False):
                continue
            if score < effective_settings.get('min_score', 0.0):
                continue

            # Get bounding box from Detection object
            bounding_box_for_drawing = detection.bounding_box()

            # Apply configured censors
            censors_to_apply = effective_settings.get('censors', [])
            for censor_config in censors_to_apply:
                censor_type = censor_config.get('type')
                censor_settings = censor_config.get('settings', {})

                # Delegate the drawing of the censor effect to the drawing_context
                self.drawing_context.apply_censor_effect(
                    image_np=frame_ndarray,
                    detection=detection,
                    censor_type=censor_type,
                    settings=censor_settings,
                    stats=stats
                )
        return frame_ndarray
