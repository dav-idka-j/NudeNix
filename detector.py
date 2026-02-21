import os
import sys
import numpy as np
from nudenet import NudeDetector
from typing import List

from detection_types import Detection

class NsfwDetector:
    def __init__(self, model_path=None, inference_resolution=320, nude_detector_instance=None):
        if nude_detector_instance:
            self.detector = nude_detector_instance
        else:
            self.detector = NudeDetector(
                model_path=model_path,
                inference_resolution=inference_resolution
            )
        print(f"NudeDetector initialized with model_path={model_path}, inference_resolution={inference_resolution}.")

    def detect(self, frame_ndarray: np.ndarray) -> List[Detection]:
        """
        Detects NSFW content in a single video frame (NumPy array).

        Args:
            frame_ndarray (np.ndarray): The video frame as a NumPy array (e.g., RGB).

        Returns:
            list[Detection]: A list of Detection objects.
        """
        raw_detections = self.detector.detect(frame_ndarray)
        return [Detection(d) for d in raw_detections]
