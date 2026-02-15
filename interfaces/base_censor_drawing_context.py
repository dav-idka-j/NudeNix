from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np

class BaseCensorDrawingContext(ABC):
    """
    Abstract Base Class for censor drawing contexts.
    Defines the interface for how censor effects are applied to an image.
    This decouples the CensorProcessor from the specific rendering mechanism
    (e.g., direct modification of a NumPy array, or accumulating instructions for an overlay).
    """

    @abstractmethod
    def apply_censor_effect(self, image_np: np.ndarray, bounding_box: Tuple[int, int, int, int], censor_type: str, settings: Dict) -> None:
        """
        Applies a generic censor effect to the specified bounding box on the image.

        Args:
            image_np (np.ndarray): The image array to draw on.
            bounding_box (Tuple[int, int, int, int]): The (x_min, y_min, x_max, y_max) of the area to censor.
            censor_type (str): The type of censor to apply (e.g., 'box', 'pixelation').
            settings (Dict): Censor-specific settings (e.g., color for box, pixel_size for pixelation).
        """
        pass
