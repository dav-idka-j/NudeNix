from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from detection_types import Detection

class BaseCensorDrawingContext(ABC):
    """
    Abstract Base Class for censor drawing contexts.
    Defines the interface for how censor effects are applied to an image.
    This decouples the CensorProcessor from the specific rendering mechanism
    (e.g., direct modification of a NumPy array, or accumulating instructions for an overlay).
    """

    @abstractmethod
    def apply_censor_effect(self, image_np: np.ndarray, detection: 'Detection', censor_type: str, settings: Dict) -> None:
        """
        Applies a generic censor effect to the specified detection on the image.

        Args:
            image_np (np.ndarray): The image array to draw on.
            detection (Detection): The Detection object containing bounding box and other info.
            censor_type (str): The type of censor to apply (e.g., 'box', 'pixelation').
            settings (Dict): Censor-specific settings (e.g., color for box, pixel_size for pixelation).
        """
        pass
