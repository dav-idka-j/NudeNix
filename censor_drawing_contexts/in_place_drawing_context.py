import sys
import numpy as np
from typing import Tuple, Dict, Any, TYPE_CHECKING # Added TYPE_CHECKING

from interfaces.base_censor_drawing_context import BaseCensorDrawingContext
from censors_registry import CensorsRegistry # Import CensorsRegistry
from detection_types import Detection # Added for CensorProtocol type hint

if TYPE_CHECKING:
    from typing import Protocol # Or you could define a simple Protocol
    # Define a Protocol that represents the convention
    class CensorProtocol(Protocol):
        def __init__(self, settings: Dict[str, Any]): ...
        def apply(self, image_np: np.ndarray, detection: 'Detection') -> None: ... # Changed to Detection

class InPlaceDrawingContext(BaseCensorDrawingContext):
    """
    An implementation of BaseCensorDrawingContext that applies censor effects
    directly to a NumPy image array using OpenCV, modifying it in place.
    """
    def __init__(self, censors_registry: CensorsRegistry):
        self.censors_registry = censors_registry

    def apply_censor_effect(self, image_np: np.ndarray, detection: 'Detection', censor_type: str, settings: Dict) -> None: # Changed to detection

        try:
            # Type hint using CensorProtocol for static analysis
            label = detection.label();
            censor_instance: 'CensorProtocol' = self.censors_registry.get_censor_instance(label, censor_type, settings) # Pass settings
            censor_instance.apply(image_np, detection) # Pass detection here
        except ValueError as e:
            # Re-raise the ValueError so that the calling test can catch it.
            # The warning message is useful for debugging but the error needs propagation.
            print(f"Warning: {e}", file=sys.stderr) # Keep warning for general use
            raise e # Re-raise the exception
