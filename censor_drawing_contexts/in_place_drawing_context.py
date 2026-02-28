import sys
import numpy as np
from typing import Tuple, Dict, Any, TYPE_CHECKING

from interfaces.base_censor_drawing_context import BaseCensorDrawingContext
from censors_registry import CensorsRegistry
from detection_types import Detection, CensorMetadata

if TYPE_CHECKING:
    from typing import Protocol
    class CensorProtocol(Protocol):
        def __init__(self, settings: Dict[str, Any]): ...
        def apply(self, image_np: np.ndarray, detection: 'Detection') -> None: ...

class InPlaceDrawingContext(BaseCensorDrawingContext):
    """
    An implementation of BaseCensorDrawingContext that applies censor effects
    directly to a NumPy image array using OpenCV, modifying it in place.
    """
    def __init__(self, censors_registry: CensorsRegistry):
        self.censors_registry = censors_registry

    def apply_censor_effect(self, image_np: np.ndarray, detection: 'Detection', censor_type: str, settings: Dict, stats=None, metadata: 'CensorMetadata' = None) -> None:

        try:
            label = detection.label()
            censor_instance: 'CensorProtocol' = self.censors_registry.get_censor_instance(label, censor_type, settings)

            if stats:
                with stats.time_censor(censor_type):
                    censor_instance.apply(image_np, detection, metadata)
            else:
                censor_instance.apply(image_np, detection, metadata)

        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
            raise e
