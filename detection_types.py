from typing import Dict, Tuple, Any
from dataclasses import dataclass

class Detection:
    """
    A wrapper class for a single NudeNet detection, providing a consistent
    interface for accessing detection properties.
    """
    def __init__(self, raw_detection: Dict[str, Any]):
        self._raw = raw_detection
        self._box = raw_detection['box'] # NudeNet box format: [x, y, width, height]

    def x(self) -> int:
        """Returns the x-coordinate of the top-left corner of the bounding box."""
        return int(self._box[0])

    def y(self) -> int:
        """Returns the y-coordinate of the top-left corner of the bounding box."""
        return int(self._box[1])

    def width(self) -> int:
        """Returns the width of the bounding box."""
        return int(self._box[2])

    def height(self) -> int:
        """Returns the height of the bounding box."""
        return int(self._box[3])

    def x1(self) -> int:
        """Returns the x-coordinate of the top-left corner (same as x())."""
        return self.x()

    def y1(self) -> int:
        """Returns the y-coordinate of the top-left corner (same as y())."""
        return self.y()

    def x2(self) -> int:
        """Returns the x-coordinate of the bottom-right corner."""
        return self.x() + self.width()

    def y2(self) -> int:
        """Returns the y-coordinate of the bottom-right corner."""
        return self.y() + self.height()

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Returns the bounding box in (x_min, y_min, x_max, y_max) format."""
        return self.x1(), self.y1(), self.x2(), self.y2()

    def label(self) -> str:
        """Returns the detected class label."""
        return self._raw['class']

    def score(self) -> float:
        """Returns the confidence score of the detection."""
        return self._raw['score']

    def raw(self) -> Dict[str, Any]:
        """Returns the original raw detection dictionary."""
        return self._raw

@dataclass
class CensorMetadata:
    """
    Provides environmental context to censor implementations during application.
    """
    label: str
    score: float
    min_score: float
    confidence_ratio: float # 0.0 at min_score, 1.0 at score=1.0
    frame_time: float = 0.0
