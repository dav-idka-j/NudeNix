import numpy as np
from typing import List, Dict, Tuple, Any

from detection_types import Detection

class BoxClusterer:
    """
    Clusters closely located Detection objects of the same class into a single, larger bounding box.
    """
    def __init__(self, max_distance: float):
        """
        Initializes the BoxClusterer.

        Args:
            max_distance: The maximum Euclidean distance between the closest sides of two boxes
                          of the same class to be considered for clustering.
        """
        self.max_distance = max_distance

    def _boxes_distance(self, det1: Detection, det2: Detection) -> float:
        """
        Calculates the Euclidean distance between the closest sides of two Detection objects.
        """
        x1_min, y1_min, x1_max, y1_max = det1.x1(), det1.y1(), det1.x2(), det1.y2()
        x2_min, y2_min, x2_max, y2_max = det2.x1(), det2.y1(), det2.x2(), det2.y2()

        # Calculate horizontal distance
        if x1_max < x2_min: # det1 is to the left of det2
            dx = x2_min - x1_max
        elif x2_max < x1_min: # det2 is to the left of det1
            dx = x1_min - x2_max
        else: # boxes overlap horizontally
            dx = 0

        # Calculate vertical distance
        if y1_max < y2_min: # det1 is above det2
            dy = y2_min - y1_max
        elif y2_max < y1_min: # det2 is above det1
            dy = y1_min - y2_max
        else: # boxes overlap vertically
            dy = 0

        return np.sqrt(dx*dx + dy*dy)


    def _merge_detections(self, detections_to_merge: List[Detection]) -> Detection:
        """
        Merges a list of Detection objects into a single new Detection object.
        The merged box is an Axis-Aligned Bounding Box (AABB) encompassing all merged boxes.
        The score is the maximum score among the merged detections.
        """
        if not detections_to_merge:
            return None

        x_min_coords = [det.x1() for det in detections_to_merge]
        y_min_coords = [det.y1() for det in detections_to_merge]
        x_max_coords = [det.x2() for det in detections_to_merge]
        y_max_coords = [det.y2() for det in detections_to_merge]

        merged_x_min = min(x_min_coords)
        merged_y_min = min(y_min_coords)
        merged_x_max = max(x_max_coords)
        merged_y_max = max(y_max_coords)

        merged_box_raw = [
            float(merged_x_min),
            float(merged_y_min),
            float(merged_x_max - merged_x_min),
            float(merged_y_max - merged_y_min),
        ]

        merged_score = max([det.score() for det in detections_to_merge])

        # Assume all detections in the merge group have the same class
        # (This is ensured by grouping by class before clustering)
        # Create a raw dictionary first, then wrap it in a Detection object
        merged_raw_detection = {
            "box": merged_box_raw,
            "score": merged_score,
            "class": detections_to_merge[0].label() # Take class from the first detection
        }
        return Detection(merged_raw_detection)

    def cluster_boxes(self, detections: List[Detection]) -> List[Detection]:
        """
        Clusters detected boxes based on class and proximity.

        Args:
            detections: A list of Detection objects.

        Returns:
            A new list of Detection objects with clustered boxes.
        """
        if not detections:
            return []

        clustered_detections_list: List[Detection] = []

        # Group detections by class
        detections_by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            det_class = det.label()
            if det_class not in detections_by_class:
                detections_by_class[det_class] = []
            detections_by_class[det_class].append(det)

        for det_class, dets_for_class in detections_by_class.items():
            is_clustered = [False] * len(dets_for_class)

            for i, det_i in enumerate(dets_for_class):
                if is_clustered[i]:
                    continue

                current_cluster: List[Detection] = [det_i]
                is_clustered[i] = True

                for j, det_j in enumerate(dets_for_class):
                    if is_clustered[j]:
                        continue

                    if self._boxes_distance(det_i, det_j) <= self.max_distance:
                        current_cluster.append(det_j)
                        is_clustered[j] = True

                merged_det = self._merge_detections(current_cluster)
                if merged_det:
                    clustered_detections_list.append(merged_det)

        return clustered_detections_list
