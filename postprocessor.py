import numpy as np
from typing import List, Dict, Tuple

class BoxClusterer:
    """
    Clusters closely located detection boxes of the same class into a single, larger bounding box.
    """
    def __init__(self, max_distance: float):
        """
        Initializes the BoxClusterer.

        Args:
            max_distance: The maximum Euclidean distance between the closest sides of two boxes
                          of the same class to be considered for clustering.
        """
        self.max_distance = max_distance

    def _boxes_distance(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculates the Euclidean distance between the closest sides of two bounding boxes (AABB distance).
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to [x_min, y_min, x_max, y_max] format
        box1_xmin, box1_ymin, box1_xmax, box1_ymax = x1, y1, x1 + w1, y1 + h1
        box2_xmin, box2_ymin, box2_xmax, box2_ymax = x2, y2, x2 + w2, y2 + h2

        # Calculate horizontal distance
        if box1_xmax < box2_xmin: # box1 is to the left of box2
            dx = box2_xmin - box1_xmax
        elif box2_xmax < box1_xmin: # box2 is to the left of box1
            dx = box1_xmin - box2_xmax
        else: # boxes overlap horizontally
            dx = 0

        # Calculate vertical distance
        if box1_ymax < box2_ymin: # box1 is above box2
            dy = box2_ymin - box1_ymax
        elif box2_ymax < box1_ymin: # box2 is above box1
            dy = box1_ymin - box2_ymax
        else: # boxes overlap vertically
            dy = 0

        return np.sqrt(dx*dx + dy*dy)


    def _merge_boxes(self, boxes: List[List[float]], scores: List[float]) -> Dict:
        """
        Merges a list of boxes and their scores into a single new detection dictionary.
        The merged box is an Axis-Aligned Bounding Box (AABB) encompassing all merged boxes.
        The score is the maximum score among the merged boxes.
        """
        if not boxes:
            return None

        # Convert [x, y, w, h] to [x_min, y_min, x_max, y_max] for easier merging
        x_min_coords = [b[0] for b in boxes]
        y_min_coords = [b[1] for b in boxes]
        x_max_coords = [b[0] + b[2] for b in boxes]
        y_max_coords = [b[1] + b[3] for b in boxes]

        merged_x_min = min(x_min_coords)
        merged_y_min = min(y_min_coords)
        merged_x_max = max(x_max_coords)
        merged_y_max = max(y_max_coords)

        # Convert back to [x, y, w, h]
        merged_box = [
            merged_x_min,
            merged_y_min,
            merged_x_max - merged_x_min,
            merged_y_max - merged_y_min,
        ]

        # TODO: Perhaps we should use the average here?
        merged_score = max(scores) # For simplicity, take the max score

        # Assume all boxes in the merge group have the same class
        # (This is ensured by grouping by class before clustering)
        return {"box": merged_box, "score": merged_score, "class": None} # Class will be set by caller

    def cluster_boxes(self, detections: List[Dict]) -> List[Dict]:
        """
        Clusters detected boxes based on class and proximity.

        Args:
            detections: A list of detection dictionaries, each with at least 'box', 'score', and 'class'.

        Returns:
            A new list of detection dictionaries with clustered boxes.
        """
        if not detections:
            return []

        clustered_detections = []

        # Group detections by class
        detections_by_class = {}
        for det in detections:
            det_class = det["class"]
            if det_class not in detections_by_class:
                detections_by_class[det_class] = []
            detections_by_class[det_class].append(det)

        for det_class, dets_for_class in detections_by_class.items():
            # Use a list of booleans to track if a detection has been clustered
            is_clustered = [False] * len(dets_for_class)

            for i, det_i in enumerate(dets_for_class):
                if is_clustered[i]:
                    continue

                # Start a new cluster
                current_cluster_boxes = [det_i["box"]]
                current_cluster_scores = [det_i["score"]]
                is_clustered[i] = True

                # Find other boxes to merge into this cluster
                for j, det_j in enumerate(dets_for_class):
                    if is_clustered[j]:
                        continue

                    if self._boxes_distance(det_i["box"], det_j["box"]) <= self.max_distance:
                        current_cluster_boxes.append(det_j["box"])
                        current_cluster_scores.append(det_j["score"])
                        is_clustered[j] = True # Mark as clustered

                # Merge the boxes in the current cluster
                merged_det = self._merge_boxes(current_cluster_boxes, current_cluster_scores)
                if merged_det:
                    merged_det["class"] = det_class # Assign the correct class
                    clustered_detections.append(merged_det)

        return clustered_detections
