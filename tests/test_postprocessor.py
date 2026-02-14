import pytest
import numpy as np
from postprocessor import BoxClusterer # Import the class to be tested

@pytest.fixture
def clusterer_default():
    """Fixture for a BoxClusterer instance with a default max_distance."""
    return BoxClusterer(max_distance=10)

def test_no_clustering_far_apart(clusterer_default):
    """Test case: No clustering when boxes are far apart or have different classes."""
    dets = [
        {"box": [10, 10, 20, 20], "score": 0.9, "class": "A"},
        {"box": [100, 10, 20, 20], "score": 0.8, "class": "A"}, # AABB dist will be 70
        {"box": [10, 10, 20, 20], "score": 0.9, "class": "B"}, # Different class
    ]
    clustered = clusterer_default.cluster_boxes(dets)
    assert len(clustered) == 3
    # Assert specific properties if needed, e.g., classes, scores, box coords
    assert clustered[0]["class"] == "A" and np.allclose(clustered[0]["box"], [10,10,20,20])
    assert clustered[1]["class"] == "A" and np.allclose(clustered[1]["box"], [100,10,20,20])
    assert clustered[2]["class"] == "B" and np.allclose(clustered[2]["box"], [10,10,20,20])


def test_simple_overlapping_clustering():
    """Test case: Simple clustering of two overlapping boxes of the same class."""
    clusterer = BoxClusterer(max_distance=0) # Max dist 0 means they must overlap or be adjacent
    dets = [
        {"box": [10, 10, 20, 20], "score": 0.9, "class": "A"},
        {"box": [20, 20, 20, 20], "score": 0.8, "class": "A"},
    ]
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 1
    assert clustered[0]["class"] == "A"
    assert clustered[0]["score"] == 0.9
    assert np.allclose(clustered[0]["box"], [10, 10, 30, 30])


def test_multiple_clusters_mixed_labels():
    """Test case: Multiple clusters with mixed classes, some overlapping/adjacent, some separated."""
    clusterer = BoxClusterer(max_distance=0) # Only merge if overlapping/adjacent
    dets = [
        {"box": [10, 10, 20, 20], "score": 0.9, "class": "A"},
        {"box": [20, 20, 20, 20], "score": 0.8, "class": "A"}, # Overlapping, AABB dist 0
        {"box": [50, 20, 20, 20], "score": 0.7, "class": "A"}, # Separated from previous A by 20 units
        {"box": [150, 150, 30, 30], "score": 0.7, "class": "B"},
        {"box": [160, 160, 20, 20], "score": 0.95, "class": "B"}, # Overlapping, AABB dist 0
        {"box": [50, 50, 10, 10], "score": 0.6, "class": "C"},
    ]
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 4
    
    # The order of clusters might not be guaranteed, so we'll check based on properties
    a_clusters = [c for c in clustered if c["class"] == "A"]
    b_clusters = [c for c in clustered if c["class"] == "B"]
    c_clusters = [c for c in clustered if c["class"] == "C"]

    assert len(a_clusters) == 2
    assert len(b_clusters) == 1
    assert len(c_clusters) == 1

    # Check 'A' clusters
    assert any(np.allclose(c["box"], [10, 10, 30, 30]) and c["score"] == 0.9 for c in a_clusters)
    assert any(np.allclose(c["box"], [50, 20, 20, 20]) and c["score"] == 0.7 for c in a_clusters)

    # Check 'B' cluster
    assert b_clusters[0]["class"] == "B"
    assert np.allclose(b_clusters[0]["box"], [150, 150, 30, 30])
    assert b_clusters[0]["score"] == 0.95

    # Check 'C' cluster
    assert c_clusters[0]["class"] == "C"
    assert np.allclose(c_clusters[0]["box"], [50, 50, 10, 10])
    assert c_clusters[0]["score"] == 0.6


def test_adjacent_boxes_clustering():
    """Test case: Clustering two adjacent boxes of the same class."""
    clusterer = BoxClusterer(max_distance=0) # AABB dist 0
    dets = [
        {"box": [0, 0, 50, 50], "score": 0.7, "class": "X"},
        {"box": [50, 0, 50, 50], "score": 0.9, "class": "X"},
    ]
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 1
    assert clustered[0]["class"] == "X"
    assert clustered[0]["score"] == 0.9
    assert np.allclose(clustered[0]["box"], [0, 0, 100, 50])


def test_separated_boxes_within_max_distance_merge():
    """Test case: Separated boxes with a gap just within max_distance should merge."""
    clusterer = BoxClusterer(max_distance=5)
    dets = [
        {"box": [0, 0, 10, 10], "score": 0.7, "class": "Y"},
        {"box": [15, 0, 10, 10], "score": 0.9, "class": "Y"}, # 5 units apart horizontally
    ]
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 1
    assert clustered[0]["class"] == "Y"
    assert clustered[0]["score"] == 0.9
    assert np.allclose(clustered[0]["box"], [0, 0, 25, 10])


def test_separated_boxes_outside_max_distance_no_merge():
    """Test case: Separated boxes with a gap just outside max_distance should not merge."""
    clusterer = BoxClusterer(max_distance=5)
    dets = [
        {"box": [0, 0, 10, 10], "score": 0.7, "class": "Z"},
        {"box": [16, 0, 10, 10], "score": 0.9, "class": "Z"}, # 6 units apart horizontally
    ]
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 2


def test_empty_detections():
    """Test case: Empty input detections list."""
    clusterer = BoxClusterer(max_distance=10)
    dets = []
    clustered = clusterer.cluster_boxes(dets)
    assert len(clustered) == 0