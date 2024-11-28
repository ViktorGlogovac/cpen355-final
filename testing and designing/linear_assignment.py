import numpy as np
from scipy.optimize import linear_sum_assignment

def min_cost_matching(cost_matrix, max_distance, tracks, detections):
    """
    Perform minimum cost matching using the Hungarian algorithm (linear sum assignment).
    :param cost_matrix: Cost matrix (distance matrix) between tracks and detections
    :param max_distance: Maximum allowed distance for a match
    :param tracks: List of active tracks
    :param detections: List of new detections
    :return: A tuple of matched indices, unmatched tracks, and unmatched detections
    """
    # Debugging: Check the shape and contents of the cost matrix
    print("Cost matrix shape:", cost_matrix.shape)
    print("Cost matrix contents:", cost_matrix)

    if cost_matrix.size == 0:
        return [], np.arange(len(tracks)), np.arange(len(detections))

    # Check if any elements are invalid (infinite, NaN)
    if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
        raise ValueError("Cost matrix contains invalid values (NaN or Inf).")

    # Solve the linear sum assignment problem (Hungarian algorithm)
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    # Filter out matches where the cost exceeds the max distance
    matches, unmatched_tracks, unmatched_detections = [], [], []

    for t, d in zip(track_indices, detection_indices):
        if cost_matrix[t, d] > max_distance:
            unmatched_tracks.append(t)
            unmatched_detections.append(d)
        else:
            matches.append((t, d))

    unmatched_tracks += list(set(range(len(tracks))) - set(track_indices))
    unmatched_detections += list(set(range(len(detections))) - set(detection_indices))

    return matches, unmatched_tracks, unmatched_detections
