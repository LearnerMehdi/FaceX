from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class EmbeddingDataClass:
    identity: str
    embedding: List

@dataclass
class EmbeddingDataBase:
    data: List[EmbeddingDataClass]

    def append(self, data_class: EmbeddingDataClass) -> None:
        self.data.append(data_class)

from sort.sort import Sort

class SortV2(Sort):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        super().__init__(max_age, min_hits, iou_threshold)
        self.unique_identities = {}  # Track IDs that have been seen
        self.identity_database = {}  # Track ID -> Identity mapping
        self.frame_count = 0
    
    def set_identity(self, track_id: int, identity: str):
        """Store identity for a track ID"""
        self.identity_database[track_id] = identity
    
    def get_identity(self, track_id: int) -> str:
        """Retrieve identity for a track ID"""
        return self.identity_database.get(track_id, "Unknown")
    
    def update(self, dets=None):
        """
        Enhanced update method with better empty handling and identity tracking.
        
        Returns:
            np.ndarray: Tracks array with shape (N, 5) where each row is [x1, y1, x2, y2, track_id]
                       Returns empty array (0, 5) if no tracks are active
        """
        if dets is None:
            dets = np.empty((0, 5))
        
        # Call parent update method
        tracks = super().update(dets)
        
        # Ensure consistent return type - always return numpy array
        if tracks.size == 0:
            return np.empty((0, 5), dtype=float)
        
        return tracks