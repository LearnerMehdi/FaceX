from typing import List

ASCENDING_RANK_METRICS: List = ["cosine", "angular"]
DESCENDING_RANK_METRICS: List = ["euclidean", "euclidean_l2"]

# Parameters for object tracker
MAX_AGE: int = 3
MIN_HITS: int = 2
IOU_THRESHOLD: float = 0.3
