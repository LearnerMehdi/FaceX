from typing import List
from dataclasses import dataclass

@dataclass
class EmbeddingDataClass:
    identity: str
    embedding: List

@dataclass
class EmbeddingDataBase:
    data: List[EmbeddingDataClass]

    def append(self, data_class: EmbeddingDataClass) -> None:
        self.data.append(data_class)