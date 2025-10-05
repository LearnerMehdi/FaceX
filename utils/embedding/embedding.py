try:
    from utils.data import EmbeddingDataBase, EmbeddingDataClass
    from utils.config import ASCENDING_RANK_METRICS, DESCENDING_RANK_METRICS
except:
    from FaceX.utils.data import EmbeddingDataBase, EmbeddingDataClass
    from FaceX.utils.config import ASCENDING_RANK_METRICS, DESCENDING_RANK_METRICS

from typing import List, Dict, Union
import os
import glob
import pickle
import numpy as np

from deepface.modules.representation import represent
from deepface.modules.verification import find_distance


def build_embedding_gallery(database_dir: str):
    embedding_file_path: str = os.path.join(
        os.path.join(
            os.path.dirname(database_dir),
            "embeddings"
            ),
        "embeddings.pkl"
    )
    person_dirs = [d for d in glob.glob(os.path.join(database_dir, "*")) if os.path.isdir(d)]
    person_pictures = [d for person_dir in person_dirs for d in glob.glob(os.path.join(person_dir, "*"))]

    database = EmbeddingDataBase(data=[])

    for person_picture in person_pictures:
        directory_name = os.path.basename(os.path.dirname(person_picture))
        identity = " ".join(directory_name.split("_"))
        results = represent(img_path=person_picture, model_name="Facenet", 
                  detector_backend="skip", enforce_detection=False, 
                  align=False)
        embedding = results[0]["embedding"]
        data_class = EmbeddingDataClass(identity=identity, embedding=embedding)
        database.append(data_class)

    pickle.dump(
        database,
        open(embedding_file_path, "wb")
    )

def load_embedding_gallery(embedding_file_path: str) -> EmbeddingDataBase:
    return pickle.load(
        open(embedding_file_path, "rb")
    )


def query_embedding_gallery(embedding_query: Union[List, np.ndarray], embedding_file_path: str, distance_metric: str = "cosine"
                            ) -> Dict:
    database = load_embedding_gallery(embedding_file_path=embedding_file_path)

    if distance_metric in ASCENDING_RANK_METRICS:
        agg_func = np.argmax

    elif distance_metric in DESCENDING_RANK_METRICS:
        agg_func = np.argmin

    else:
        raise ValueError("Wrong distance_metric chosen: {}. Choose among {}".format(
            distance_metric,
            ASCENDING_RANK_METRICS + DESCENDING_RANK_METRICS            
            ))  

    embedding_database = [data_class.embedding for data_class in database.data]
    identities = np.asarray([data_class.identity for data_class in database.data])

    distance_map = find_distance(
        alpha_embedding=embedding_query, 
        beta_embedding=embedding_database, 
        distance_metric=distance_metric
        )

    indexes = agg_func(distance_map, axis=0)

    return identities[indexes]

def get_embedding(img_path: Union[str, List]):
    n = 1
    if isinstance(img_path, list):
        n = len(img_path)

    results = represent(
        img_path=img_path,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="opencv",
        align=False
    )

    if n == 1:
        return [results[0]["embedding"]]
    
    embeddings = []
    for face_data in results:
        embeddings.append(face_data[0]["embedding"])
    
    return embeddings

if __name__ == "__main__":
    print(
        load_embedding_gallery(embedding_file_path="../../database/embeddings/embeddings.pkl")
    )