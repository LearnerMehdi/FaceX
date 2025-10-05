from utils.embedding import build_embedding_gallery, query_embedding_gallery, get_embedding
from utils.data import EmbeddingDataBase, EmbeddingDataClass

# embeddings = get_embedding(
#     img_path=[
#         "unlabeled_images/who.jpg",
#         "unlabeled_images/who.jpg"
#     ]
# )

# identities = query_embedding_gallery(
#     embedding_query=embeddings,
#     embedding_file_path="./database/embeddings/embeddings.pkl",
#     distance_metric="cosine"
# )

# print(identities)

import cv2

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame.shape)
    # frame = frame.transpose([2, 0, 1])
    print(frame.shape)

    embeddings = get_embedding(img_path=frame)
    print(len(embeddings))
    identities = query_embedding_gallery(
        embedding_query=embeddings,
        embedding_file_path="./database/embeddings/embeddings.pkl",
        distance_metric="cosine"
        )
    print(identities)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    process_frame(frame)

    if not ret:
        break

    cv2.imshow("Video Capturing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




# embeddings = get_embedding(img_path="unlabeled_images/who.jpg")

from deepface.commons import image_utils

image, _ = image_utils.load_image("./unlabeled_images/image.jpg")
print(image.shape)
# embedding = get_embedding(img_path=image)

