from time import time
import cv2
import numpy as np
# from deepface.modules.representation import represent

from utils.embedding import build_embedding_gallery, query_embedding_gallery, get_embedding
from utils.data import EmbeddingDataBase, EmbeddingDataClass
from utils.config import MAX_AGE, MIN_HITS, IOU_THRESHOLD
from utils.representation import represent
from utils.data import SortV2

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = represent(
        img_path=frame,
        tracker=tracker,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="opencv",
        align=False
    )
    
    # Handle empty results case
    if not results["detected_faces"]:
        return None
    
    # Process new faces for recognition
    if results["new_faces"]:
        print(f"New faces detected: {len(results['new_faces'])}")
        for face in results["new_faces"]:
            embedding = [face["embedding"]]
            identities = query_embedding_gallery(
                embedding_query=embedding,
                embedding_file_path="../database/embeddings/embeddings.pkl",
                distance_metric="cosine"
            )
            identity = identities[0]
            face["identity"] = identity
            
            # Store identity in tracker for future reference
            if "track_id" in face:
                tracker.set_identity(face["track_id"], identity)
                print(f"Stored identity '{identity}' for track ID {face['track_id']}")
    
    # Create a comprehensive face list with all faces (new and existing)
    all_faces = []
    
    # Add all detected faces with their basic info
    for face_data in results["detected_faces"]:
        face_info = {
            "facial_area": face_data["facial_area"],
            "face_confidence": face_data["face_confidence"]
        }
        
        # Add track_id if available
        if "track_id" in face_data:
            face_info["track_id"] = face_data["track_id"]
            
            # Get identity from tracker's database
            identity = tracker.get_identity(face_data["track_id"])
            face_info["identity"] = identity
            
            # If this is a new face, add embedding info
            for new_face in results["new_faces"]:
                if new_face.get("track_id") == face_data["track_id"]:
                    face_info["embedding"] = new_face["embedding"]
                    face_info["identity"] = new_face["identity"]  # Use fresh identity
                    break
        else:
            face_info["identity"] = "Unknown"
        
        all_faces.append(face_info)
    
    return all_faces, results["active_tracks"]

cap = cv2.VideoCapture(0)
tracker = SortV2(
    max_age=MAX_AGE,
    min_hits=MIN_HITS,
    iou_threshold=IOU_THRESHOLD
    )

while True:
    ret, frame = cap.read()
    
    start = time()
    result = process_frame(frame)
    end = time()

    latency = end - start
    fps = round(1 / latency, 3)

    if result is not None:
        all_faces, active_tracks = result
        
        # Draw all detected faces
        for i, face_data in enumerate(all_faces):
            x1 = face_data["facial_area"]["x"]
            y1 = face_data["facial_area"]["y"]
            w = face_data["facial_area"]["w"]
            h = face_data["facial_area"]["h"]
            x2 = x1 + w
            y2 = y1 + h

            # Different colors for new vs existing faces
            color = (0, 255, 0) if face_data.get("embedding") else (0, 255, 255)  # Green for new, Yellow for existing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display identity
            identity = face_data.get("identity", "Unknown")
            cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show track info if available
            if face_data.get("track_id"):
                cv2.putText(frame, f"ID:{face_data['track_id']}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display summary info
        cv2.putText(frame, f"Faces: {len(all_faces)}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Tracks: {len(active_tracks)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if not ret:
        break

    cv2.imshow("Video Capturing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
