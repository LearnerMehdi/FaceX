# built-in dependencies
from typing import Any, Dict, List, Union, Optional, Sequence, IO
from collections import defaultdict

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import image_utils
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition

from utils.data import SortV2

def _format_batch_for_tracker(batch_regions, batch_confidences):
    dets = []
    for region, confidence in zip(batch_regions, batch_confidences):
        x1 = region["x"]
        y1 = region["y"]
        x2 = x1 + region["w"]
        y2 = y1 + region["h"]
        dets.append([x1, y1, x2, y2, confidence])
    dets = np.asarray(dets)
    dets = np.atleast_2d(dets)
    return dets

def represent(
    img_path: Union[str, IO[bytes], np.ndarray, Sequence[Union[str, np.ndarray, IO[bytes]]]],
    tracker: SortV2,
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Represent facial images with object tracking integration.

    Args:
        img_path: Single image (str, np.ndarray, or IO[bytes])
        tracker: SortV2 tracker instance
        model_name: Model for face recognition
        enforce_detection: If no face detected, raise exception
        detector_backend: Face detector backend
        align: Perform alignment based on eye positions
        expand_percentage: Expand detected facial area
        normalization: Normalize input image
        anti_spoofing: Enable anti spoofing
        max_faces: Limit number of faces to process

    Returns:
        Dict with:
        - detected_faces: List of all detected faces with facial_area, face_confidence
        - new_faces: List of new faces with embeddings (only faces needing recognition)
        - active_tracks: List of active track IDs from tracker
        - track_mapping: Dict mapping track_id to face data
    """
    # Initialize model
    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )
    
    # Extract faces from image
    if detector_backend != "skip":
        img_objs = detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
    else:
        # Skip detection - use full image
        img, _ = image_utils.load_image(img_path)
        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")
        img = img[:, :, ::-1]  # Convert to RGB
        img_objs = [{
            "face": img,
            "facial_area": {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0]},
            "confidence": 0,
        }]

    if not img_objs:
        return {
            "detected_faces": [],
            "new_faces": [],
            "active_tracks": [],
            "track_mapping": {}
        }

    # Sort by largest facial areas if max_faces limit exists
    if max_faces is not None and max_faces < len(img_objs):
        img_objs = sorted(
            img_objs,
            key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
            reverse=True,
        )[:max_faces]

    # Prepare detections for tracker
    batch_regions = []
    batch_confidences = []
    
    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")
        
        batch_regions.append(img_obj["facial_area"])
        batch_confidences.append(img_obj["confidence"])

    # Format detections for tracker
    dets = _format_batch_for_tracker(batch_regions, batch_confidences)
    
    # Update tracker and get tracks
    tracks = tracker.update(dets=dets)
    
    # Handle empty tracks case
    if tracks.size == 0:
        tracks = np.empty((0, 5))
    
    # Identify new vs existing tracks
    new_track_ids = []
    existing_track_ids = []
    active_tracks = []
    
    for track in tracks:
        track_id = int(track[4])  # track_id is in column 4
        active_tracks.append(track_id)
        
        if track_id not in tracker.unique_identities:
            new_track_ids.append(track_id)
            tracker.unique_identities[track_id] = True
        else:
            existing_track_ids.append(track_id)

    # Prepare data for all detected faces
    detected_faces = []
    new_faces = []
    track_mapping = {}
    
    for i, (img_obj, region, confidence) in enumerate(zip(img_objs, batch_regions, batch_confidences)):
        face_data = {
            "facial_area": region,
            "face_confidence": confidence
        }
        
        # If this face corresponds to a track, add track ID
        if i < len(tracks):
            track_id = int(tracks[i][4])
            face_data["track_id"] = track_id
            track_mapping[track_id] = face_data
            
            if track_id in new_track_ids:
                # Process this face for recognition
                img = img_obj["face"]
                
                # Convert RGB to BGR for preprocessing
                img = img[:, :, ::-1]
                
                # Resize to model input size
                target_size = model.input_shape
                img = preprocessing.resize_image(
                    img=img,
                    target_size=(target_size[1], target_size[0]),
                )
                
                # Normalize input
                img = preprocessing.normalize_input(img=img, normalization=normalization)
                
                # Add to batch for recognition
                new_faces.append({
                    "facial_area": region,
                    "face_confidence": confidence,
                    "track_id": track_id,
                    "processed_image": img
                })
        
        detected_faces.append(face_data)

    # Run recognition only on new faces
    if new_faces:
        batch_images = np.array([face["processed_image"][0] for face in new_faces])
        print(batch_images.shape)
        embeddings = model.forward(batch_images)
        print(embeddings)
        print(len(embeddings))
        
        # Add embeddings to new faces
        for i, face in enumerate(new_faces):
            face["embedding"] = embeddings[i]

    return {
        "detected_faces": detected_faces,
        "new_faces": new_faces,
        "active_tracks": active_tracks,
        "track_mapping": track_mapping
    }
