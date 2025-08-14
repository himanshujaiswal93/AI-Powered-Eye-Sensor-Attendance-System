import face_recognition
import numpy as np
from pathlib import Path

def get_face_embedding(image_path: str):
    """
    Takes an image path, detects the face, and returns a unique embedding vector.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        raise ValueError("No face detected in the image.")
    elif len(face_locations) > 1:
        raise ValueError("Multiple faces detected. Provide an image with a single face.")
    
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0]

if __name__ == "__main__":
    try:
        embedding = get_face_embedding("man.png")  # Pass file name here
        print("Face Embedding Vector:\n", embedding)
        print("Embedding Length:", len(embedding))
    except Exception as e:
        print


        

