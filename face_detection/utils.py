import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from numpy import dot
from numpy.linalg import norm


# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0, )

_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0)
    return _face_app

def extract_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    selected_frame = None

    while success:
        selected_frame = frame
        break 
    cap.release()

    if selected_frame is not None:
        return cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    return None

def enhance_image(image):
    image = apply_clahe(image)
    gamma = 1.2  # Slight gamma correction
    look_up_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)

def apply_clahe(image):
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel back with A and B
    limg = cv2.merge((cl, a, b))

    # Convert LAB back to RGB
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

def get_embedding(image_input, target_size=(224, 224)):
    if isinstance(image_input, np.ndarray):
        try:
            image_input = enhance_image(image_input)
            image_input = Image.fromarray(image_input).convert("RGB")
        except Exception as e:
            print("Error converting from NumPy to PIL:", e)
            return None
    elif isinstance(image_input, Image.Image):
        image_input = image_input.convert("RGB")
    else:
        print("Invalid image input type:", type(image_input))
        return None
    try:
        image_resized = image_input.resize(target_size)
        
    except Exception as e:
        print("Resize failed:", e)
        return None

    # Convert resized image back to NumPy array
    image_array = np.array(image_resized)

    # Run face detection and get embedding
    app = get_face_app()
    
    faces = app.get(image_array)
    for face in faces:
        print("Detection score:", face.det_score)
    if faces and len(faces) > 0:
        return faces[0].embedding
    return None

def compare_embeddings(emb1, emb2, threshold=1.0):
    if emb1 is None or emb2 is None:
        return False
    distance = np.linalg.norm(emb1 - emb2)
    print(distance, "----------------------------------------")
    return distance < threshold

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return False
    sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    print("Cosine similarity:-------------------------", sim)
    # return sim > 0.35  # You can adjust this (0.3–0.4 is common)
    return sim > 0.15
