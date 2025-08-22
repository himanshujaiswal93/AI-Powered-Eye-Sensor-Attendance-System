import cv2
import mediapipe as mp
import numpy as np

# Load image
image = cv2.imread("test_images/unknown2.jpg")   # 👈 replace with your image
h, w, _ = image.shape

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Left iris (468–471), Right iris (473–476)
        iris_indices = {
            "left": [468, 469, 470, 471],
            "right": [473, 474, 475, 476]
        }

        for eye, indices in iris_indices.items():
            iris_points = []
            for idx in indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                iris_points.append((x, y))

            iris_points = np.array(iris_points, dtype=np.int32)
            (x, y), radius = cv2.minEnclosingCircle(iris_points)
            center = (int(x), int(y))
            radius = int(radius)

            # Create mask
            mask = np.zeros_like(image)
            cv2.circle(mask, center, radius, (255, 255, 255), -1)

            # Apply mask
            iris_crop = cv2.bitwise_and(image, mask)
            iris_crop = iris_crop[center[1]-radius:center[1]+radius,
                                  center[0]-radius:center[0]+radius]

            # Save output
            filename = f"{eye}_iris_clean.jpg"
            cv2.imwrite(filename, iris_crop)
            print(f"✅ {eye.capitalize()} iris saved as {filename}")
