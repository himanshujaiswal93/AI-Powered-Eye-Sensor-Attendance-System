import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.dist(point1, point2)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_indices):
    # Vertical distances
    v1 = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    # Horizontal distance
    h = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Eye landmark indices for Mediapipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
BLINK_FRAMES = 2
blink_counter = 0

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_FRAMES:
                        print("👀 Blink Detected!")
                    blink_counter = 0

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

