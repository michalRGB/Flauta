import cv2, joblib, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

L_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
R_IDX = L_IDX + [17, 18, 19, 20]

def normalize(lms, label):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms])
    norm = (arr - arr[0]) / (np.linalg.norm(arr[9] - arr[0]))
    idx = R_IDX if label == "Right" else L_IDX
    return norm[idx].flatten()

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#taken directly from google ai edge collab
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN


  return annotated_image

def cleanup(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

CSV_FILE, NOTES, TARGET, COOLDOWN, SWITCH = "recorder_dataset.csv", ["c1","d1","e1","f1","g1","a1","h1"], 50, 20, 200

ref = cv2.imread("notes.png")
ref = cv2.resize(ref, (200, int(200 * ref.shape[0]/ref.shape[1])))

landmarks = True
reference = True

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.75, min_tracking_confidence=0.75  
)