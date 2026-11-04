import cv2, joblib, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from shared import *
import random
from collections import deque, Counter
import time




pred_buffer = deque(maxlen=10)
clf = joblib.load("recorder_note_model.pkl")

game = True
note_target = random.choice(NOTES)
hold_counter = 0
frame_target = 30
last = "WAITING..."

with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cleanup(cap)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000) # hacky but works
        hands = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
        if landmarks:
            annotated = draw_landmarks_on_image(rgb_frame, hands)
            frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        if reference:
            frame[10:10+ref.shape[0], -210:-10] = ref
        msg = last

        if len(hands.hand_landmarks) == 2:
            l_feat = r_feat = None
            for i in range(2):
                lbl = hands.handedness[i][0].category_name
                if lbl == "Left": l_feat = normalize(hands.hand_landmarks[i], lbl)
                if lbl == "Right": r_feat = normalize(hands.hand_landmarks[i], lbl)
            
            if l_feat is not None and r_feat is not None:
                feat = np.concatenate([l_feat, r_feat]).reshape(1, -1)
                probs = clf.predict_proba(feat)[0] # avoid array
                pred = clf.classes_[np.argmax(probs)]
                pred_buffer.append(pred)
                most_common = Counter(pred_buffer).most_common(1)[0][0] # 1 most common 1 st element of first tuple
                last = most_common 
                if game:
                    if most_common == note_target:
                        hold_counter += 1
                        msg = f"{most_common.upper()} {hold_counter}/{frame_target}"
                        if hold_counter >= frame_target:
                            hold_counter = 0
                            note_target = random.choice(NOTES)
                    else:
                        msg = most_common.upper()
                        hold_counter = 0
                else:
                    msg = f"{most_common.upper()} ({np.max(probs):.2f})"
        if game:
            msg = f"target: {note_target.upper()} | {msg}"

        cv2.putText(frame, msg, (20, 50), 2, 1, (255, 255, 255), 2)
        cv2.imshow('clasify', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cleanup(cap)
        elif key == ord('l'):
            landmarks = not landmarks
        elif key == ord('r'):
            reference = not reference
        elif key == ord('g'):
            game = not game