import cv2, csv, os, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from shared import *
import time


if not os.path.exists(CSV_FILE):
    header = [f'L{i}{c}' for i in L_IDX for c in 'xyz'] + [f'R{i}{c}' for i in R_IDX for c in 'xyz'] + ['label']
    with open(CSV_FILE, 'w', newline='') as f: csv.writer(f).writerow(header)


with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    for note in NOTES:
        count, timer = 0, 0
        is_waiting = True
        while count < TARGET:
            ret, frame = cap.read()
            if not ret:
                cleanup(cap)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            hands = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
            #hands = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)) # for image mode
            timer += 1
            if is_waiting:
                remaining = max(0, SWITCH - timer)
                
                if timer >= SWITCH:
                    is_waiting = False
                    timer = COOLDOWN
            else:
                if timer >= COOLDOWN:
                    # if len(hands.hand_landmarks) == 1:
                    #     print(hands.handedness[0][0].category_name) # debug hand right/left
                    if len(hands.hand_landmarks) == 2:
                        h = {hands.handedness[i][0].category_name: normalize(hands.hand_landmarks[i], hands.handedness[i][0].category_name) for i in range(2)}
                        if h.get("Left") is not None and h.get("Right") is not None:
                            with open(CSV_FILE, 'a', newline='') as f:
                                csv.writer(f).writerow(np.append(np.concatenate([h["Left"], h["Right"]]), note))
                            count, timer = count + 1, 0
            if landmarks:
                annotated = draw_landmarks_on_image(rgb_frame, hands)
                frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            if reference:
                frame[10:10+ref.shape[0], -210:-10] = ref

            if is_waiting:
                cv2.putText(frame, f"SWITCH TO: {note.upper()}", (50, 200), 2, 1.2, (0, 255, 255), 3)

            

            cv2.putText(frame, f"{note.upper()}: {count}/{TARGET}", (20, 50), 2, 1, (0,255,0), 2)
            cv2.imshow('data capture', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cleanup(cap)
            elif key == ord('l'):
                landmarks = not landmarks
            elif key == ord('r'):
                reference = not reference