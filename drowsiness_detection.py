import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import simpleaudio as sa
import threading
from datetime import datetime
import matplotlib.pyplot as plt

EYE_AR_THRESH = 0.25
DROWSINESS_SECONDS = 3

def load_config():
    global EYE_AR_THRESH, DROWSINESS_SECONDS
    try:
        with open("config.txt", "r") as f:
            for line in f:
                if "EYE_AR_THRESH" in line:
                    EYE_AR_THRESH = float(line.split("=")[1].strip())
                elif "DROWSINESS_SECONDS" in line:
                    DROWSINESS_SECONDS = int(line.split("=")[1].strip())
    except FileNotFoundError:
        print("[WARNING] config.txt not found. Using default values.")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

load_config()
COUNTER = 0
ALARM_ON = False
ALARM_LOGGED = False
ear_values = []  # Store EAR values over time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def sound_alarm():
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    play_obj = wave_obj.play()

def log_drowsiness():
    with open("drowsiness_log.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Drowsiness detected at: {now}\n")
    print(f"[LOGGED] Drowsiness detected at: {now}")

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20

EYE_AR_CONSEC_FRAMES = int(fps * DROWSINESS_SECONDS)
print(f"Webcam FPS: {fps}")
print(f"Alarm after {DROWSINESS_SECONDS} seconds of eyes closed.")
print(f"EAR threshold set to: {EYE_AR_THRESH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        ear_values.append(ear)  # Save EAR for graph later

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
                    log_drowsiness()
                    ALARM_LOGGED = True

                # Flash screen red by overlaying a translucent red rectangle
                overlay = frame.copy()
                red_color = (0, 0, 255)  # BGR for red
                alpha = 0.6  # Transparency factor
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), red_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Larger, bold alert text with blinking effect
                # Blink every 15 frames
                if (COUNTER // 15) % 2 == 0:
                    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        else:
            COUNTER = 0
            ALARM_ON = False
            ALARM_LOGGED = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot EAR over time
plt.figure(figsize=(10, 4))
plt.plot(ear_values, label="EAR")
plt.axhline(y=EYE_AR_THRESH, color='r', linestyle='--', label="Threshold")
plt.title("Sleepiness Score (EAR) Over Time")
plt.xlabel("Frame Number")
plt.ylabel("EAR Value")
plt.legend()
plt.show()
