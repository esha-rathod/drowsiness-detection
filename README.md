Real-Time Drowsiness Detection System

This project detects drowsiness in real time by monitoring eye closure using computer vision. When prolonged eye closure is detected, it sounds an alarm and visually alerts the user.

Features:
- Monitors eye aspect ratio (EAR) using dlib facial landmarks.
- Plays alarm sound and flashes red screen on drowsiness.
- Logs drowsiness events with timestamps.
- Allows configuring sensitivity and duration via config.txt.
- Visualizes EAR values over time after execution.

Setup instructions:

1. Clone the repository:
git clone https://github.com/esha-rathod/drowsiness-detection.git
cd drowsiness-detection

2. (Optional) Create and activate a Python virtual environment: python -m venv drowsy
Windows:
.\drowsy\Scripts\activate
macOS/Linux:
source drowsy/bin/activate

3. Install dependencies:
pip install -r requirements.txt


4. Ensure the file `shape_predictor_68_face_landmarks.dat` is in the project folder. If missing, download it from:  
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
and extract the `.dat` file.

5. Run the program: python drowsiness_detection.py

6. Press 'q' to quit.

Files included:
- `drowsiness_detection.py`: main detection code
- `alarm.wav`: alarm audio
- `shape_predictor_68_face_landmarks.dat`: facial landmark model
- `config.txt`: optional settings for threshold and duration
- `drowsiness_log.txt`: logs detected events
- `.gitignore`: ignores local environment folder `drowsy/`
- `requirements.txt`: project dependencies

Notes:
- Adjust sensitivity by editing `config.txt`.
- Ensure webcam access is allowed.



