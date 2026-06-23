import cv2
import numpy as np
import time
import os
import threading
from gtts import gTTS
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# --- STABILITY SETTINGS ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Load Model and Labels
try:
    model = load_model("model.h5", compile=False)
    labels = np.load("labels.npy")
    print(f"Model Ready: {labels}")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Configuration
FRAME_REQUIRED = 40
CONFIDENCE_THRESHOLD = 0.85

cap = cv2.VideoCapture(0)
# Lower resolution slightly to save CPU/RAM and prevent crashes
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = HandDetector(detectionCon=0.7, maxHands=2)

cv2.namedWindow("Stability Mode", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stability Mode", 1920, 1080)

sequence = []
sentence = ""
last_spoken = ""
last_time = 0
is_speaking = False # New flag to prevent audio overlap

def play_audio(text):
    global is_speaking
    is_speaking = True
    try:
        clean_text = text.replace("_", " ")
        print("Speaking:", clean_text)
        tts = gTTS(text=clean_text, lang='en')
        tts.save("voice.mp3")
        # Added /wait so the thread knows when the audio is actually finished
        os.system("start /min /wait voice.mp3")
    except:
        pass
    finally:
        is_speaking = False # Ready for next word

print("System Active. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, draw=True, flipType=False)

    if hands:
        landmarks = []
        for i in range(2):
            if i < len(hands):
                lmList = hands[i]["lmList"]
                wrist = lmList[0]
                for lm in lmList:
                    landmarks.extend([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
            else:
                landmarks.extend([0] * 63)

        sequence.append(landmarks)
        
        # Keep only the last 40 frames to prevent RAM bloat
        if len(sequence) > FRAME_REQUIRED:
            sequence.pop(0)

        # 3. Prediction (Only if we have a full sequence)
        if len(sequence) == FRAME_REQUIRED:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            index = np.argmax(res)
            
            if res[index] > CONFIDENCE_THRESHOLD:
                sentence = labels[index].replace("_", " ")
                
                # Visual UI
                cv2.rectangle(frame, (0,0), (640, 45), (0, 255, 0), -1)
                cv2.putText(frame, f"{sentence} ({int(res[index]*100)}%)", (15, 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 4. Smart Audio (Only plays if not already speaking)
                current_time = time.time()
                if not is_speaking and (sentence != last_spoken or current_time - last_time > 5):
                    audio_thread = threading.Thread(target=play_audio, args=(sentence,))
                    audio_thread.setDaemon(True) # Ensure thread closes with the app
                    audio_thread.start()
                    
                    last_spoken = sentence
                    last_time = current_time

    else:
        # Fast Reset: If no hands, clear the sequence immediately
        sequence = []
        cv2.putText(frame, "Waiting for Hands...", (15, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Stability Mode", frame)

    # Use a slightly longer waitKey for better CPU stability
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
