import cv2
import math
from tkinter import *
from PIL import Image, ImageTk
import mediapipe as md
import winsound
import time

# Importing necessary libraries
md_drawing = md.solutions.drawing_utils
md_face_mesh = md.solutions.face_mesh

# Landmark indices for both eyes
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Landmark indices for lips
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Variables for blink detection
prev_eye_open_state = None
blink_counter = 0

# Variables for yawn detection
prev_mouth_open_state = None
yawn_counter = 0

# Variables for alert
eyes_closed_start_time = None
ALERT_THRESHOLD = 3  # Seconds

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Function to check if an eye is open
def is_eye_open(landmarks, left=True):
    if left:
        eye_top = landmarks[386]
        eye_bottom = landmarks[374]
    else:
        eye_top = landmarks[159]
        eye_bottom = landmarks[145]
    distance = calculate_distance(eye_top, eye_bottom)
    return distance > 0.02  # Adjust threshold as necessary

# Function to check if mouth is open
def is_mouth_open(landmarks):
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]
    distance = calculate_distance(mouth_top, mouth_bottom)
    return distance > 0.05  # Adjust threshold as necessary

# Function to draw landmarks on the image
def draw_landmarks(image, landmarks, indices, color=(0, 255, 0)):
    for idx in indices:
        landmark = landmarks[idx]
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 2, color, -1)

# Initialize Face Mesh object for face detection
face_mesh = md_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to process each frame
def process_frame():
    global prev_eye_open_state, prev_mouth_open_state, blink_counter, yawn_counter, eyes_closed_start_time

    success, image = cap.read()
    if not success:
        print("Empty Camera")
        return

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            # Draw landmarks for eyes and lips
            draw_landmarks(image, face_landmarks.landmark, LEFT_EYE)
            draw_landmarks(image, face_landmarks.landmark, RIGHT_EYE)
            draw_landmarks(image, face_landmarks.landmark, LIPS)
            
            # Check eye and mouth state
            left_eye_open = is_eye_open(face_landmarks.landmark, left=True)
            right_eye_open = is_eye_open(face_landmarks.landmark, left=False)
            mouth_open = is_mouth_open(face_landmarks.landmark)

            # Blink Detection
            if prev_eye_open_state is not None:
                if not left_eye_open and not right_eye_open and prev_eye_open_state:
                    # Eyes have just closed
                    blink_counter += 1
                    # If both eyes are closed, start the timer
                    if not eyes_closed_start_time:
                        eyes_closed_start_time = time.time()
                # Check if both eyes are closed
                if not left_eye_open and not right_eye_open:
                    if time.time() - eyes_closed_start_time >= ALERT_THRESHOLD:
                        cv2.putText(image, "Alert: Both Eyes Closed", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        trigger_warning("Both eyes closed!")
            else:
                # Reset the timer if at least one eye is open
                if left_eye_open or right_eye_open:
                    eyes_closed_start_time = None
            prev_eye_open_state = left_eye_open or right_eye_open

            # Yawn Detection
            if prev_mouth_open_state is not None:
                if mouth_open and not prev_mouth_open_state:
                    # Mouth has just opened
                    yawn_counter += 1
                elif not mouth_open and prev_mouth_open_state:
                    # Mouth has just closed
                    pass
            prev_mouth_open_state = mouth_open

            # Add text to image
            left_eye_state_text = "open" if left_eye_open else "close"
            right_eye_state_text = "open" if right_eye_open else "close"

            cv2.putText(image, f'Left Eye: {left_eye_state_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Eye: {right_eye_state_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Blinks: {blink_counter}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Yawns: {yawn_counter}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Check if both eyes are closed for the ALERT_THRESHOLD seconds
            if not left_eye_open and not right_eye_open and time.time() - eyes_closed_start_time >= ALERT_THRESHOLD:
                cv2.putText(image, "Alert: Both Eyes Closed", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    frame = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    video_label.config(image=frame)
    video_label.image = frame

    root.after(1, process_frame)

# Function to trigger a warning
def trigger_warning(message):
    print(message)
    winsound.Beep(1000, 500)  # Beep sound as a warning

# Main Tkinter window
root = Tk()
root.title("Driver State Detection")
root.geometry('800x600+268+82')
root.configure(bg="#FFD700")

video_label = Label(root)
video_label.pack()

# Initializing the video capture object
cap = cv2.VideoCapture(0)

# Start processing frames
process_frame()

# Run Tkinter main loop
root.mainloop()

# Release video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
