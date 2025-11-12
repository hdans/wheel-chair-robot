import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
from controller import ArduinoController

# Konfigurasi Utama
ARDUINO_PORT = 'COM3'
FRAME_SKIP = 5
ENABLE_DRAW = True
CAM_WIDTH, CAM_HEIGHT = 320, 240
osd_font = cv2.FONT_HERSHEY_SIMPLEX

ACTION_COLORS = {
    "STOP": (0, 0, 255),
    "FORWARD": (0, 255, 0),
    "LEFT": (0, 255, 255),
    "RIGHT": (255, 0, 255),
    "BACKWARD": (255, 0, 0),
}

# Treshold untuk Deteksi
EAR_THRESHOLD = 0.22  
DOUBLE_BLINK_WINDOW = 1.2 
LIDAH_RED_RATIO = 0.45  


# Inisialisasi Arduino
arduino = ArduinoController(port=ARDUINO_PORT)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    arduino.close()
    exit()

print("Kamera berhasil dibuka dengan resolusi:", CAM_WIDTH, "x", CAM_HEIGHT)
time.sleep(1.0)

frame_queue = queue.Queue(maxsize=1)
result_lock = threading.Lock()

current_action = "STOP"
last_command = "STOP"
frame_counter = 0
running = True
backward_active = False
blink_times = []


def get_distance(p1, p2):
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y])

def eye_aspect_ratio(lm, idx_top, idx_bottom, idx_left, idx_right):
    vert = get_distance(lm[idx_top], lm[idx_bottom])
    horiz = get_distance(lm[idx_left], lm[idx_right])
    return vert / (horiz + 1e-6)

# Deteksi lidah & Mulut dengan tekstur
def detect_mulut(frame, landmarks, img_w, img_h):
    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    lip_left = landmarks[61]
    lip_right = landmarks[291]

    x1 = int(lip_left.x * img_w) - 10
    x2 = int(lip_right.x * img_w) + 10
    y1 = int(lip_top.y * img_h) - 5
    y2 = int(lip_bottom.y * img_h) + 15
    x1, x2 = sorted([max(0, x1), min(img_w - 1, x2)])
    y1, y2 = sorted([max(0, y1), min(img_h - 1, y2)])

    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return False

    eye_left = landmarks[133]
    eye_right = landmarks[362]
    eye_distance = get_distance(eye_left, eye_right)
    mouth_open_ratio = get_distance(lip_top, lip_bottom) / (eye_distance + 1e-6)

    gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = lap.var()  

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return (mouth_open_ratio > 0.3) and (var_lap > 60)


def detect_expression(frame, landmarks, img_w, img_h):
    """Deteksi ekspresi berbasis mulut penuh (tanpa kedipan)."""
    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    lip_left = landmarks[61]
    lip_right = landmarks[291]
    nose_tip = landmarks[1]

    # Region of interest for mouth
    x1 = int(lip_left.x * img_w) - 10
    x2 = int(lip_right.x * img_w) + 10
    y1 = int(lip_top.y * img_h) - 5
    y2 = int(lip_bottom.y * img_h) + 15
    x1, x2 = sorted([max(0, x1), min(img_w - 1, x2)])
    y1, y2 = sorted([max(0, y1), min(img_h - 1, y2)])

    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return "STOP"

    # Measure mouth geometry
    mouth_width = get_distance(lip_left, lip_right)
    mouth_height = get_distance(lip_top, lip_bottom)
    mouth_ratio = mouth_height / (mouth_width + 1e-6)

    # Compute mouth corner displacement relative to nose
    mouth_left_dx = lip_left.x - nose_tip.x
    mouth_right_dx = nose_tip.x - lip_right.x

    # --- Tongue detection (red ratio) ---
    hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = np.sum(mask_red > 0) / (mask_red.size + 1e-6)

    # --- Decision logic ---
    if red_ratio > 0.35:          # Tongue detected
        return "BACKWARD"
    elif mouth_ratio > 0.32:      # Mouth open wide
        return "FORWARD"
    elif mouth_left_dx < -0.065:  # Mouth pulled left
        return "LEFT"
    elif mouth_right_dx < -0.065: # Mouth pulled right
        return "RIGHT"
    else:
        return "STOP"


def detection_thread():
    global current_action, running, frame_counter
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        img_h, img_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                action = detect_expression(frame, face_landmarks.landmark, img_w, img_h)
                with result_lock:
                    current_action = action
                arduino.send_command(current_action) 

        time.sleep(0.01)

def serial_thread():
    global last_command, current_action, running
    while running:
        with result_lock:
            action_to_send = current_action

        if action_to_send != last_command:
            try:
                arduino.send_command(action_to_send)
                last_command = action_to_send
            except Exception as e:
                print("[Serial Warning]", e)
        time.sleep(0.3)

print("Memulai kontrol ekspresi wajah...")
cv2.namedWindow("Kontrol Robot Ekspresi Wajah", cv2.WINDOW_NORMAL)

t_det = threading.Thread(target=detection_thread, daemon=True)
t_ser = threading.Thread(target=serial_thread, daemon=True)
t_det.start()
t_ser.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img_h, img_w, _ = frame.shape
        frame = cv2.flip(frame, 1)

        with result_lock:
            action_display = current_action
        color = ACTION_COLORS.get(action_display, (255, 255, 255))

        if ENABLE_DRAW:
            cv2.putText(frame, f"Aksi: {action_display}",
                        (10, 30), osd_font, 0.8, color, 2)

        cv2.imshow("Kontrol Robot Ekspresi Wajah", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    running = False
    cap.release()
    face_mesh.close()
    arduino.close()
    cv2.destroyAllWindows()
    print("Program selesai.")
