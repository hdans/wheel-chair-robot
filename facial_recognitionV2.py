import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import queue
import time
from controller import ArduinoController

# Konfigurasi Utama
ARDUINO_PORT = 'COM8'
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

baseline_yaw = None
baseline_pitch = None
calibrated = False

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


def detect_head_pose(frame, landmarks, img_w, img_h):
    global baseline_yaw, baseline_pitch, calibrated

    FACE_POINTS = [33, 263, 1, 61, 291, 199]

    image_points = np.array([
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h),
        (landmarks[199].x * img_w, landmarks[199].y * img_h)
    ], dtype="double")

    model_points = np.array([
        (-30.0, 0.0, 30.0),
        (30.0, 0.0, 30.0),
        (0.0, 0.0, 0.0),
        (-20.0, -30.0, 20.0),
        (20.0, -30.0, 20.0),
        (0.0, -60.0, 0.0)
    ])

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return "STOP"

    rot_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rot_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(a) for a in euler_angles]
    pitch, yaw, roll = [math.degrees(a) for a in (pitch, yaw, roll)]

    # Deteksi buka mulut
    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    mouth_open = abs(lip_bottom.y - lip_top.y) > 0.04

    cv2.putText(frame, f"Pitch: {pitch:.2f}  Yaw: {yaw:.2f}",
            (10, 80), osd_font, 0.7, (255, 255, 255), 2)
    
    # Kalau belum kalibrasi, jangan gerakkan apa-apa
    if not calibrated:
        cv2.putText(frame, "Tekan 'c' untuk kalibrasi wajah netral", (10, 30), osd_font, 0.6, (0, 255, 255), 2)
        return "STOP"

    # Hitung deviasi relatif terhadap baseline
    dyaw = yaw - baseline_yaw
    dpitch = pitch - baseline_pitch

    if mouth_open:
        return "BACKWARD"
    elif dyaw > 20:
        return "RIGHT"
    elif dyaw < -20:
        return "LEFT"
    # elif dpitch > 15:
    #     return "BACKWARD"
    elif dpitch < -15:
        return "FORWARD"
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
                action = detect_head_pose(frame, face_landmarks.landmark, img_w, img_h)
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

        # Proses mediapipe di frame utama agar bisa digunakan saat kalibrasi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        with result_lock:
            action_display = current_action
        color = ACTION_COLORS.get(action_display, (255, 255, 255))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Simpan baseline saat wajah netral
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                FACE_POINTS = [33, 263, 1, 61, 291, 199]
                image_points = np.array([
                    (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),
                    (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h),
                    (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),
                    (face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h),
                    (face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h),
                    (face_landmarks.landmark[199].x * img_w, face_landmarks.landmark[199].y * img_h)
                ], dtype="double")

                model_points = np.array([
                    (-30.0, 0.0, 30.0),
                    (30.0, 0.0, 30.0),
                    (0.0, 0.0, 0.0),
                    (-20.0, -30.0, 20.0),
                    (20.0, -30.0, 20.0),
                    (0.0, -60.0, 0.0)
                ])

                focal_length = img_w
                center = (img_w / 2, img_h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")

                dist_coeffs = np.zeros((4, 1))
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rot_matrix, _ = cv2.Rodrigues(rotation_vector)
                    proj_matrix = np.hstack((rot_matrix, translation_vector))
                    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    pitch, yaw, roll = euler_angles.flatten()

                    baseline_pitch, baseline_yaw = pitch, yaw
                    calibrated = True
                    print(f"[Kalibrasi OK] Pitch={pitch:.2f}, Yaw={yaw:.2f}")

        if ENABLE_DRAW:
            cv2.putText(frame, f"Aksi: {action_display}",
                        (10, 30), osd_font, 0.8, color, 2)
            if not calibrated:
                cv2.putText(frame, "Tekan 'C' untuk kalibrasi wajah netral",
                            (10, 60), osd_font, 0.6, (0, 255, 255), 2)

        cv2.imshow("Kontrol Robot Ekspresi Wajah", frame)
        if key == 27:  # tombol ESC
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

