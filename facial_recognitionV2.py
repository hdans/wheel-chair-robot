import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time
from controller import ArduinoController

# --- KONFIGURASI UTAMA ---
ARDUINO_PORT = 'COM8'
FRAME_SKIP = 2
CAM_WIDTH, CAM_HEIGHT = 640, 480
osd_font = cv2.FONT_HERSHEY_SIMPLEX

# --- SENSITIVITAS ---
EAR_THRESHOLD = 0.30  # Batas merem
LONG_BLINK_DURATION = 2.0 

# --- VARIABEL GLOBAL UTAMA ---
running = True
current_action = "STOP"
current_ear_value = 0.0  # Variabel baru untuk menampung nilai mata agar bisa ditampilkan
is_system_paused = False
eyes_closed_start_time = None
pause_toggle_lock = False
baseline_yaw = 0
baseline_pitch = 0
calibrated = False
last_command = "STOP"
frame_counter = 0

# Lock untuk thread safe
result_lock = threading.Lock()

# Inisialisasi
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
    print("ERROR: Kamera tidak terdeteksi!", flush=True)
    exit()

print("Sistem Mulai... Tunggu sebentar...", flush=True)

# Fungsi Matematika
def get_distance(p1, p2):
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y])

def calculate_ear(landmarks, indices):
    p_top = landmarks[indices[0]]
    p_bot = landmarks[indices[1]]
    p_left = landmarks[indices[2]]
    p_right = landmarks[indices[3]]
    return get_distance(p_top, p_bot) / (get_distance(p_left, p_right) + 1e-6)

def check_long_blink_toggle(landmarks):
    global is_system_paused, eyes_closed_start_time, pause_toggle_lock, current_ear_value

    left_ear = calculate_ear(landmarks, [159, 145, 33, 133])
    right_ear = calculate_ear(landmarks, [386, 374, 362, 263])
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Update global variable untuk ditampilkan di layar
    current_ear_value = avg_ear

    # Print Log Paksa (Flush=True)
    if frame_counter % 15 == 0:
        state = "MEREM" if avg_ear < EAR_THRESHOLD else "MELEK"
        print(f"EAR: {avg_ear:.3f} | Status: {state}", flush=True)

    if avg_ear < EAR_THRESHOLD:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = time.time()
        else:
            elapsed = time.time() - eyes_closed_start_time
            if elapsed > LONG_BLINK_DURATION and not pause_toggle_lock:
                is_system_paused = not is_system_paused
                pause_toggle_lock = True
                print(f"!!! SYSTEM TOGGLE: {is_system_paused} !!!", flush=True)
    else:
        eyes_closed_start_time = None
        pause_toggle_lock = False

    return is_system_paused

def detect_logic(frame, landmarks, img_w, img_h):
    global baseline_yaw, baseline_pitch

    # Cek Pause/Resume
    paused = check_long_blink_toggle(landmarks)
    if paused: return "STOP"

    # Head Pose Logic
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
        (-30.0, 0.0, 30.0), (30.0, 0.0, 30.0), (0.0, 0.0, 0.0),
        (-20.0, -30.0, 20.0), (20.0, -30.0, 20.0), (0.0, -60.0, 0.0)
    ])

    cam_matrix = np.array([[img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]], dtype="double")
    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, np.zeros((4,1)))
    
    if not success: return "STOP"

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_mat = np.hstack((rot_mat, trans_vec))
    euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
    pitch, yaw, roll = [math.degrees(math.radians(a)) for a in euler]

    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    mouth_open = abs(lip_bottom.y - lip_top.y) > 0.05

    if not calibrated: return "STOP"

    dyaw = yaw - baseline_yaw
    dpitch = pitch - baseline_pitch

    if mouth_open: return "BACKWARD"
    if dyaw > 20: return "RIGHT"
    elif dyaw < -20: return "LEFT"
    elif dpitch < -15: return "FORWARD"
    else: return "STOP"

# --- THREADS ---
def detection_thread():
    global current_action, frame_counter
    while running:
        ret, frame = cap.read()
        if not ret: continue
        
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0: continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for lms in results.multi_face_landmarks:
                act = detect_logic(frame, lms.landmark, w, h)
                with result_lock: current_action = act
        else:
            with result_lock: current_action = "STOP"
        time.sleep(0.01)

def serial_thread():
    global last_command
    while running:
        with result_lock: act = current_action
        
        if is_system_paused: act = "STOP"
        
        if act != last_command:
            arduino.send_command(act)
            last_command = act
        time.sleep(0.1)

# Mulai Thread
t1 = threading.Thread(target=detection_thread, daemon=True)
t2 = threading.Thread(target=serial_thread, daemon=True)
t1.start()
t2.start()

print("Program berjalan. LIHAT LAYAR VIDEO UNTUK DEBUG.", flush=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        # Ambil data dari thread dengan aman
        with result_lock:
            act_disp = "PAUSED" if is_system_paused else current_action
            ear_disp = current_ear_value

        # --- VISUALISASI UTAMA (DEBUG DI SINI) ---
        
        # 1. Tampilkan EAR di pojok kanan atas
        # Jika nilai < 0.30 maka teks jadi MERAH (Terdeteksi Merem)
        ear_color = (0, 0, 255) if ear_disp < EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Eye EAR: {ear_disp:.3f}", (400, 30), osd_font, 0.7, ear_color, 2)
        cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (400, 55), osd_font, 0.5, (255, 255, 0), 1)

        # 2. Status Aksi
        cv2.putText(frame, f"ACTION: {act_disp}", (10, 40), osd_font, 1.0, (255, 0, 255), 2)

        # 3. Bar Progress Merem
        if eyes_closed_start_time is not None:
             elapsed = time.time() - eyes_closed_start_time
             bar_width = int(min(elapsed / LONG_BLINK_DURATION, 1.0) * 200)
             cv2.rectangle(frame, (10, 70), (10 + bar_width, 90), (0, 165, 255), -1)
             cv2.putText(frame, "HOLD TO TOGGLE...", (10, 65), osd_font, 0.5, (0, 165, 255), 1)

        cv2.imshow("DEBUG MODE - CONTROL", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
             calibrated = True
             baseline_yaw, baseline_pitch = 0, 0
             print("Kalibrasi OK", flush=True)
        if key == 27: break

except KeyboardInterrupt: pass
finally:
    running = False
    cap.release()
    arduino.close()
    cv2.destroyAllWindows()