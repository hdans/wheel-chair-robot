# facial_recognition.py
import cv2
import mediapipe as mp
import time
import numpy as np
from controller import ArduinoController

# --- KONFIGURASI DAN INISIALISASI ---

# 1. Inisialisasi Kontroler Arduino
# Ganti 'COM3' dengan port serial Arduino Anda
ARDUINO_PORT = 'COM3' 
arduino = ArduinoController(port=ARDUINO_PORT)

# 2. Inisialisasi Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,   # << penting agar dia pakai tracking, bukan deteksi penuh
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0)) # Warna mesh hijau

# 3. Inisialisasi Kamera OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    arduino.close()
    exit()

# Beri waktu kamera untuk "pemanasan" (seperti diminta)
print("Kamera sedang disiapkan...")
time.sleep(1.0)
print("Kamera siap.")

# 4. Variabel Helper
# Landmark index untuk kalkulasi
# (Ini adalah indeks standar dari Mediapipe Face Mesh)
LANDMARKS = {
    "lip_top": 13,
    "lip_bottom": 14,
    "lip_left": 61,     # Kiri dari perspektif pengguna
    "lip_right": 291,   # Kanan dari perspektif pengguna
    "eye_left": 133,    # Kiri dari perspektif pengguna
    "eye_right": 362,   # Kanan dari perspektif pengguna
    "brow_left": 107,   # Kiri dari perspektif pengguna
    "brow_right": 336,  # Kanan dari perspektif pengguna
    "nose_tip": 1
}

# Thresholds (ini mungkin perlu disesuaikan)
THRESH = {
    "MOUTH_OPEN": 0.4,
    "BROW_RAISE": 0.3,
    "MOUTH_SKEW_LEFT": 1.4,  # Rasio jarak (kiri/kanan)
    "MOUTH_SKEW_RIGHT": 0.6  # Rasio jarak (kiri/kanan)
}

# Warna visualisasi
ACTION_COLORS = {
    "STOP": (0, 0, 255),       # Merah
    "FORWARD": (0, 255, 0),    # Hijau
    "LEFT": (0, 255, 255),     # Kuning
    "RIGHT": (255, 0, 255),    # Magenta
    "BACKWARD": (255, 0, 0)    # Biru
}

last_command = "STOP"
pTime = 0 # Untuk kalkulasi FPS

# Buat jendela GUI (seperti diminta)
cv2.namedWindow('Kontrol Robot Ekspresi Wajah', cv2.WINDOW_NORMAL)

# --- FUNGSI HELPER ---

def get_normalized_distance(p1, p2):
    """Menghitung jarak euclidean antara dua titik (landmark)."""
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])

def draw_action_visuals(image, action, color, img_w, img_h):
    """Menggambar teks aksi dan panah indikator."""
    # Tampilkan teks Aksi
    cv2.putText(image, f"Aksi: {action}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Tampilkan panah (fitur opsional)
    mid_x, mid_y = img_w // 2, img_h // 2
    arrow_length = 70
    
    if action == "FORWARD":
        cv2.arrowedLine(image, (mid_x, mid_y + arrow_length), (mid_x, mid_y - arrow_length), color, 6)
    elif action == "BACKWARD":
        cv2.arrowedLine(image, (mid_x, mid_y - arrow_length), (mid_x, mid_y + arrow_length), color, 6)
    elif action == "LEFT":
        cv2.arrowedLine(image, (mid_x + arrow_length, mid_y), (mid_x - arrow_length, mid_y), color, 6)
    elif action == "RIGHT":
        cv2.arrowedLine(image, (mid_x - arrow_length, mid_y), (mid_x + arrow_length, mid_y), color, 6)


# --- LOOP UTAMA ---

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Mengabaikan frame kamera yang kosong.")
            continue

        # Hitung FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        img_h, img_w, _ = image.shape

        # Pra-pemrosesan: Flip gambar & konversi BGR ke RGB
        # Flip (cv2.flip(image, 1)) agar seperti cermin
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Optimasi

        # Deteksi Face Mesh
        results = face_mesh.process(image)

        # Post-pemrosesan: Konversi RGB ke BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_action = "STOP" # Default aksi

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Gambar mesh wajah
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=drawing_spec
                )

                # --- LOGIKA DETEKSI EKSPRESI ---
                lm = face_landmarks.landmark
                
                # 1. Normalisasi: Gunakan jarak antar mata sebagai skala
                # (Menggunakan koordinat 3D untuk akurasi)
                eye_left_pt = lm[LANDMARKS["eye_left"]]
                eye_right_pt = lm[LANDMARKS["eye_right"]]
                eye_distance = get_normalized_distance(eye_left_pt, eye_right_pt)

                # 2. Alis Naik (BACKWARD)
                brow_left_pt = lm[LANDMARKS["brow_left"]]
                brow_right_pt = lm[LANDMARKS["brow_right"]]
                brow_dist_left = get_normalized_distance(brow_left_pt, eye_left_pt)
                brow_dist_right = get_normalized_distance(brow_right_pt, eye_right_pt)
                avg_brow_dist = (brow_dist_left + brow_dist_right) / 2
                brow_raise_ratio = avg_brow_dist / eye_distance

                # 3. Mulut Terbuka (FORWARD)
                lip_top_pt = lm[LANDMARKS["lip_top"]]
                lip_bottom_pt = lm[LANDMARKS["lip_bottom"]]
                mouth_open_dist = get_normalized_distance(lip_top_pt, lip_bottom_pt)
                mouth_open_ratio = mouth_open_dist / eye_distance

                # 4. Mulut Miring (LEFT/RIGHT)
                lip_left_pt = lm[LANDMARKS["lip_left"]]
                lip_right_pt = lm[LANDMARKS["lip_right"]]
                nose_tip_pt = lm[LANDMARKS["nose_tip"]]
                dist_left_to_nose = get_normalized_distance(lip_left_pt, nose_tip_pt)
                dist_right_to_nose = get_normalized_distance(lip_right_pt, nose_tip_pt)
                
                # Rasio > 1 = miring kiri, Rasio < 1 = miring kanan
                # Tambahkan epsilon kecil untuk menghindari pembagian dengan nol
                mouth_skew_ratio = dist_left_to_nose / (dist_right_to_nose + 1e-6)

                # --- TENTUKAN AKSI ---
                if brow_raise_ratio > THRESH["BROW_RAISE"]:
                    current_action = "BACKWARD"
                elif mouth_open_ratio > THRESH["MOUTH_OPEN"]:
                    current_action = "FORWARD"
                elif mouth_skew_ratio > THRESH["MOUTH_SKEW_LEFT"]:
                    current_action = "LEFT"
                elif mouth_skew_ratio < THRESH["MOUTH_SKEW_RIGHT"]:
                    current_action = "RIGHT"
                else:
                    current_action = "STOP"

        # Kirim perintah jika aksi berubah
        if current_action != last_command:
            arduino.send_command(current_action)
            last_command = current_action

        # --- VISUALISASI GUI ---
        
        # Tampilkan FPS (Fitur opsional)
        cv2.putText(image, f"FPS: {int(fps)}", (img_w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tampilkan Aksi dan Panah
        action_color = ACTION_COLORS.get(current_action, (255, 255, 255))
        draw_action_visuals(image, current_action, action_color, img_w, img_h)

        # Tampilkan hasil
        cv2.imshow('Kontrol Robot Ekspresi Wajah', image)

        # Keluar jika menekan tombol 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Selalu pastikan resource ditutup dengan aman
    print("\nMenutup program...")
    cap.release()
    arduino.close()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Semua resource ditutup.")