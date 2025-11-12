# controller.py (versi realtime ready)
import serial, time, threading, sys, queue

class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600, timeout=1, retry=3):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retry = retry
        self.command_queue = queue.Queue(maxsize=1)
        self.arrduino = None
        self.dummy_mode = False
        self.running = True
        self.lock = threading.Lock()

        self._connect()

        # Jalankan thread listener (mengambil command dari queue)
        self.thread = threading.Thread(target=self._listen_queue, daemon=True)
        self.thread.start()

    def _connect(self):
        for i in range(self.retry):
            try:
                self.arduino = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                time.sleep(2)
                print(f"[INFO] Arduino connected on {self.port}")
                self.dummy_mode = False
                return
            except serial.SerialException:
                print(f"[WARN] Retry connecting... ({i+1}/{self.retry})")
                time.sleep(1)
        print("[ERROR] Could not connect to Arduino. Running in dummy mode.")
        self.dummy_mode = True

    def send_command(self, cmd: str):
        """Tambahkan command baru ke queue (diproses real-time)."""
        if not cmd:
            return
        if self.command_queue.full():
            self.command_queue.get_nowait()  # buang command lama
        self.command_queue.put_nowait(cmd)

    def read_feedback(self):
        """Baca umpan balik dari Arduino jika ada data masuk."""
        if self.dummy_mode or not self.arduino:
            return
        try:
            if self.arduino.in_waiting:
                line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[RX] {line}")
        except Exception:
            pass

    def _listen_queue(self):
        last_cmd = ""
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.2)
                if cmd != last_cmd:
                    with self.lock:
                        if self.dummy_mode:
                            print(f"[DUMMY] {cmd}")
                        else:
                            self.arduino.write(f"{cmd}\n".encode('utf-8'))
                            self.arduino.flush()
                            print(f"[TX] {cmd}")
                            self.read_feedback()  # 🔥 Tambahkan di sini!
                    last_cmd = cmd
            except queue.Empty:
                continue


    def close(self):
        self.running = False
        if self.arduino and not self.dummy_mode:
            self.arduino.close()
            print("[INFO] Arduino connection closed.")
