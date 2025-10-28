# controller.py
import serial
import time
import sys

class ArduinoController:
    """
    Kelas untuk mengelola komunikasi serial dengan Arduino.
    """
    def __init__(self, port='COM3', baudrate=9600):
        """
        Inisialisasi koneksi serial.
        """
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.dummy_mode = False

        try:
            self.arduino = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
            # Jeda 2 detik seperti yang diminta, agar Arduino sempat reset
            print(f"Menghubungkan ke Arduino di port {self.port}...")
            time.sleep(2)
            print("Arduino Terhubung.")
            self.dummy_mode = False
        except serial.SerialException as e:
            print(f"Error: Tidak dapat terhubung ke Arduino di {self.port}.")
            print("Program berjalan dalam mode DUMMY (hanya log).")
            print(f"Detail error: {e}")
            self.dummy_mode = True
        except Exception as e:
            print(f"Error tak terduga: {e}")
            sys.exit(1)

    def send_command(self, command):
        """
        Mengirim perintah ke Arduino jika terhubung,
        atau mencetak log jika dalam mode dummy.
        """
        if not self.dummy_mode and self.arduino and self.arduino.is_open:
            try:
                # Tambahkan newline (b'\n') agar dibaca oleh readStringUntil('\n') di Arduino
                self.arduino.write(f"{command}\n".encode('utf-8'))
                # print(f"Perintah terkirim: {command}") # Opsional: untuk debug
            except Exception as e:
                print(f"Error saat mengirim data: {e}")
                self.dummy_mode = True # Masuk ke mode dummy jika ada error penulisan
        else:
            # Mode dummy seperti yang diminta
            print(f"Log (Dummy): Perintah -> {command}")

    def close(self):
        """
        Menutup koneksi serial dengan aman.
        """
        if not self.dummy_mode and self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Koneksi Arduino ditutup.")

# Untuk pengujian cepat (opsional)
if __name__ == "__main__":
    # Ganti 'COM3' dengan port Anda
    controller = ArduinoController(port='COM3') 
    
    try:
        controller.send_command("FORWARD")
        time.sleep(1)
        controller.send_command("STOP")
        time.sleep(1)
        controller.send_command("LEFT")
    finally:
        controller.close()