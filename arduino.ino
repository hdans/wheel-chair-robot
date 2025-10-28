// arduino_dummy.ino

// Pin LED bawaan (biasanya pin 13 di Arduino Uno)
const int ledPin = LED_BUILTIN;

String command; // Variabel untuk menyimpan perintah dari Python

void setup() {
  // Inisialisasi Serial Monitor
  Serial.begin(9600);
  while (!Serial) {
    ; // Tunggu koneksi serial (penting untuk beberapa board)
  }

  // Set pin LED sebagai OUTPUT
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW); // Pastikan LED mati
  
  Serial.println("Arduino siap menerima perintah...");
}

void loop() {
  // Cek apakah ada data serial yang masuk
  if (Serial.available() > 0) {
    // Baca data sampai karakter newline ('\n')
    command = Serial.readStringUntil('\n');
    
    // Hapus whitespace (jika ada)
    command.trim(); 

    // Proses perintah
    if (command == "FORWARD") {
      Serial.println("Status: Bergerak MAJU");
      blinkLED();
    } 
    else if (command == "BACKWARD") {
      Serial.println("Status: Bergerak MUNDUR");
      blinkLED();
    } 
    else if (command == "LEFT") {
      Serial.println("Status: Belok KIRI");
      blinkLED();
    } 
    else if (command == "RIGHT") {
      Serial.println("Status: Belok KANAN");
      blinkLED();
    } 
    else if (command == "STOP") {
      Serial.println("Status: BERHENTI");
      // Tidak berkedip saat berhenti
    }
    else {
      // Perintah tidak dikenal
      Serial.print("Perintah tidak dikenal: ");
      Serial.println(command);
    }
  }
}

// Fungsi helper untuk mengedipkan LED
void blinkLED() {
  digitalWrite(ledPin, HIGH);
  delay(300); // Kedip selama 300ms
  digitalWrite(ledPin, LOW);
}