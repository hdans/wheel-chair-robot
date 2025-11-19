// ======================================================
// ROBOT KENDALI SERIAL (Kompatibel dengan controller.py)
// ======================================================
// Fitur: Kendali maju, mundur, kiri, kanan, stop via serial
// + Kecepatan dapat diatur + Pengereman halus
// ======================================================

// --- Pin Motor ---
const int E1 = 5;   // PWM Motor Kiri
const int M1 = 6;   // Arah Motor Kiri
const int E2 = 9;   // PWM Motor Kanan
const int M2 = 10;  // Arah Motor Kanan

// --- Variabel global ---
int speedVal = 100; 
const int brakeStep = 5;  // penurunan kecepatan per step (semakin kecil = lebih halus)
const int brakeDelay = 30; // jeda antar step pengereman (ms)

// --- Variabel input ---
String command = "";  // tempat menyimpan perintah terbaru dari serial

// --- Fungsi Gerakan ---
void maju() {
  digitalWrite(M1, LOW);   // motor kiri maju
  digitalWrite(M2, LOW);   // motor kanan maju
  analogWrite(E1, speedVal);
  analogWrite(E2, speedVal);
  Serial.println("🚗 Maju");
}

void mundur(){
  digitalWrite(E1, LOW);  // motor kiri mundur
  digitalWrite(E2, LOW);  // motor kanan mundur
  analogWrite(M1, speedVal);
  analogWrite(M2, speedVal);
  Serial.println("🚗 Mundur");
}

void belokKiri() {
  // Roda kiri mundur, kanan maju (pivot left)
  digitalWrite(E1, LOW);
  digitalWrite(M2, LOW);
  analogWrite(M1, speedVal);
  analogWrite(E2, speedVal);
  Serial.println("↩️ Belok kiri di tempat");
}

void belokKanan() {
  // Roda kiri maju, kanan mundur (pivot right)
  digitalWrite(M1, LOW);
  digitalWrite(E2, LOW);
  analogWrite(E1, speedVal);
  analogWrite(M2, speedVal);
  Serial.println("↪️ Belok kanan di tempat");
}

void berhenti() {
  // Matikan arah dulu supaya tidak mendorong
  digitalWrite(M1, LOW);
  digitalWrite(M2, LOW);

  // Pengereman halus
  for (int s = speedVal; s >= 0; s -= brakeStep) {
    analogWrite(E1, s);
    analogWrite(E2, s);
    delay(brakeDelay);
  }

  analogWrite(E1, 0);
  analogWrite(E2, 0);
  Serial.println("🛑 Robot berhenti dengan halus");
}

// --- Program Utama ---
void setup() {
  Serial.begin(9600);

  pinMode(E1, OUTPUT);
  pinMode(M1, OUTPUT);
  pinMode(E2, OUTPUT);
  pinMode(M2, OUTPUT);

  Serial.println("✅ Robot siap menerima perintah dari Python (controller.py)");
  Serial.println("Perintah: FORWARD | BACKWARD | LEFT | RIGHT | STOP | SPEEDxxx");
  Serial.print("Kecepatan awal: ");
  Serial.println(speedVal);
}

void loop() {
  // Baca perintah dari serial
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();

    Serial.print("📩 Perintah diterima: ");
    Serial.println(command);

    // --- Eksekusi Perintah ---
    if (command == "FORWARD") {
      maju();
    } 
    else if (command == "BACKWARD") {
      mundur();
    } 
    else if (command == "LEFT") {
      belokKiri();
    } 
    else if (command == "RIGHT") {
      belokKanan();
    } 
    else if (command == "STOP") {
      berhenti();
    }
    else if (command.startsWith("SPEED")) {
      int newSpeed = command.substring(5).toInt();
      if (newSpeed >= 0 && newSpeed <= 255) {
        speedVal = newSpeed;
        Serial.print("⚙️ Kecepatan diubah ke: ");
        Serial.println(speedVal);
      } else {
        Serial.println("❌ Nilai kecepatan tidak valid (0–255).");
      }
    }
    else {
      Serial.println("❓ Perintah tidak dikenali. Gunakan FORWARD, BACKWARD, LEFT, RIGHT, STOP, atau SPEEDxxx.");
    }

    // Kirim feedback ke Python
    Serial.print("✅ Aksi dijalankan: ");
    Serial.println(command);
  }
}
