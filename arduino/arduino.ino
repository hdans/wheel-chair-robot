// ======================================================
// ARDUINO SERIAL CONTROLLER (Kompatibel dengan controller.py)
// ======================================================

String command = "";  // tempat nyimpan perintah terbaru

// Pin LED dummy (bisa kamu ganti ke pin motor driver)
const int LED_STOP = 13;  // LED bawaan
const int LED_FORWARD = 5;
const int LED_LEFT = 6;
const int LED_RIGHT = 9;
const int LED_BACKWARD = 10;

void setup() {
  Serial.begin(9600);
  pinMode(LED_STOP, OUTPUT);
  pinMode(LED_FORWARD, OUTPUT);
  pinMode(LED_LEFT, OUTPUT);
  pinMode(LED_RIGHT, OUTPUT);
  pinMode(LED_BACKWARD, OUTPUT);

  Serial.println("✅ Arduino siap menerima perintah...");
}

void loop() {
  // Baca perintah dari serial
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();

    Serial.print("📩 Perintah diterima: ");
    Serial.println(command);

    // Matikan semua dulu
    digitalWrite(LED_STOP, LOW);
    digitalWrite(LED_FORWARD, LOW);
    digitalWrite(LED_LEFT, LOW);
    digitalWrite(LED_RIGHT, LOW);
    digitalWrite(LED_BACKWARD, LOW);

    // Nyalakan sesuai command
    if (command == "FORWARD") {
      digitalWrite(LED_FORWARD, HIGH);
    } 
    else if (command == "LEFT") {
      digitalWrite(LED_LEFT, HIGH);
    } 
    else if (command == "RIGHT") {
      digitalWrite(LED_RIGHT, HIGH);
    } 
    else if (command == "BACKWARD") {
      digitalWrite(LED_BACKWARD, HIGH);
    } 
    else if (command == "STOP") {
      digitalWrite(LED_STOP, HIGH);
    }

    // Kirim feedback ke Python (bisa dibaca controller.py)
    Serial.print("✅ Aksi dijalankan: ");
    Serial.println(command);
  }
}
