/*
  ESP32 OBC controller v3.2

  - Commands the OpenMV over CRC-protected protocol v2.
  - Runs several samples per selected mode to reduce model reloads.
  - Streams source images through RAM blocks, validates them, then discards them.
  - Commits a sample only after the result and file integrity are validated.
  - Emits combined OpenMV/OBC telemetry as JSON Lines over USB Serial.

  Wiring defaults:
    ESP32 RX16 <- OpenMV P4/TX3
    ESP32 TX17 -> OpenMV P5/RX3
    ESP32 GPIO4 -> OpenMV P0/Wake
*/

#include <Arduino.h>

HardwareSerial PayloadSerial(2);

// ----------------------------- Hardware/config -----------------------------
static const int PAYLOAD_RX_PIN = 16;
static const int PAYLOAD_TX_PIN = 17;
static const int WAKE_PIN = 4;
static const uint32_t UART_BAUD = 460800;
static const size_t UART_RX_BUFFER_BYTES = 32768;

static const uint8_t SAMPLES_PER_MODE_PASS = 5;
static const uint32_t MISSION_PERIOD_MS = 3000;
static const uint8_t MAX_PASS_RETRIES = 3;

// The ESP32 has no SD. Structured output is emitted at 115200 baud and can be
// captured on the host under paper_tests/test_2 with capture_serial.py.

// ------------------------------- Protocol v2 -------------------------------
static const uint8_t PROTOCOL_VERSION = 2;
static const uint8_t MSG_READY = 0x01;
static const uint8_t MSG_ACK = 0x02;
static const uint8_t MSG_RESULT = 0x04;
static const uint8_t MSG_ERROR = 0x05;
static const uint8_t MSG_DONE = 0x06;
static const uint8_t MSG_DEBUG = 0x07;
static const uint8_t MSG_COMMAND = 0x10;
static const uint8_t MSG_FILE_BEGIN = 0x20;
static const uint8_t MSG_FILE_CHUNK = 0x21;
static const uint8_t MSG_FILE_END = 0x22;

static const size_t HEADER_PREFIX_SIZE = 16;
static const size_t HEADER_SIZE = 20;
static const uint32_t MAX_FRAME_PAYLOAD = 16384;
static const uint32_t READY_TIMEOUT_MS = 8000;
static const uint32_t ACK_TIMEOUT_MS = 3000;
static const uint32_t FRAME_TIMEOUT_MS = 5000;
static const uint32_t TRANSACTION_NO_PROGRESS_MS = 120000;

static const char* MODE_SEQUENCE[] = {"RGB_CLOUD", "THERMAL_FIRE"};
static const uint8_t MODE_COUNT = sizeof(MODE_SEQUENCE) / sizeof(MODE_SEQUENCE[0]);

struct Frame {
  uint8_t type = 0;
  uint16_t sequence = 0;
  uint16_t flags = 0;
  uint32_t length = 0;
  uint8_t* payload = nullptr;
};

struct LinkCounters {
  uint32_t txFrames = 0;
  uint32_t rxFrames = 0;
  uint32_t txBytes = 0;
  uint32_t rxBytes = 0;
  uint32_t headerCrcErrors = 0;
  uint32_t payloadCrcErrors = 0;
  uint32_t lengthErrors = 0;
  uint32_t timeouts = 0;
  uint32_t sequenceMismatches = 0;
} linkStats;

struct SampleMetrics {
  uint32_t wakeToReadyMs = 0;
  uint32_t pingRttMs = 0;
  uint32_t transactionMs = 0;
  uint32_t fileRxMs = 0;
  uint32_t fileBytes = 0;
  uint32_t fileCrcCalculated = 0;
  uint32_t fileCrcExpected = 0;
  bool fileSeen = false;
  bool fileValid = true;
  bool resultSeen = false;
  bool noMoreImages = false;
  String sampleId;
  String resultJson;
  String faultDomain = "NONE";
  String faultCode = "NONE";
  String faultDetail;
  String openmvError;
};

struct ModeAggregate {
  uint32_t committed = 0;
  uint32_t failures = 0;
  uint32_t tp = 0;
  uint32_t tn = 0;
  uint32_t fp = 0;
  uint32_t fn = 0;
  uint64_t transactionMsTotal = 0;
  uint32_t transactionMsMin = UINT32_MAX;
  uint32_t transactionMsMax = 0;
};

ModeAggregate aggregates[MODE_COUNT];
bool modeExhausted[MODE_COUNT] = {false};
uint8_t currentMode = 0;
uint16_t nextSequence = 1;
bool missionCompleted = false;
String lastFaultDomain = "NONE";
String lastFaultCode = "NONE";
String lastFaultDetail;

void logEvent(const char* level, const char* source, const char* event,
              const String& detail = "");
void setFault(const char* domain, const char* code, const String& detail = "");
void writeSummary();

// ------------------------------- Byte helpers -------------------------------
uint16_t readLe16(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

uint32_t readLe32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

void writeLe16(uint8_t* data, uint16_t value) {
  data[0] = value & 0xFF;
  data[1] = (value >> 8) & 0xFF;
}

void writeLe32(uint8_t* data, uint32_t value) {
  data[0] = value & 0xFF;
  data[1] = (value >> 8) & 0xFF;
  data[2] = (value >> 16) & 0xFF;
  data[3] = (value >> 24) & 0xFF;
}

uint32_t crc32Update(uint32_t previous, const uint8_t* data, size_t length) {
  uint32_t crc = previous ^ 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

uint32_t crc32Buffer(const uint8_t* data, size_t length) {
  return crc32Update(0, data, length);
}

void freeFrame(Frame& frame) {
  if (frame.payload != nullptr) {
    free(frame.payload);
    frame.payload = nullptr;
  }
  frame.length = 0;
}

bool readExact(uint8_t* destination, size_t length, uint32_t timeoutMs) {
  size_t received = 0;
  uint32_t lastProgress = millis();
  while (received < length) {
    int available = PayloadSerial.available();
    if (available > 0) {
      size_t count = min((size_t)available, length - received);
      size_t actual = PayloadSerial.readBytes(destination + received, count);
      if (actual > 0) {
        received += actual;
        linkStats.rxBytes += actual;
        lastProgress = millis();
      }
    } else if (millis() - lastProgress >= timeoutMs) {
      linkStats.timeouts++;
      return false;
    } else {
      delay(1);
    }
  }
  return true;
}

bool findMagic(uint32_t timeoutMs) {
  uint32_t started = millis();
  bool gotO = false;
  while (millis() - started < timeoutMs) {
    if (PayloadSerial.available()) {
      int value = PayloadSerial.read();
      if (value < 0) continue;
      linkStats.rxBytes++;
      if (!gotO) gotO = value == 'O';
      else if (value == 'M') return true;
      else gotO = value == 'O';
    } else {
      delay(1);
    }
  }
  linkStats.timeouts++;
  return false;
}

bool readFrame(Frame& frame, uint32_t timeoutMs = FRAME_TIMEOUT_MS) {
  freeFrame(frame);
  if (!findMagic(timeoutMs)) return false;
  uint8_t tail[HEADER_SIZE - 2];
  if (!readExact(tail, sizeof(tail), timeoutMs)) return false;
  uint8_t header[HEADER_SIZE];
  header[0] = 'O';
  header[1] = 'M';
  memcpy(header + 2, tail, sizeof(tail));

  uint32_t expectedHeaderCrc = readLe32(header + 16);
  if (crc32Buffer(header, HEADER_PREFIX_SIZE) != expectedHeaderCrc) {
    linkStats.headerCrcErrors++;
    setFault("UART_INTEGRITY", "HEADER_CRC", "Header CRC32 mismatch");
    return false;
  }
  if (header[2] != PROTOCOL_VERSION) {
    setFault("UART_INTEGRITY", "PROTOCOL_VERSION", String("received=") + header[2]);
    return false;
  }
  frame.type = header[3];
  frame.sequence = readLe16(header + 4);
  frame.flags = readLe16(header + 6);
  frame.length = readLe32(header + 8);
  uint32_t expectedPayloadCrc = readLe32(header + 12);
  if (frame.length > MAX_FRAME_PAYLOAD) {
    linkStats.lengthErrors++;
    setFault("UART_INTEGRITY", "FRAME_LENGTH", String("bytes=") + frame.length);
    return false;
  }
  if (frame.length) {
    frame.payload = (uint8_t*)malloc(frame.length + 1);
    if (!frame.payload) {
      setFault("ESP32_MEMORY", "FRAME_MALLOC", String("bytes=") + frame.length);
      return false;
    }
    if (!readExact(frame.payload, frame.length, timeoutMs)) {
      freeFrame(frame);
      return false;
    }
    frame.payload[frame.length] = 0;
  }
  if (crc32Buffer(frame.payload, frame.length) != expectedPayloadCrc) {
    linkStats.payloadCrcErrors++;
    setFault("UART_INTEGRITY", "PAYLOAD_CRC", String("seq=") + frame.sequence);
    freeFrame(frame);
    return false;
  }
  linkStats.rxFrames++;
  return true;
}

bool writeAll(const uint8_t* data, size_t length) {
  size_t sent = 0;
  while (sent < length) {
    size_t count = PayloadSerial.write(data + sent, length - sent);
    if (count == 0) {
      delay(1);
      continue;
    }
    sent += count;
  }
  linkStats.txBytes += length;
  return true;
}

bool sendFrame(uint8_t type, uint16_t sequence, const uint8_t* payload,
               uint32_t length, uint16_t flags = 0) {
  uint8_t header[HEADER_SIZE] = {0};
  header[0] = 'O';
  header[1] = 'M';
  header[2] = PROTOCOL_VERSION;
  header[3] = type;
  writeLe16(header + 4, sequence);
  writeLe16(header + 6, flags);
  writeLe32(header + 8, length);
  writeLe32(header + 12, crc32Buffer(payload, length));
  writeLe32(header + 16, crc32Buffer(header, HEADER_PREFIX_SIZE));
  if (!writeAll(header, sizeof(header))) return false;
  if (length && !writeAll(payload, length)) return false;
  linkStats.txFrames++;
  return true;
}

uint16_t allocateSequence() {
  uint16_t value = nextSequence++;
  if (nextSequence == 0) nextSequence = 1;
  return value;
}

void flushPayloadRx(uint32_t quietMs = 20) {
  uint32_t lastRx = millis();
  while (millis() - lastRx < quietMs) {
    while (PayloadSerial.available()) {
      PayloadSerial.read();
      lastRx = millis();
    }
    delay(1);
  }
}

// ------------------------------- JSON helpers -------------------------------
String jsonString(const String& json, const char* key) {
  String pattern = String("\"") + key + "\":";
  int pos = json.indexOf(pattern);
  if (pos < 0) return "";
  pos += pattern.length();
  while (pos < (int)json.length() && isspace(json[pos])) pos++;
  if (pos >= (int)json.length() || json[pos] != '"') return "";
  pos++;
  int end = json.indexOf('"', pos);
  return end < 0 ? "" : json.substring(pos, end);
}

long jsonLong(const String& json, const char* key, long fallback = -1) {
  String pattern = String("\"") + key + "\":";
  int pos = json.indexOf(pattern);
  if (pos < 0) return fallback;
  pos += pattern.length();
  while (pos < (int)json.length() && isspace(json[pos])) pos++;
  int end = pos;
  if (end < (int)json.length() && json[end] == '-') end++;
  while (end < (int)json.length() && isdigit(json[end])) end++;
  return end == pos ? fallback : json.substring(pos, end).toInt();
}

uint32_t jsonUInt32(const String& json, const char* key, uint32_t fallback = 0) {
  String pattern = String("\"") + key + "\":";
  int pos = json.indexOf(pattern);
  if (pos < 0) return fallback;
  pos += pattern.length();
  while (pos < (int)json.length() && isspace(json[pos])) pos++;
  const char* start = json.c_str() + pos;
  char* end = nullptr;
  unsigned long value = strtoul(start, &end, 10);
  return end == start ? fallback : (uint32_t)value;
}

bool jsonBool(const String& json, const char* key, bool fallback = false) {
  String pattern = String("\"") + key + "\":";
  int pos = json.indexOf(pattern);
  if (pos < 0) return fallback;
  pos += pattern.length();
  while (pos < (int)json.length() && isspace(json[pos])) pos++;
  if (json.substring(pos, pos + 4) == "true") return true;
  if (json.substring(pos, pos + 5) == "false") return false;
  return fallback;
}

String sanitizeFilename(String value) {
  for (size_t i = 0; i < value.length(); ++i) {
    char c = value[i];
    if (!(isalnum(c) || c == '.' || c == '_' || c == '-')) value.setCharAt(i, '_');
  }
  return value;
}

String jsonEscape(const String& value) {
  String output;
  for (size_t i = 0; i < value.length(); ++i) {
    char c = value[i];
    if (c == '\\' || c == '"') {
      output += '\\';
      output += c;
    } else if (c == '\n') {
      output += "\\n";
    } else if (c == '\r') {
      output += "\\r";
    } else if ((uint8_t)c >= 0x20) {
      output += c;
    }
  }
  return output;
}

// -------------------------- Storage and telemetry ---------------------------
void initializeOutput() {
  Serial.println("[BOOT] terminal-only output; no SD or flash persistence on ESP32");
}

void logEvent(const char* level, const char* source, const char* event,
              const String& detail) {
  Serial.printf("{\"record_type\":\"event\",\"uptime_ms\":%lu,",
                (unsigned long)millis());
  Serial.printf("\"level\":\"%s\",\"source\":\"%s\",", level, source);
  Serial.print("\"event\":\"");
  Serial.print(jsonEscape(String(event)));
  Serial.print("\",\"detail\":\"");
  Serial.print(jsonEscape(detail));
  Serial.printf("\",\"free_heap_bytes\":%lu,\"rx_frames\":%lu,\"tx_frames\":%lu,",
                (unsigned long)ESP.getFreeHeap(),
                (unsigned long)linkStats.rxFrames,
                (unsigned long)linkStats.txFrames);
  Serial.printf("\"header_crc_errors\":%lu,\"payload_crc_errors\":%lu,",
                (unsigned long)linkStats.headerCrcErrors,
                (unsigned long)linkStats.payloadCrcErrors);
  Serial.printf("\"timeouts\":%lu}\n", (unsigned long)linkStats.timeouts);
}

void setFault(const char* domain, const char* code, const String& detail) {
  lastFaultDomain = domain;
  lastFaultCode = code;
  lastFaultDetail = detail;
  logEvent("ERROR", domain, code, detail);
}

void setSampleFault(SampleMetrics& metrics, const char* domain, const char* code,
                    const String& detail) {
  metrics.faultDomain = domain;
  metrics.faultCode = code;
  metrics.faultDetail = detail;
  setFault(domain, code, detail);
}

void writeSummary() {
  Serial.printf("{\"record_type\":\"summary\",\"updated_uptime_ms\":%lu,",
                (unsigned long)millis());
  Serial.printf("\"mission_completed\":%s,", missionCompleted ? "true" : "false");
  Serial.print("\"last_fault_domain\":\"");
  Serial.print(jsonEscape(lastFaultDomain));
  Serial.print("\",\"last_fault_code\":\"");
  Serial.print(jsonEscape(lastFaultCode));
  Serial.printf("\",\"link\":{\"rx_frames\":%lu,\"tx_frames\":%lu,",
                (unsigned long)linkStats.rxFrames, (unsigned long)linkStats.txFrames);
  Serial.printf("\"rx_bytes\":%lu,\"tx_bytes\":%lu,\"header_crc_errors\":%lu,",
                (unsigned long)linkStats.rxBytes, (unsigned long)linkStats.txBytes,
                (unsigned long)linkStats.headerCrcErrors);
  Serial.printf("\"payload_crc_errors\":%lu,\"length_errors\":%lu,",
                (unsigned long)linkStats.payloadCrcErrors,
                (unsigned long)linkStats.lengthErrors);
  Serial.printf("\"timeouts\":%lu,\"sequence_mismatches\":%lu},\"modes\":[",
                (unsigned long)linkStats.timeouts,
                (unsigned long)linkStats.sequenceMismatches);
  for (uint8_t i = 0; i < MODE_COUNT; ++i) {
    if (i) Serial.print(',');
    const ModeAggregate& a = aggregates[i];
    Serial.printf("{\"name\":\"%s\",\"exhausted\":%s,\"committed\":%lu,",
                  MODE_SEQUENCE[i], modeExhausted[i] ? "true" : "false",
                  (unsigned long)a.committed);
    Serial.printf("\"failures\":%lu,\"tp\":%lu,\"tn\":%lu,\"fp\":%lu,\"fn\":%lu,",
                  (unsigned long)a.failures, (unsigned long)a.tp,
                  (unsigned long)a.tn, (unsigned long)a.fp, (unsigned long)a.fn);
    Serial.printf("\"transaction_ms_total\":%llu,\"transaction_ms_min\":%lu,",
                  (unsigned long long)a.transactionMsTotal,
                  (unsigned long)(a.transactionMsMin == UINT32_MAX ? 0 : a.transactionMsMin));
    Serial.printf("\"transaction_ms_max\":%lu}", (unsigned long)a.transactionMsMax);
  }
  Serial.println("]}");
}

bool appendTelemetry(uint8_t modeIndex, const SampleMetrics& metrics, bool committed) {
  Serial.print("{\"record_type\":\"telemetry\",\"obc\":{");
  Serial.printf("\"mode\":\"%s\",", MODE_SEQUENCE[modeIndex]);
  Serial.printf("\"committed\":%s,", committed ? "true" : "false");
  Serial.printf("\"wake_to_ready_ms\":%lu,", (unsigned long)metrics.wakeToReadyMs);
  Serial.printf("\"ping_rtt_ms\":%lu,", (unsigned long)metrics.pingRttMs);
  Serial.printf("\"transaction_ms\":%lu,", (unsigned long)metrics.transactionMs);
  Serial.printf("\"file_rx_ms\":%lu,", (unsigned long)metrics.fileRxMs);
  Serial.printf("\"file_bytes\":%lu,", (unsigned long)metrics.fileBytes);
  Serial.printf("\"file_crc32\":%lu,", (unsigned long)metrics.fileCrcCalculated);
  Serial.printf("\"file_valid\":%s,", metrics.fileValid ? "true" : "false");
  Serial.print("\"storage\":\"discarded_after_crc\",");
  Serial.print("\"fault_domain\":\"");
  Serial.print(jsonEscape(metrics.faultDomain));
  Serial.print("\",\"fault_code\":\"");
  Serial.print(jsonEscape(metrics.faultCode));
  Serial.print("\",\"fault_detail\":\"");
  Serial.print(jsonEscape(metrics.faultDetail));
  Serial.print("\",\"openmv_error\":\"");
  Serial.print(jsonEscape(metrics.openmvError));
  Serial.print("\",");
  Serial.printf("\"rx_frames\":%lu,", (unsigned long)linkStats.rxFrames);
  Serial.printf("\"tx_frames\":%lu,", (unsigned long)linkStats.txFrames);
  Serial.printf("\"header_crc_errors\":%lu,", (unsigned long)linkStats.headerCrcErrors);
  Serial.printf("\"payload_crc_errors\":%lu", (unsigned long)linkStats.payloadCrcErrors);
  Serial.print("},\"openmv\":");
  Serial.print(metrics.resultJson.length() ? metrics.resultJson : "null");
  Serial.println("}");
  return true;
}

// ----------------------------- Link transactions ----------------------------
void wakePulse() {
  digitalWrite(WAKE_PIN, HIGH);
  delay(80);
  digitalWrite(WAKE_PIN, LOW);
}

bool waitReady(uint32_t timeoutMs, uint32_t& elapsedMs) {
  uint32_t started = millis();
  while (millis() - started < timeoutMs) {
    Frame frame;
    if (!readFrame(frame, 1000)) continue;
    bool ready = frame.type == MSG_READY;
    if (frame.type == MSG_DEBUG && frame.payload) {
      logEvent("DEBUG", "OPENMV", "DEBUG_FRAME", String((char*)frame.payload));
    }
    freeFrame(frame);
    if (ready) {
      elapsedMs = millis() - started;
      return true;
    }
  }
  return false;
}

bool wakeAndSynchronize(uint32_t& wakeToReadyMs) {
  flushPayloadRx(20);
  uint32_t totalStarted = millis();
  for (uint8_t attempt = 0; attempt < 3; ++attempt) {
    wakePulse();
    uint32_t ignored = 0;
    if (waitReady(READY_TIMEOUT_MS, ignored)) {
      wakeToReadyMs = millis() - totalStarted;
      return true;
    }
    Serial.printf("[WARN] READY retry %u/3\n", attempt + 1);
  }
  setFault("UART_TIMEOUT", "READY_TIMEOUT",
           "No valid READY; inspect OpenMV power/wake and UART wiring");
  return false;
}

bool waitAck(uint16_t sequence, uint32_t timeoutMs) {
  uint32_t started = millis();
  while (millis() - started < timeoutMs) {
    Frame frame;
    if (!readFrame(frame, 1000)) continue;
    if (frame.sequence != sequence && frame.type != MSG_READY && frame.type != MSG_DEBUG) {
      linkStats.sequenceMismatches++;
      freeFrame(frame);
      continue;
    }
    bool ack = frame.type == MSG_ACK;
    bool error = frame.type == MSG_ERROR;
    if (error && frame.payload) {
      String remote((char*)frame.payload);
      Serial.printf("[OPENMV ERROR] %s\n", remote.c_str());
      setFault("OPENMV_REMOTE", "ACK_ERROR", remote);
    }
    freeFrame(frame);
    if (ack) {
      logEvent("DEBUG", "OPENMV", "ACK", String("seq=") + sequence);
      return true;
    }
    if (error) return false;
  }
  setFault("UART_TIMEOUT", "ACK_TIMEOUT", String("seq=") + sequence);
  return false;
}

bool startCommand(const String& command, uint16_t& sequence) {
  sequence = allocateSequence();
  Serial.printf("[TX seq=%u] %s\n", sequence, command.c_str());
  logEvent("DEBUG", "ESP32", "COMMAND_TX",
           String("seq=") + sequence + ",cmd=" + command);
  if (!sendFrame(MSG_COMMAND, sequence, (const uint8_t*)command.c_str(), command.length())) {
    setFault("ESP32_UART", "COMMAND_TX", command);
    return false;
  }
  return waitAck(sequence, ACK_TIMEOUT_MS);
}

bool pingPayload(uint32_t& rttMs) {
  uint16_t sequence;
  uint32_t started = millis();
  if (!startCommand("PING", sequence)) return false;
  rttMs = millis() - started;
  return true;
}

bool receiveTransaction(uint16_t sequence, SampleMetrics& metrics, bool acceptFile) {
  bool done = false;
  uint32_t lastProgress = millis();
  uint32_t fileStarted = 0;
  uint16_t expectedChunk = 0;

  while (!done) {
    Frame frame;
    if (!readFrame(frame, FRAME_TIMEOUT_MS)) {
      if (millis() - lastProgress >= TRANSACTION_NO_PROGRESS_MS) break;
      continue;
    }
    lastProgress = millis();
    if (frame.sequence != sequence && frame.type != MSG_READY && frame.type != MSG_DEBUG) {
      linkStats.sequenceMismatches++;
      freeFrame(frame);
      continue;
    }

    if (frame.type == MSG_FILE_BEGIN) {
      if (!acceptFile || !frame.payload) {
        metrics.fileValid = false;
        setSampleFault(metrics, "ESP32_STATE", "UNEXPECTED_FILE_BEGIN",
                       String("seq=") + sequence);
      } else {
        metrics.fileSeen = true;
        metrics.fileValid = true;
        metrics.fileBytes = 0;
        metrics.fileCrcCalculated = 0;
        expectedChunk = 0;
        fileStarted = millis();
        String metadata((char*)frame.payload);
      }
    } else if (frame.type == MSG_FILE_CHUNK) {
      if (!metrics.fileSeen || frame.flags != expectedChunk) {
        metrics.fileValid = false;
        setSampleFault(metrics, "UART_INTEGRITY", "CHUNK_ORDER",
                       String("received=") + frame.flags + ",expected=" + expectedChunk);
      } else {
        metrics.fileCrcCalculated = crc32Update(
            metrics.fileCrcCalculated, frame.payload, frame.length);
        metrics.fileBytes += frame.length;
        expectedChunk++;
      }
    } else if (frame.type == MSG_FILE_END) {
      metrics.fileRxMs = millis() - fileStarted;
      String footer = frame.payload ? String((char*)frame.payload) : String();
      metrics.fileCrcExpected = jsonUInt32(footer, "crc32", UINT32_MAX);
      uint32_t expectedBytes = jsonUInt32(footer, "bytes", UINT32_MAX);
      metrics.fileValid = metrics.fileValid &&
                          metrics.fileCrcCalculated == metrics.fileCrcExpected &&
                          metrics.fileBytes == expectedBytes;
      if (!metrics.fileValid && metrics.faultCode == "NONE") {
        setSampleFault(metrics, "UART_INTEGRITY", "FILE_FINAL_CHECK",
                       String("bytes=") + metrics.fileBytes + "/" + expectedBytes);
      }
    } else if (frame.type == MSG_RESULT) {
      metrics.resultSeen = true;
      metrics.resultJson = frame.payload ? String((char*)frame.payload) : String("{}");
      metrics.sampleId = jsonString(metrics.resultJson, "sample_id");
      metrics.noMoreImages = jsonBool(metrics.resultJson, "no_more_images", false);
    } else if (frame.type == MSG_ERROR) {
      metrics.openmvError = frame.payload ? String((char*)frame.payload) : String("UNKNOWN");
      setSampleFault(metrics, "OPENMV_REMOTE", "REMOTE_ERROR", metrics.openmvError);
      freeFrame(frame);
      return false;
    } else if (frame.type == MSG_DONE) {
      done = true;
    } else if (frame.type == MSG_DEBUG && frame.payload) {
      logEvent("DEBUG", "OPENMV", "DEBUG_FRAME", String((char*)frame.payload));
    }
    freeFrame(frame);
  }
  bool ok = done && metrics.resultSeen && metrics.fileValid;
  if (!ok && metrics.faultCode == "NONE") {
    if (!done || !metrics.resultSeen) {
      setSampleFault(metrics, "UART_TIMEOUT", "TRANSACTION_INCOMPLETE",
                     String("seq=") + sequence + ",done=" + done +
                     ",result=" + metrics.resultSeen);
    }
  }
  return ok;
}

bool commandTransaction(const String& command, SampleMetrics& metrics, bool acceptFile = false) {
  uint16_t sequence;
  uint32_t started = millis();
  if (!startCommand(command, sequence)) {
    metrics.faultDomain = lastFaultDomain;
    metrics.faultCode = lastFaultCode;
    metrics.faultDetail = lastFaultDetail;
    return false;
  }
  bool ok = receiveTransaction(sequence, metrics, acceptFile);
  metrics.transactionMs = millis() - started;
  return ok;
}

bool simpleTransaction(const String& command, String* result = nullptr) {
  SampleMetrics metrics;
  bool ok = commandTransaction(command, metrics, false);
  if (result) *result = metrics.resultJson;
  return ok;
}

void requestSleep() {
  uint16_t sequence;
  startCommand("SLEEP", sequence);
  delay(20);
  flushPayloadRx(30);
}

// ------------------------------ Mission logic -------------------------------
void updateAggregate(uint8_t modeIndex, const SampleMetrics& metrics) {
  ModeAggregate& aggregate = aggregates[modeIndex];
  aggregate.committed++;
  aggregate.transactionMsTotal += metrics.transactionMs;
  aggregate.transactionMsMin = min(aggregate.transactionMsMin, metrics.transactionMs);
  aggregate.transactionMsMax = max(aggregate.transactionMsMax, metrics.transactionMs);
  int expected = (int)jsonLong(metrics.resultJson, "expected_id", -1);
  int predicted = (int)jsonLong(metrics.resultJson, "predicted_id", -1);
  if (expected == 1 && predicted == 1) aggregate.tp++;
  else if (expected == 0 && predicted == 0) aggregate.tn++;
  else if (expected == 0 && predicted == 1) aggregate.fp++;
  else if (expected == 1 && predicted == 0) aggregate.fn++;
}

void printAggregate(uint8_t modeIndex) {
  const ModeAggregate& a = aggregates[modeIndex];
  uint32_t labelled = a.tp + a.tn + a.fp + a.fn;
  float accuracy = labelled ? (float)(a.tp + a.tn) / labelled : NAN;
  float precision = (a.tp + a.fp) ? (float)a.tp / (a.tp + a.fp) : NAN;
  float recall = (a.tp + a.fn) ? (float)a.tp / (a.tp + a.fn) : NAN;
  float specificity = (a.tn + a.fp) ? (float)a.tn / (a.tn + a.fp) : NAN;
  float f1 = (!isnan(precision) && !isnan(recall) && precision + recall > 0)
                 ? 2.0f * precision * recall / (precision + recall) : NAN;
  float meanMs = a.committed ? (float)a.transactionMsTotal / a.committed : NAN;
  Serial.printf("[SUMMARY %s] n=%lu failures=%lu TP=%lu TN=%lu FP=%lu FN=%lu ",
                MODE_SEQUENCE[modeIndex], (unsigned long)a.committed,
                (unsigned long)a.failures, (unsigned long)a.tp,
                (unsigned long)a.tn, (unsigned long)a.fp, (unsigned long)a.fn);
  Serial.printf("acc=%.4f precision=%.4f recall=%.4f specificity=%.4f f1=%.4f mean_ms=%.1f\n",
                accuracy, precision, recall, specificity, f1, meanMs);
}

bool runModePass(uint8_t modeIndex) {
  logEvent("INFO", "ESP32", "MODE_PASS_START", MODE_SEQUENCE[modeIndex]);
  uint32_t wakeMs = 0;
  if (!wakeAndSynchronize(wakeMs)) {
    aggregates[modeIndex].failures++;
    writeSummary();
    return false;
  }
  uint32_t pingMs = 0;
  if (!pingPayload(pingMs)) {
    aggregates[modeIndex].failures++;
    writeSummary();
    requestSleep();
    return false;
  }
  String setResult;
  if (!simpleTransaction(String("SET_MODE:") + MODE_SEQUENCE[modeIndex], &setResult)) {
    aggregates[modeIndex].failures++;
    writeSummary();
    requestSleep();
    return false;
  }

  for (uint8_t count = 0; count < SAMPLES_PER_MODE_PASS; ++count) {
    SampleMetrics metrics;
    metrics.wakeToReadyMs = wakeMs;
    metrics.pingRttMs = pingMs;
    if (!commandTransaction("INFER", metrics, true)) {
      aggregates[modeIndex].failures++;
      appendTelemetry(modeIndex, metrics, false);
      writeSummary();
      requestSleep();
      return false;
    }
    if (metrics.noMoreImages) {
      modeExhausted[modeIndex] = true;
      logEvent("INFO", "OPENMV", "MODE_EXHAUSTED", MODE_SEQUENCE[modeIndex]);
      writeSummary();
      break;
    }
    if (metrics.sampleId.length() == 0) {
      setSampleFault(metrics, "ESP32_STATE", "MISSING_SAMPLE_ID", MODE_SEQUENCE[modeIndex]);
      aggregates[modeIndex].failures++;
      appendTelemetry(modeIndex, metrics, false);
      writeSummary();
      requestSleep();
      return false;
    }
    String commitResult;
    bool commitTransport = simpleTransaction(String("COMMIT:") + metrics.sampleId, &commitResult);
    bool committed = commitTransport && jsonBool(commitResult, "committed", false);
    if (!committed) {
      if (commitTransport) {
        setSampleFault(metrics, "ESP32_STATE", "COMMIT_REJECTED", metrics.sampleId);
      } else {
        metrics.faultDomain = lastFaultDomain;
        metrics.faultCode = lastFaultCode;
        metrics.faultDetail = lastFaultDetail;
      }
    }
    appendTelemetry(modeIndex, metrics, committed);
    if (!committed) {
      aggregates[modeIndex].failures++;
      writeSummary();
      requestSleep();
      return false;
    }
    updateAggregate(modeIndex, metrics);
    logEvent("INFO", "ESP32", "SAMPLE_COMMITTED", metrics.sampleId);
    writeSummary();
    Serial.printf("[SAMPLE] %s file=%luB crc=%s storage=DISCARDED transaction=%lums\n",
                  metrics.sampleId.c_str(), (unsigned long)metrics.fileBytes,
                  metrics.fileValid ? "OK" : "FAIL",
                  (unsigned long)metrics.transactionMs);
  }
  requestSleep();
  printAggregate(modeIndex);
  logEvent("INFO", "ESP32", "MODE_PASS_END", MODE_SEQUENCE[modeIndex]);
  return true;
}

int nextAvailableMode() {
  for (uint8_t offset = 0; offset < MODE_COUNT; ++offset) {
    uint8_t index = (currentMode + offset) % MODE_COUNT;
    if (!modeExhausted[index]) return index;
  }
  return -1;
}

bool allModesExhausted() {
  for (uint8_t index = 0; index < MODE_COUNT; ++index) {
    if (!modeExhausted[index]) return false;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(WAKE_PIN, OUTPUT);
  digitalWrite(WAKE_PIN, LOW);
  PayloadSerial.setRxBufferSize(UART_RX_BUFFER_BYTES);
  PayloadSerial.begin(UART_BAUD, SERIAL_8N1, PAYLOAD_RX_PIN, PAYLOAD_TX_PIN);
  initializeOutput();
  Serial.println("[BOOT] ESP32 OBC v3.2 ready");
  logEvent("INFO", "ESP32", "BOOT_OBC_READY", "firmware=3.2-terminal-only");
  writeSummary();
}

void loop() {
  if (missionCompleted) {
    delay(10000);
    return;
  }
  int selected = nextAvailableMode();
  if (selected < 0) {
    missionCompleted = true;
    return;
  }
  currentMode = (uint8_t)selected;
  bool success = false;
  for (uint8_t attempt = 0; attempt < MAX_PASS_RETRIES && !success; ++attempt) {
    success = runModePass(currentMode);
    if (!success) delay(500u << attempt);
  }
  if (!success) {
    logEvent("ERROR", "ESP32", "MODE_PASS_RETRIES_EXHAUSTED",
             MODE_SEQUENCE[currentMode]);
    writeSummary();
  }
  if (success) currentMode = (currentMode + 1) % MODE_COUNT;
  if (allModesExhausted()) {
    missionCompleted = true;
    logEvent("INFO", "ESP32", "MISSION_COMPLETED", "all modes exhausted");
    for (uint8_t i = 0; i < MODE_COUNT; ++i) printAggregate(i);
    writeSummary();
  }
  delay(MISSION_PERIOD_MS);
}
