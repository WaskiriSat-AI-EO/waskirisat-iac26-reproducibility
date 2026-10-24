# HIL environment

Experimental hardware, software and operating configuration for **Test Surrogate** and **Test Payload**. The configuration below corresponds to the embedded implementations and measurements documented in `hil/`.

## Hardware and runtime

| Component | Configuration |
|---|---|
| Payload board | OpenMV Cam H7 Plus, STM32H743, board identifier `OPENMV4P` |
| OpenMV firmware reference | 5.0.0; identified in the experimental reports |
| Embedded MicroPython | 1.28.0-49 |
| OBC board | ESP32-32X |
| OpenMV deployment | OpenMV IDE 5.0.0; manual execution of the selected test entry point |
| OBC development environment | Arduino IDE 2.3.10 |
| Arduino board selection | ESP32 Development Kit |
| CPU, flash and partition settings | Board-target defaults, without manual overrides |
| ESP32 board package reference | Espressif Arduino-ESP32 3.3.11 |
| Host Python | 3.14.7 |
| Payload storage | 8 GB SD card containing model files, labeled images and progress journals |
| Board power and host connections | Separate USB connections for OpenMV and ESP32; common signal ground |
| Desktop inference reference | LiteRT 2.2.0, one CPU thread |

The IDE, board selection, default build settings, SD capacity and host Python version are operator-confirmed. Firmware and board-package references describe the documented reproduction environment; their original installations have not been independently inspected.

## Applications and artifacts

| Item | Version / location |
|---|---|
| Test Surrogate entry point | [`model_execution.py`](openmv/test1_surrogate/model_execution.py), software/schema 6.3 |
| Desktop evaluator | [`validate_tflite_LOCALTEST.py`](openmv/test1_surrogate/validate_tflite_LOCALTEST.py) |
| Test Payload controller | [`main_code.py`](openmv/test2_payload/main_code.py), payload application v3.1 |
| Payload configuration | [`config.py`](openmv/test2_payload/config.py) |
| Inference and checkpoint management | [`model_runner.py`](openmv/test2_payload/model_runner.py) |
| UART implementation | [`payload_protocol.py`](openmv/test2_payload/payload_protocol.py), wire protocol v2 |
| OBC controller | [`obc_v3.ino`](obc_esp32/obc_v3.ino), application v3.2 |
| Test Payload result schema | 3.1 |
| Evaluation catalogue | [`manifest.csv`](catalogue/manifest.csv), 100 Cloud and 100 Fire samples |

Application, result-schema and wire-protocol versions identify different interfaces and are independent.

## Electrical interfaces

| Parameter | Configuration |
|---|---|
| Payload UART | 460800 baud, 8 data bits, no parity, one stop bit |
| Logic level | 3.3 V |
| Hardware flow control | Disabled |
| OpenMV UART | UART3; TX=P4, RX=P5 |
| ESP32 UART | UART2; RX=GPIO16, TX=GPIO17 |
| Wake connection | ESP32 GPIO4 → OpenMV P0, rising edge, input pull-down |
| Ground | Shared between both boards |
| ESP32 host log | USB Serial at 115200 baud |
| UART RX buffers | OpenMV 4096 bytes; ESP32 32768 bytes |

## Operating configuration

| Setting | Value / behavior |
|---|---|
| Power mode | `LOW_POWER_ENABLED=False`; wake-controlled idle wait |
| Source transfer | `TRANSFER_SOURCE_FILE=True` |
| Chunk size | 512 bytes; final chunk may be shorter |
| File handling on OBC | Incremental CRC32 validation and discard after reception |
| Persistent progress | `PERSIST_PROGRESS=True`; alternating OpenMV SD journal slots |
| Journal base path | `paper_tests/test_2/mission_state_v3` |
| Scheduling | Up to five samples per mode pass; Cloud and Fire mode rotation |
| Inter-pass loop delay | 3000 ms |
| Decision threshold | Positive class when probability ≥ 0.5 |
| Model residency | One model at a time; reuse within a mode and unload on mode change |
| Host evidence | ESP32 event, telemetry and summary records captured as a complete terminal stream |

The applications resolve the data root from `/sdcard/paper_tests`, `/sd/paper_tests`, `/paper_tests`, then `paper_tests`. Test Surrogate writes results under `test_1/`; Test Payload persists positions under `test_2/`.

Full-model SHA-256 verification is performed on the host rather than immediately before model allocation on OpenMV. This avoids the heap-fragmentation behavior documented in the Test Surrogate report.

See the [protocol reference](protocol/README.md) for wiring, deployment, model contracts, timeouts and state diagrams, and the [Test Surrogate](results/test1/documentation.md) and [Test Payload](results/test2/documentation.md) reports for measurements and their experimental scope.
