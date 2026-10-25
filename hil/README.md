# hil/ — Surrogate hardware-in-the-loop (rung V5)

Embedded implementations, experimental procedures and measurements supporting the paper are documented here. Test Surrogate evaluates standalone inference on OpenMV; Test Payload evaluates its integration with the ESP32 OBC.

| Test | Purpose | Code | Results |
|---|---|---|---|
| Test Surrogate | Both model artifacts on OpenMV H7 Plus via `ml.Model()`; 200 images, latency/arena/heap and desktop fidelity | [`openmv/test1_surrogate/`](openmv/test1_surrogate/) | [`results/test1/`](results/test1/) |
| Test Payload | ESP32 stand-in OBC, OpenMV payload v3.1, CRC32-framed UART, source streaming and persistent commit progress | [`openmv/test2_payload/`](openmv/test2_payload/), [`obc_esp32/`](obc_esp32/) | [`results/test2/`](results/test2/) |

The experimental firmware is preserved. Behavior changes require a new acquisition with separately identified results; do not overwrite historical evidence to match modified firmware.

## Documentation map

| Document | Contents |
|---|---|
| [`protocol/README.md`](protocol/README.md) | Hardware, connections, per-board files, Arduino/OpenMV setup, modes, wire fields, commands, CRC, checkpoints, errors and telemetry |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Hardware, runtime versions, electrical interfaces and operating configuration |
| [`catalogue/README.md`](catalogue/README.md) | 200-image catalog format, ordering and content hashes |
| [Test Surrogate report](results/test1/documentation.md) | Embedded model validation, preprocessing, timing/memory and desktop comparisons |
| [Test Payload report](results/test2/documentation.md) | Integration experiment, transaction behavior and measured results |

## Running Test Surrogate

This test runs the Cloud and Fire classifiers directly on OpenMV, without OBC commands. It produces a per-image baseline for classification, desktop agreement, latency and memory measurements.

1. Prepare the 8 GB OpenMV SD card with `paper_tests/models/`, `paper_tests/cloud_detection/{clear,cloud}/` and `paper_tests/fire_detection/{no_fire,fire}/`. Each class contains 50 images. Follow the [storage layout](protocol/README.md#required-files-and-storage) and verify file identities using the [sample catalogue](catalogue/README.md) and model hashes in the protocol reference.
2. Connect the OpenMV Cam H7 Plus to OpenMV IDE 5.0.0. Use the runtime configuration in [`ENVIRONMENT.md`](ENVIRONMENT.md). Stop any running payload controller before starting the standalone test.
3. Open [`model_execution.py`](openmv/test1_surrogate/model_execution.py) in the IDE and run it. The evaluator resolves the data root, validates catalog ordering and tensor contracts, then executes Cloud followed by Fire. This ordering gives the larger Cloud tensor arena the initial memory state; only one model is resident at a time.
4. Allow both 100-image batches to finish. Check the IDE console for completion or image errors, then retain `paper_tests/test_1/cloud_detection_results.json` and `paper_tests/test_1/fire_detection_results.json`. Copy results only after execution has finished so that the JSON documents are complete.
5. Review record counts, image errors, timing, memory and desktop comparison fields against the [Test Surrogate report](results/test1/documentation.md). Preserve the original result files before starting another run, which writes to the same output paths.

Full-model hashing immediately before `ml.Model()` caused heap fragmentation and tensor-allocation failure during development. Verify hashes on the host, or perform a separate preflight followed by restart, rather than inserting hashing into the inference sequence.

Optional desktop evaluation uses [`validate_tflite_LOCALTEST.py`](openmv/test1_surrogate/validate_tflite_LOCALTEST.py) with the same `paper_tests/` root. Its host dependencies and comparison command are described in the [protocol reference](protocol/README.md). Desktop outputs and measured OpenMV outputs should be retained separately.

## Running Test Payload

This test uses the ESP32-32X to select the operating mode, request inference, validate source-image transfers and commit sample progress on OpenMV.

1. Prepare the same SD models and image folders used by Test Surrogate. Copy `config.py`, `payload_protocol.py`, `model_runner.py` and `main_code.py` from [`openmv/test2_payload/`](openmv/test2_payload/) together into the OpenMV import root.
2. With board power removed, connect ESP32 RX16 to OpenMV P4/TX, ESP32 TX17 to P5/RX, GPIO4 to P0/wake, and GND to GND. Power through the boards' USB connections. Signals use 3.3 V logic; see the [connection table](protocol/README.md#hardware-and-connections).
3. Place [`obc_v3.ino`](obc_esp32/obc_v3.ino) inside a matching `obc_v3/` sketch directory. Compile and upload using Arduino IDE 2.3.10, ESP32 Development Kit, and default CPU/flash/partition settings. Keep the OBC from beginning acquisition until the payload and host capture are ready.
4. Keep `LOW_POWER_ENABLED=False`, `TRANSFER_SOURCE_FILE=True` and `PERSIST_PROGRESS=True`. For continuation, retain the progress journals and unchanged dataset. For a new independent run, preserve previous evidence and reset both mode positions using the [restart procedure](protocol/README.md#acquisition-progress-and-restart).
5. Run `main_code.py` manually from OpenMV IDE. Open host file capture on the ESP32 USB port at 115200 baud, then reset/start ESP32. The payload link operates separately at 460800 baud. The OBC autonomously wakes the payload, selects modes and processes up to five samples per mode pass.
6. Monitor the event and telemetry stream. Each accepted sample has a valid inference result, a CRC-validated source transfer and a successful commit. Retain capture until `MISSION_COMPLETED` and the final `mission_completed=true` summary appear.
7. Save the complete terminal log and final OpenMV progress journals. Review per-mode counts, fault/link counters and prediction agreement using the [acceptance criteria](protocol/README.md#acceptance-criteria). ESP32 aggregates are volatile, and source images are discarded after CRC validation.

The results obtained contain 89 Cloud and 96 Fire commits from starting checkpoints 11 and 4, respectively. Both 100-image catalogs reached exhaustion, and all 185 recorded predictions matched Test Surrogate. Counts, timing and link statistics describe this observation interval.

## Checking the recorded results

The repository provides offline checks connecting the recorded measurements to the paper tables. Run these commands from the repository root in a Python environment with the project dependencies:

```bash
python -m reproduce.table7
python -m reproduce.table8
python -m reproduce.test2_summary
```

| Command | What it checks |
|---|---|
| `reproduce.table7` | Model size, reported tensor arena, free heap, image counts and estimated inference latency from Test Surrogate summaries |
| `reproduce.table8` | Native-device decision agreement, probability correlation and bias against the desktop references embedded in the image records |
| `reproduce.test2_summary` | Final mission/link counters and agreement between Test Payload predictions and Test Surrogate, joined by dataset and filename |

The Make equivalent is `make table7 table8 test2-summary`. These commands analyze the stored JSON and terminal results; generating new board measurements requires the execution procedures above. See [`reproduce/paper_map.md`](../reproduce/paper_map.md) for the association between scripts and paper items.

## Experimental scope and interpretation

Test Surrogate characterizes the two deployed inference pipelines on OpenMV using a fixed, balanced 200-image catalogue. Test Payload evaluates nominal OBC coordination, mode switching, model reuse, source-file streaming, recorded CRC agreement and persistent catalog advancement on the two-board laboratory setup.

Agreement between the tests establishes consistency of the recorded embedded predictions across standalone and OBC-controlled execution. It does not establish classifier accuracy beyond the evaluated data or desktop equivalence of the preprocessing paths. In particular, native Cloud preprocessing uses RGB565 and OpenMV resizing, which differ from the desktop path.

The operating configuration uses a wake-controlled idle wait with deep-stop disabled. Live camera acquisition, flight OBC compatibility, power/EMI characterization, radiation tolerance and systematic fault-injection coverage are outside these experiments. CRC and error handling describe implemented mechanisms; their presence and zero recorded nominal errors do not constitute an injected-fault detection study.
