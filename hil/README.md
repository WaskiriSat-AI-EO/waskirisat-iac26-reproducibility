# hil/ — Surrogate hardware-in-the-loop (rung V5)

**Owner:** Misael. Everything the paper reports as measured on silicon comes from here.

| Test | What it is | Code | Raw results |
|---|---|---|---|
| Test 1 | Both INT8 artifacts on an OpenMV H7 Plus via `ml.Model()`; 200 inferences, latency/arena/free heap, fidelity vs desktop LiteRT | `openmv/test1_surrogate/` | `results/test1/` |
| Test 2 | ESP32 as stand-in OBC driving the OpenMV payload (v3.1) over a framed CRC32 UART protocol, model load/unload cycled | `openmv/test2_payload/`, `obc_esp32/` | `results/test2/` |

The code is the delivered code, copied verbatim on 2026-09-05. Do not "tidy" it: if it
changes, results must be re-run and `results/` replaced.

## Delivery checklist (Misael)
- [ ] OpenMV firmware version, MicroPython version, board revision (H7 Plus) — fill `ENVIRONMENT.md`
- [ ] ESP32 board, Arduino core / PlatformIO version, wiring (pins, 460800 baud) — `ENVIRONMENT.md`
- [ ] Step-by-step: how to run Test 1 and Test 2 from a clean board — below
- [ ] `protocol/` — protocol v2 spec (framing, CRC32 per packet and file, journal, commit)
- [ ] `catalogue/` — manifest of the 200 samples (IDs + sha256). Images are NOT committed
- [ ] Confirm `results/` regenerates Table 7 (latency, arena) and Table 8 (94 / 69 / 97 % fidelity)
- [ ] Note the known footgun: on-device SHA-256 right before `ml.Model()` fragments the heap
- [ ] State what was and was not exercised (no fault was injected; detection paths are V0 only)

## Models
Artifacts under test are listed in `../models/MANIFEST.json` (sha256). The two `.tflite`
files used in Test 2 are not in git. Fetch them as described in `../models/README.md`.

## Running
_TODO (Misael): Test 1 procedure._

_TODO (Misael): Test 2 procedure._
