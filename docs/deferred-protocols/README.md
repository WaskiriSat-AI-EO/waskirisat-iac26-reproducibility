# Deferred campaign (paper Table 9)

Everything the paper cannot claim is listed here as a specified test with a pass criterion and the hardware it waits on. These items are **not results**; none of them appears in Tables 5–8.

Two kinds of item are mixed here, and the difference matters:

- **Hardware-blocked**: waits on the STM32N6 board, the flight sensors or an environmental facility.
- **Scope cut**: exercisable on the surrogate (OpenMV H7 Plus) but not run within the 2026-09-13 submission window.

## V4: toolchain-in-the-loop (scope cut)

| | |
|---|---|
| Claim | Both INT8 artifacts compile for the N6 and map to the Neural-ART NPU as expected |
| Test | `stedgeai analyze` on the actual target: MACC count, weight and activation memory, NPU-mappable layer coverage |
| Pass | The exported graph maps to the accelerator; no unexpected CPU fallback. For the thermal artifact, check whether its float32 input boundary keeps the input-side affine off the NPU |
| Needs | The compiled artifact and the toolchain only (no board) |

## V6: target hardware-in-the-loop (hardware-blocked)

| Claim | Test | Pass criterion |
|---|---|---|
| N6 inference latency | Benchmark the `stedgeai`-compiled artifact through `ml.Model()` | Meets the imaging duty cycle with margin; confirms the surrogate bounds |
| NPU layer coverage in execution | Compare measured throughput with the V4 static analysis | No unexpected CPU fallback layers |
| Payload power draw | Instrumented measurement, standby and peak | Within the allocated envelope |
| Bus interface conformance | Exercise the payload connector against bus-emulation hardware | All specified transactions pass |
| Sensor-in-the-loop imaging | Lepton 3.5 and PAG7936LT capture through the flight path | Frames within the modelled radiometric range |
| IQC on real frames | Run the cloud classifier on at least 5 real PAG7936LT frames | Accuracy at least 70% against independent ground truth (mission criterion P2.2) |
| **Simulation fidelity check** | Compare real sensor frames with the synthetic pipeline output | Resolves at least 5 K between surface features (mission criterion P2.1). This retroactively validates the dataset methodology and is the most important item of the campaign |

## V7: environmental qualification (hardware-blocked)

TVAC and vibration follow standard practice for the carrier board and start only after V6. Radiation: a beam campaign at a stated facility class would measure the upset **rate**, which the emulated injection below (a **tolerance** curve) cannot. The two are complementary, not redundant.

## Emulated SEU injection (scope cut)

| | |
|---|---|
| Claim | INT8 recall degrades gracefully with weight-memory bit corruption, and a bit-error rate exists below which the recall floor (0.99) holds |
| Procedure | Baseline recall, precision and F-beta; flip random bits in the quantized weight tensor, sweeping BER 1e-7 to 1e-3 logarithmically; 20–30 seeds per point, reporting the distribution and not only the mean; optionally stratify by bit position (sign/MSB vs LSB) and by layer; repeat for both payloads |
| Where | Sweep host-side (thousands of inferences); run a small confirmatory subset on the H7 Plus by corrupting the weight buffer in device memory |
| Metrics | Reuse `nina_*/evaluation/metrics.py` so numbers match every other reported one |
| Projection | Crossing the tolerance BER with a published SAA flux for 525 km SSO gives a scrub-interval requirement. This is a **projection**, labelled so every time: the flux-to-BER mapping assumes a device cross-section that has not been measured (T3, arguably T4), so give a range, not a point |
| Link to mission | The in-flight completion is mission objective P3.3 (NPU SEU cross-section, latch-up and TID, correlated with SAA passage) |

## Other surrogate items (scope cut)

| Item | Test | Pass criterion |
|---|---|---|
| Reduced-power standby | Enter and exit stop mode under external wake, logging reset cause and supply current on external instrumentation | Wake latency and standby current recorded and within the envelope |
| Autonomous payload restart | Install the controller as an auto-boot script and force restarts during transactions | No operator intervention; no sample lost or repeated |
| Root cause of the two payload-side aborts | Instrument repeated thermal-transaction cycles (both Test 2 aborts happened there) | Cause fixed or bounded as benign |
| Energy per inference | Current-sensed supply, sustained inference loop and idle | Measured on the integrated payload. It does not transfer from the surrogate |
| Cloud preprocessing alignment (C-19) | Reproduce the flight image path (RGB565 plus resize) in the training and calibration pipeline, or keep full colour depth through resize | On-device probabilities within the acceptance-gate tolerance of the host reference on a held-out set |
| Fault-detection paths | Inject framing and CRC32 faults into the v2 protocol | Every injected fault detected and recovered. They were never triggered on hardware |

## Closure criterion for TRL 5

The payload moves from TRL 3→4 (this paper) to TRL 5 when (1) the V6 simulation-fidelity check confirms the synthetic training distribution against real sensor data, and (2) the surrogate latency and memory bounds are confirmed on the actual N6. The full V7 campaign targets a later maturity level. For the cloud classifier, C-19 alignment is a prerequisite of (1).
