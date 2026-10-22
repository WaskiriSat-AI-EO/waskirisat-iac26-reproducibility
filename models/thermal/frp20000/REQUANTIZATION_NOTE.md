# Note: the 0.181 INT8 threshold

The published INT8 artifact is the one validated in HIL: sha256 `6e486fc7…01e8`, exported
2026-08-30, val threshold **0.2875** (`quantization_report.json`).
On 2026-09-03 the same FP32 checkpoint (`fe7ff10f…`) was re-quantized. That run produced
a different file (sha256 `fdbdf938…5588`) with val threshold **0.181**. This is where the
0.181 in earlier drafts of the paper came from. It is not the published artifact and was
not tested on hardware. It is kept outside git, together with its report and gate (F-beta drop 0.058,
SQNR 8.3 dB, KL 11.07 nats; it also fails 3 of 4).
