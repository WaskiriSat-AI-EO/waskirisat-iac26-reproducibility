# Final Technical Validation of TFLite Cloud and Fire Classification on OpenMV

## Embedded Batch Evaluation, Cross-Runtime Analysis, and Final Conclusions

**Software artifact:** `openmv_batch_test.py`  
**Software and JSON schema version:** 6.3  
**Target board:** OpenMV Cam H7 Plus (`OPENMV4P`, STM32H743)  
**Embedded runtime:** OpenMV 5.0.0, MicroPython 1.28.0-49  
**Desktop reference runtime:** LiteRT 2.2.0, one CPU thread  
**Evaluation date:** 5 September 2026

## Abstract

This report documents the final implementation and validation of two binary
TensorFlow Lite image classifiers on an OpenMV Cam H7 Plus: cloud detection
and fire detection. The final program discovers a fixed labeled PNG dataset on
the SD card, verifies the dataset manifest and each model's tensor contract,
executes model-specific preprocessing, performs inference with only one model
resident at a time, and writes one self-contained JSON result document per
model.

Both final version 6.3 batches processed 100 of 100 images without runtime or
serialization errors. Cloud detection achieved 74.00% accuracy using the
production-relevant OpenMV RGB565 and bilinear-resize pipeline. Fire detection
achieved 68.00% accuracy using grayscale-to-float32 NCHW preprocessing. The
final Cloud and Fire predictions reproduce their respective previously
validated embedded pipelines exactly for all 100 samples.

Cross-runtime behavior differs substantially by model. Fire decisions agree
with desktop for 94% of the images, with a probability correlation of 0.9557
and a mean absolute error of 0.0383. Cloud decisions agree with desktop for
only 69% of the images when PNG files are processed natively on OpenMV, with a
probability correlation of 0.6049 and a strong positive probability bias of
0.2432. A controlled Cloud experiment using the exact desktop-generated int8
input tensors increased agreement to 97%, correlation to 0.9974, and embedded
accuracy to 84%, compared with 85% on desktop. This isolation experiment
demonstrates that the principal Cloud discrepancy is preprocessing mismatch,
not an incorrect input shape, quantization equation, output interpretation, or
model invocation.

The different cross-runtime behavior is not explained solely by Fire using a
float32 input. Fire starts from native 160 x 120 grayscale images that already
match the model geometry and can be reproduced almost exactly on OpenMV.
Cloud starts from 1280 x 800 RGB888 images, is reduced to RGB565 before
resampling, and is resized by a different interpolation implementation. Color
precision loss and non-equivalent resampling therefore alter the Cloud tensor
before inference.

## 1. Final Acceptance Statement

The following conclusions are supported by the final records and independent
recalculation:

1. The version 6.3 JSON files are structurally valid and internally
   consistent.
2. Both final batches contain 100 unique, successful, sequentially indexed
   records and zero image-level errors.
3. Folder-derived ground truth is balanced at 50 negative and 50 positive
   images per task.
4. Every recorded class decision is consistent with the stored probability
   and the fixed threshold of 0.5.
5. Every correctness flag, confusion-matrix count, and reported aggregate
   metric can be reconstructed from the individual records.
6. Every embedded desktop reference matches the independently generated
   desktop result for the same filename.
7. The final Cloud output sequence is exactly identical to the corrected
   native RGB565 Cloud run previously validated as schema 4.2.
8. The final Fire output sequence is exactly identical to the previously
   validated grayscale Fire implementation.
9. The final implementation is accepted as a correct evaluator of the actual
   embedded pipelines.
10. Native Cloud probabilities must not be described as numerically equivalent
    to desktop probabilities. They describe a different, device-specific input
    pipeline.

Correct implementation and high classifier accuracy are separate properties.
The software implementation is valid, while the measured accuracy remains a
property of each model, preprocessing pipeline, threshold, and dataset.

## 2. Evaluation Scope and Evidence

The final audit uses the following primary result artifacts:

| Artifact | Role |
|---|---|
| `cloud_detection_results(20260905-074452).json` | Final native Cloud result, schema 6.3 |
| `fire_detection_results(20260905-074453).json` | Final Fire result, schema 6.3 |
| `cloud_detection_python_results.json` | Desktop Cloud reference |
| `fire_detection_python_results.json` | Desktop Fire reference |
| `cloud_detection_results(10).json` | Controlled exact-input Cloud experiment |
| `cloud_detection_results(9).json` | Corrected native Cloud preprocessing reference |
| `fire_detection_results(20260905-054036).json` | Previously validated Fire reference |

Intermediate runs containing input-shape errors, invalid reshaping,
unsupported pixel formats, incomplete JSON, or known double scaling were used
only to identify implementation defects. Their accuracy values are not valid
experimental results and are excluded from the final performance comparison.

The supplied dataset contains:

| Task | Negative class | Positive class | Total | Native file properties |
|---|---:|---:|---:|---|
| Cloud | 50 `clear` | 50 `cloud` | 100 | RGB PNG, 1280 x 800 |
| Fire | 50 `no_fire` | 50 `fire` | 100 | 8-bit grayscale PNG, 160 x 120 |

The final evaluator is designed for this fixed experimental dataset. It embeds
the corresponding desktop output references and requires the expected dataset
manifest. A future live-camera application should remove ground-truth folder
logic and fixed desktop references while retaining the validated preprocessing
and model-contract checks.

## 3. Required SD-Card Organization

The final version does not require a precomputed tensor package.

```text
paper_tests/
├── models/
│   ├── cloud_detection.tflite
│   └── fire_detection.tflite
├── cloud_detection/
│   ├── clear/
│   │   └── *.png
│   └── cloud/
│       └── *.png
├── fire_detection/
│   ├── no_fire/
│   │   └── *.png
│   └── fire/
│       └── *.png
└── test_1/
    ├── cloud_detection_results.json
    └── fire_detection_results.json
```

The evaluator searches for the first accessible test root in:

```python
("/sdcard/paper_tests", "/sd/paper_tests", "/paper_tests", "paper_tests")
```

Class folders are visited in negative-then-positive order, and filenames are
sorted lexicographically. This deterministic ordering is necessary because the
desktop reference outputs are stored in the same order.

## 4. Model Tensor Contracts

Version 6.3 validates the complete tensor metadata exposed by `ml.Model` before
processing any image.

| Property | Cloud detection | Fire detection |
|---|---:|---:|
| Model size | 1,941,072 bytes | 1,452,472 bytes |
| Tensor arena requirement | 1,622,736 bytes | 807,504 bytes |
| Input shape | `(1, 224, 224, 3)` | `(1, 3, 120, 160)` |
| Input layout | NHWC | NCHW |
| Input datatype | signed int8 (`'b'`) | float32 (`'f'`) |
| Input scale | 1.0 | 1.0 |
| Input zero point | -128 | 0 |
| Output shape | `(1, 1)` | `(1, 1)` |
| Output datatype | signed int8 (`'b'`) | signed int8 (`'b'`) |
| Output scale | 0.00390625 | 0.04537102 |
| Output zero point | -128 | -5 |
| Output semantic | Probability | Logit |
| Positive class | `cloud` | `fire` |
| Decision threshold | 0.5 | 0.5 |

Input and output scales are compared with a tolerance of `1e-6`; shapes,
datatypes, zero points, and layouts must match exactly. The model file size is
checked before construction. A full SHA-256 scan is intentionally not executed
on the board immediately before model loading because that validation method
caused a measured memory-allocation regression, discussed in Section 15.

The externally verified model SHA-256 values used during development were:

| Model | SHA-256 |
|---|---|
| Cloud | `f528086c23c0e997b906fea87eb02ceec6e2a1d59a596fa6e71c0a6493946d5e` |
| Fire | `6e486fc771f9c7a984d281208980692e1c31340241984f2ff487db5ada1201e8` |

Full model hashing should be performed on a desktop or as a separate
preflight operation followed by a device restart, rather than immediately
before allocating the embedded tensor arena.

## 5. Final Software Architecture

The implementation is contained in one executable Python file. Its principal
responsibilities are:

| Component | Responsibility |
|---|---|
| `root_dir()` | Resolves the available SD-card test root |
| `image_items()` | Discovers images, assigns ground truth, sorts filenames, and validates the manifest |
| `check_model()` | Verifies shapes, datatypes, quantization parameters, and scalar output |
| `load_frame()` | Loads a PNG and draws it into an uncompressed target image |
| `CloudImageInput` | Writes the RGB565-derived NHWC int8 tensor into the model-owned buffer |
| `load_fire_tensor()` | Creates the normalized grayscale-derived NCHW float32 tensor |
| `predict()` | Executes inference and applies the model-specific output transformation |
| `update_stats()` | Accumulates confusion counts, timings, memory, and desktop-comparison sums |
| `comparison_summary()` | Calculates agreement, error, bias, and correlation against desktop |
| `make_summary()` | Produces aggregate classification, runtime, and memory results |
| `write_record()` | Streams and flushes one JSON record at a time |
| `run_model()` | Controls one complete model lifecycle |
| `main()` | Executes Cloud and Fire sequentially |

Cloud runs first because it requires the larger tensor arena and therefore
benefits from the least fragmented memory state. Fire runs second and uses a
single preallocated float32 input tensor that is overwritten for each image.
This avoids 100 repeated allocations of a 230,400-byte tensor.

Only one model is retained at a time. Per-image image and ndarray references
are cleared, garbage collection is invoked at controlled points, and the model
reference is removed before the next model is constructed.

## 6. Final Cloud Preprocessing

The deployed Cloud pipeline is:

```text
1280 x 800 RGB PNG
    -> OpenMV image loading
    -> uncompressed 224 x 224 RGB565 destination
    -> explicit independent x/y bilinear scaling
    -> signed int8 HWC array written into the model buffer
    -> implicit batch dimension: NHWC (1, 224, 224, 3)
```

The destination is allocated explicitly as RGB565. `draw_image()` receives:

```python
x_scale = 224.0 / source_width
y_scale = 224.0 / source_height
hint = image.BILINEAR
```

No aspect-ratio flag is combined with the explicit scale factors. This is a
critical correction: a previous implementation used explicit scaling together
with `SCALE_ASPECT_IGNORE`, effectively applying scaling behavior twice and
placing only a small image region in a larger black tensor.

The source aspect ratio is 1.6, while the model input is square. Both the
desktop and embedded pipelines resize to 224 x 224, so both intentionally
distort the source geometry to a square. The relevant mismatch is not the
target geometry itself; it is the order and numerical implementation of color
conversion and interpolation.

The model input quantizer is:

\[
q = \operatorname{round}(x/s) + z,
\qquad s=1,\quad z=-128,
\]

therefore:

\[
q=x-128,
\qquad x\in[0,255],\quad q\in[-128,127].
\]

`CloudImageInput` calls:

```python
frame.to_ndarray("b", buffer=buffer)
```

This writes directly into the model-owned input bytearray. It avoids allocating
a second 150,528-byte input tensor and preserves HWC order, which is equivalent
to NHWC storage when the batch dimension is one.

## 7. Final Fire Preprocessing

The deployed Fire pipeline is:

```text
160 x 120 grayscale PNG
    -> uncompressed 160 x 120 OpenMV grayscale image
    -> float32 grayscale ndarray
    -> division by 255
    -> duplication into three equal channel planes
    -> NCHW float32 tensor (1, 3, 120, 160)
```

The external normalization is:

\[
x_{normalized}=\frac{x}{255},\qquad x\in[0,255].
\]

No external ImageNet mean or standard-deviation normalization is applied. The
exported model already contains its required downstream normalization logic.

The original Fire files were independently confirmed to be 8-bit grayscale
images at exactly 160 x 120. Consequently:

- no RGB888-to-RGB565 color reduction is required;
- the destination has the same geometry as the source;
- all three final channel planes can be exact copies of one normalized
  grayscale plane;
- the channel ordering is explicitly controlled as NCHW.

For every image, the implementation verifies that the tensor has shape
`(1, 3, 120, 160)` and that its extrema are finite and lie within `[0,1]`.
The first sample of each class additionally stores channel means, extrema, and
fixed row-major sample positions. In the final records, all three channels are
identical as required.

The final implementation reuses one float32 Fire tensor across the full batch.
This optimization leaves the numerical input unchanged while substantially
reducing allocator pressure.

## 8. Output Interpretation

OpenMV returns dequantized output tensors when no custom postprocessor is
provided. For a quantized output value `q`, zero point `z`, and scale `s`:

\[
y=s(q-z).
\]

For auditability, the evaluator reconstructs a quantized estimate from the
dequantized value:

\[
\hat q=\operatorname{clip}_{[-128,127]}
\left(\operatorname{round}(y/s)+z\right).
\]

### 8.1 Cloud

The Cloud model contains its final probability activation. The scalar output
is clipped to `[0,1]` and interpreted directly as:

\[
P(\mathrm{cloud})=y.
\]

### 8.2 Fire

The Fire scalar is a logit. A numerically stable sigmoid is applied:

\[
P(\mathrm{fire})=\sigma(y)=\frac{1}{1+e^{-y}}.
\]

Both models select the positive class when the resulting probability is at
least 0.5. No threshold was optimized on the final test set.

## 9. JSON Integrity and Recalculation Audit

The two final JSON documents passed all of the following checks:

| Check | Cloud | Fire |
|---|---:|---:|
| Strict JSON parsing | Pass | Pass |
| Schema version 6.3 | Pass | Pass |
| Purpose `final_embedded_batch_evaluation` | Pass | Pass |
| Records present | 100 | 100 |
| Successful records | 100 | 100 |
| Error records | 0 | 0 |
| Sequential indices | 100/100 | 100/100 |
| Filename match against desktop | 100/100 | 100/100 |
| Ground-truth match against desktop | 100/100 | 100/100 |
| Threshold-to-class consistency | 100/100 | 100/100 |
| Correctness-flag consistency | 100/100 | 100/100 |
| Reconstructed quantized-output consistency | 100/100 | 100/100 |
| Embedded desktop-reference consistency | 100/100 | 100/100 |
| Recalculated summary consistency | Pass | Pass |

The dataset manifest hashes are:

| Dataset | Manifest SHA-256 |
|---|---|
| Cloud | `595211136ed731e4085fc60cd8db5f2fd14e96fb697acfa1d354f2ba36c4b553` |
| Fire | `c6e1489a88323230a97e3273d21ac1c469a8a847bcd78adc07397f8d28781be5` |

These manifests cover class-relative filenames and their deterministic order;
they do not hash the PNG pixel contents.

## 10. Final Classification Results

| Metric | Native Cloud | Fire |
|---|---:|---:|
| Images evaluated | 100 | 100 |
| Negative / positive images | 50 / 50 | 50 / 50 |
| Runtime errors | 0 | 0 |
| True positives | 49 | 34 |
| True negatives | 25 | 34 |
| False positives | 25 | 16 |
| False negatives | 1 | 16 |
| Accuracy | 74.00% | 68.00% |
| Precision | 66.22% | 68.00% |
| Recall / sensitivity | 98.00% | 68.00% |
| Specificity | 50.00% | 68.00% |
| F1 score | 79.03% | 68.00% |
| Balanced accuracy | 74.00% | 68.00% |
| Matthews correlation coefficient | 0.5472 | 0.3600 |
| ROC-AUC, calculated post hoc | 0.9286 | 0.7060 |
| Accuracy 95% Wilson interval | 64.63-81.60% | 58.34-76.33% |

Cloud has very high sensitivity but low specificity at threshold 0.5. It
detects 49 of 50 cloud images but incorrectly labels 25 of 50 clear images as
cloud. This is consistent with the measured positive probability shift.

Fire is symmetric at the selected threshold: it correctly classifies 34 and
incorrectly classifies 16 images in each class. Its 68% accuracy is not caused
by class imbalance.

The ROC-AUC values were calculated after the run from the per-image scores and
are not fields generated by version 6.3. Cloud's AUC of 0.9286 indicates that
its ranking remains substantially stronger than its thresholded accuracy. In
other words, the native pipeline preserves much of the ordering between clear
and cloudy images but shifts and saturates the probability scale. This suggests
that independent calibration may recover some operational performance, but it
does not establish desktop-equivalent probabilities.

## 11. Score Distributions

| Model and true class | Mean | Median | Minimum | Maximum | Population SD |
|---|---:|---:|---:|---:|---:|
| Native Cloud: `clear` | 0.4663 | 0.4844 | 0.0000 | 0.9961 | 0.3309 |
| Native Cloud: `cloud` | 0.9414 | 0.9961 | 0.4688 | 0.9961 | 0.1265 |
| Fire: `no_fire` | 0.3938 | 0.3515 | 0.1060 | 0.7885 | 0.2012 |
| Fire: `fire` | 0.5253 | 0.5565 | 0.2434 | 0.7649 | 0.1089 |

Twenty-nine Cloud outputs are saturated at the largest representable
probability, `255/256 = 0.99609375`: one clear image and 28 cloud images. The
100 Cloud samples contain only 45 distinct probability levels. This behavior
is expected for a scalar int8 probability output but is intensified by the
native preprocessing shift.

Desktop Cloud scores show a materially different calibration:

| True class | Desktop mean | Desktop median | Native mean | Native median |
|---|---:|---:|---:|---:|
| `clear` | 0.1395 | 0.0117 | 0.4663 | 0.4844 |
| `cloud` | 0.7820 | 0.9258 | 0.9414 | 0.9961 |

Both classes move upward on OpenMV, particularly the clear class. This explains
the increase from six desktop false positives to 25 native false positives.

## 12. Comparison with Desktop LiteRT

| Measurement | Native Cloud | Fire |
|---|---:|---:|
| Matched filenames | 100 | 100 |
| Identical class decisions | 69 | 94 |
| Different class decisions | 31 | 6 |
| Exact reconstructed int8 outputs | 17 | 11 |
| Exact-output rate | 17% | 11% |
| Probability correlation | 0.6049 | 0.9557 |
| Probability MAE | 0.2856 | 0.0383 |
| Probability RMSE | 0.4199 | 0.0514 |
| Mean OpenMV-minus-desktop bias | +0.2432 | +0.0011 |
| Maximum absolute probability difference | 0.9844 | 0.1844 |

Exact output equality is not the appropriate primary equivalence criterion for
two different inference backends. Decision agreement, correlation, bias, and
error magnitude provide a more informative interpretation. Fire has only 11%
bit-equivalent reconstructed outputs but still preserves 94% of decisions and
has near-zero mean bias. Cloud has both low exact equality and materially lower
probability agreement in the native image path.

### 12.1 Accuracy comparison

| Task | Desktop accuracy | OpenMV accuracy | Difference |
|---|---:|---:|---:|
| Cloud, native images | 85% | 74% | -11 percentage points |
| Fire | 66% | 68% | +2 percentage points |
| Cloud, exact tensor diagnostic | 85% | 84% | -1 percentage point |

For paired predictions, an exploratory exact McNemar test gives `p = 0.0708`
for native Cloud versus desktop and `p = 0.6875` for Fire versus desktop. The
100-image sample therefore does not provide strong evidence at the conventional
0.05 level that the thresholded accuracies differ, but this does not imply
numerical equivalence. The Cloud probability bias, MAE, and 31 decision changes
remain operationally important.

Fire's two-point accuracy increase is not evidence that the embedded runtime
improves the model. Four desktop errors become correct on OpenMV, while two
desktop-correct samples become incorrect. This small net change is compatible
with threshold crossings caused by backend numerical differences.

## 13. Controlled Cloud Exact-Tensor Experiment

The exact-tensor diagnostic was designed to separate model execution from image
preprocessing. Desktop Pillow generated all 100 inputs as follows:

```text
RGB888 PNG
    -> Pillow bilinear resize to 224 x 224
    -> int8 quantization q = x - 128
    -> contiguous NHWC tensor bytes
```

Those exact bytes were supplied directly to the OpenMV model callback, bypassing
PNG decoding, RGB565 conversion, and OpenMV resizing. The package size and
SHA-256 were validated, and two individual tensor hashes were checked on the
board.

| Metric | Native RGB565 Cloud | Exact-tensor Cloud | Desktop Cloud |
|---|---:|---:|---:|
| Accuracy | 74% | 84% | 85% |
| Decision agreement with desktop | 69% | 97% | 100% |
| Probability correlation | 0.6049 | 0.9974 | 1.0000 |
| Probability MAE | 0.2856 | 0.0169 | 0 |
| Probability RMSE | 0.4199 | 0.0303 | 0 |
| Probability bias | +0.2432 | +0.0038 | 0 |
| Exact int8 outputs | 17% | 45% | 100% |
| TP / TN / FP / FN | 49 / 25 / 25 / 1 | 41 / 43 / 7 / 9 | 41 / 44 / 6 / 9 |

Three exact-tensor decisions differ from desktop. The exact-tensor run remains
non-bit-exact because LiteRT on desktop and TFLite Micro with optimized embedded
kernels can use different accumulation, rounding, saturation, and operator
implementations. Nevertheless, the near-unity correlation, low error, near-zero
bias, and one-point accuracy difference demonstrate that the deployed Cloud
model contract and output processing are correct.

The tensor package was a controlled diagnostic fixture, not a deployable image
source. Version 6.3 correctly removes it from the production-relevant batch
path.

## 14. Why Fire Transfers Better Than Native Cloud

Fire's float32 input contributes to its closer cross-runtime behavior, but it
is not the sole or principal explanation.

| Factor | Fire | Cloud | Consequence |
|---|---|---|---|
| Original image mode | Grayscale | RGB888 | Cloud has device-specific color conversion |
| Original geometry | 160 x 120 | 1280 x 800 | Cloud requires substantial downsampling |
| Model geometry | 160 x 120 | 224 x 224 | Fire already matches; Cloud does not |
| Embedded intermediate | GRAYSCALE | RGB565 | Cloud loses color precision |
| Model input type | float32 | int8 | Cloud input is directly sensitive to discrete pixel changes |
| Layout conversion | Explicit NCHW duplication | Native HWC/NHWC | Both are controlled, but Fire is simpler |
| External scaling | Divide by 255 | Subtract 128 through quantization | Both equations are validated |
| Desktop agreement | 94% | 69% native / 97% exact tensor | Cloud mismatch is primarily before inference |

### 14.1 Fire preserves the source signal

Every Fire PNG is a single-channel 8-bit image with the exact target width and
height. OpenMV can retain the grayscale value, divide it by 255 in float32, and
copy it into three planes. The desktop RGB conversion of a grayscale image also
produces three equal channels. Therefore, the two runtimes receive nearly the
same logical input tensor.

The float32 model input avoids an external signed-int8 pixel quantization step,
but Fire still has an int8 scalar output. Its residual differences are therefore
consistent with different inference kernels and output quantization. The output
logit step is `0.04537102`; after sigmoid, small logit changes near zero can move
a probability across 0.5. Six of 100 decisions differ for this reason.

### 14.2 Cloud changes the signal before inference

The desktop sequence performs bilinear interpolation on RGB888 values. The
OpenMV path first represents the image as RGB565 and then applies the embedded
bilinear implementation. RGB565 retains five red bits, six green bits, and five
blue bits. Expanded channel levels therefore cannot reproduce all 256 RGB888
values.

Furthermore, color quantization and interpolation are not commutative:

\[
\operatorname{resize}(\operatorname{RGB565}(x))
\ne
\operatorname{RGB565}(\operatorname{resize}(x)).
\]

The coordinate mapping, boundary rules, and rounding of OpenMV's bilinear
operation also differ from Pillow's implementation. Tensor probes from the
corrected diagnostic showed different corner and center values, including
boundary pixels mapped to the int8 value `-128` where the desktop tensor
contained image data. These differences affect many input pixels before the
first network operation.

The Cloud input being int8 is therefore relevant but not itself defective.
When the same int8 tensor bytes were supplied to both runtimes, OpenMV reached
97% decision agreement. The dominant failure of equivalence is how those bytes
are generated from the original image.

## 15. Debugging History and Final Technical Conclusions

The iterative tests identified several independent failure modes. The final
conclusions are summarized below.

| Observed behavior | Root cause | Corrective action | Final status |
|---|---|---|---|
| Fire reported indexing errors | HWC assumptions were applied to an NCHW input | Construct an explicit `(1,3,120,160)` tensor | Resolved |
| Fire input-shape mismatch | The callback or ndarray did not have the exact four-dimensional shape | Allocate the final NCHW shape directly | Resolved |
| Fire `cannot reshape array` | `ulab` reshape and ownership constraints were violated | Avoid reinterpreting or reshaping the model buffer | Resolved |
| Fire `Unsupported pixformat` | A compressed or incompatible image object was passed to ndarray conversion | Draw into an uncompressed GRAYSCALE image first | Resolved |
| Fire initially produced 50% accuracy | Incorrect normalization and/or channel construction | Establish the exported preprocessing contract and use `[0,1]` NCHW | Resolved |
| Fire RGB565 version reached only 58% | Grayscale originals were unnecessarily quantized through RGB565, producing unequal channels | Preserve original grayscale and duplicate it after normalization | Resolved; final 68% and 94% desktop agreement |
| Cloud initial run reached 70% and 63% desktop agreement | Explicit scale factors were combined with `SCALE_ASPECT_IGNORE`, producing double-scaling behavior | Remove aspect flags when explicit x/y scales are supplied | Resolved |
| Cloud corrected native run reached 74% but only 69% desktop agreement | RGB565 reduction and non-equivalent embedded bilinear resizing | Accept as the native device pipeline; isolate with exact tensors | Characterized, not removable in software without changing the input source |
| Cloud area-resampling diagnostic appeared to hang | The operation was too expensive for the board and dataset size | Use the completed bilinear implementation | Resolved operationally |
| Exact Cloud diagnostic reported `ENOENT` | The external tensor package was absent or misnamed | Install the diagnostic package at its exact path | Resolved for the isolation experiment |
| Fire-first combined execution produced `Failed to allocate tensors` for Cloud | Repeated Fire tensor/image allocations fragmented memory before the larger Cloud arena | Run Cloud first and reuse one Fire tensor | Resolved |
| Cloud-first version 6.2 still produced `Failed to allocate tensors` | On-device full-model SHA-256 validation left about 2 MB of transient block allocations before `ml.Model` | Remove immediate full-file hashing, check size and contract, then call `gc.collect()` | Resolved in version 6.3 |

The model-allocation failure is particularly important. Version 6.2 began with
4,280,160 free bytes but reported only 2,218,400 bytes immediately after the
Cloud model checksum. The decrease was approximately the model file size. The
Cloud arena then could not obtain a contiguous 1,622,736-byte region. Version
6.3 avoids that high-allocation preflight operation and successfully executes
both models in one run.

Intermediate all-error JSON files are diagnostic evidence only. They must not
be pooled with valid predictions or used to estimate model accuracy.

## 16. Runtime Performance

| Measurement | Native Cloud | Fire |
|---|---:|---:|
| Model load | 199.897 ms | 178.199 ms |
| Mean PNG read | 108.705 ms | 34.167 ms |
| Mean image conversion / resize | 580.738 ms | 4.819 ms |
| Mean tensor preparation | 92.758 ms | 42.896 ms |
| Median tensor preparation | 2.344 ms | 42.741 ms |
| P95 tensor preparation | 902.688 ms | 44.971 ms |
| Mean estimated model inference | 1,746.563 ms | 670.370 ms |
| Median estimated model inference | 1,746.632 ms | 670.369 ms |
| P95 estimated model inference | 1,746.995 ms | 670.755 ms |
| Mean measured sample interval | 3,454.059 ms | 893.101 ms |
| Median measured sample interval | 3,360.395 ms | 892.273 ms |
| P95 measured sample interval | 4,011.793 ms | 906.228 ms |
| JSON record writes, total | 1.883 s | 1.664 s |
| Complete batch | 396.532 s | 96.900 s |
| Effective complete-batch throughput | 0.252 images/s | 1.032 images/s |

Cloud inference is approximately 2.61 times slower than Fire inference, while
the complete Cloud batch is approximately 4.09 times longer. The additional
end-to-end difference is dominated by reading and reducing 1280 x 800 color
PNGs to 224 x 224 RGB565 images.

The Cloud model inference itself is exceptionally stable: its population
standard deviation is approximately 0.374 ms around a mean of 1,746.563 ms.
The high mean and maximum of `tensor_prepare_ms` are produced by periodic
allocation and garbage-collection events associated with the callback. Its
median is only 2.344 ms, but eight records show large preparation spikes, with
a maximum of 1,386.428 ms. These spikes raise `predict_total_ms` without
changing `inference_estimated_ms` or predictions.

The final Cloud batch is 0.80% longer than the prior corrected RGB565 run
(396.532 s versus 393.396 s) and produces exactly the same 100 outputs. This is
normal storage and garbage-collection variability, not an inference regression.

The desktop latency values, approximately 3.662 ms for Cloud and 0.938 ms for
Fire in the supplied runs, are not directly comparable performance targets.
They were measured on a general-purpose processor with a different runtime,
memory system, and optimized operator implementation. The embedded latency is
the relevant deployment measurement for the STM32H743.

## 17. Memory Results

| Measurement | Native Cloud | Fire |
|---|---:|---:|
| Free before model construction | 4,276,784 bytes | 4,291,184 bytes |
| Minimum sampled free memory | 504,960 bytes | 1,715,424 bytes |
| Free after model release | 4,275,024 bytes | 4,289,424 bytes |
| Difference after release | -1,760 bytes | -1,760 bytes |
| Nominal tensor arena | 1,622,736 bytes | 807,504 bytes |

Both models completed without an out-of-memory event. The difference between
pre-model and post-release free memory is only 1,760 bytes for each lifecycle,
which does not indicate a material progressive leak.

The Fire tensor-reuse optimization materially improved sampled headroom. The
previous final Fire implementation recorded a minimum of 494,064 free bytes;
version 6.3 records 1,715,424 bytes while producing the exact same 100 output
values. The increase is approximately 1.22 MB and confirms that eliminating
repeated large tensor allocations improved allocator behavior.

`minimum_free_bytes` is sampled after successful inference. It is not a
continuous peak-memory profiler, so a shorter transient minimum may occur
between samples. Nevertheless, the successful complete run and memory recovery
provide practical evidence that the final sequential lifecycle is stable for
this evaluator.

## 18. Statistical Interpretation and Limitations

The dataset is balanced but small. Each reported accuracy is based on 100
images, so its uncertainty is non-trivial. The Wilson intervals in Section 10
illustrate that a difference of several percentage points cannot automatically
be generalized to the deployment population.

Additional limitations are:

1. The images are processed from the SD card, not captured directly from the
   target camera sensor under operational illumination.
2. Samples may share acquisition conditions or scene characteristics, reducing
   their effective independence.
3. The test evaluates one threshold, 0.5, selected before the final comparison.
4. The same fixed dataset was used throughout debugging. It must not be used to
   select a supposedly optimal production threshold.
5. Desktop and OpenMV timing measurements were taken on fundamentally different
   hardware and are not hardware-normalized.
6. The OpenMV results apply specifically to OpenMV 5.0.0, MicroPython
   1.28.0-49, and the STM32H743 target.
7. The fixed output references make version 6.3 an experimental evaluator,
   not a general-purpose live-inference application.
8. High desktop agreement validates implementation fidelity but does not prove
   that the classifier generalizes to new scenes.

The Cloud AUC suggests that threshold calibration could improve accuracy, but
choosing a threshold on these 100 final images would leak test information and
produce an optimistic estimate. Calibration requires a separate, representative
validation set, after which the frozen threshold must be evaluated on a new
held-out test set.

## 19. Recommendations

### 19.1 Cloud model

The preferred production solution is to adapt the training pipeline to the
actual embedded input path:

1. Reproduce RGB565 channel quantization during training and validation.
2. Reproduce the OpenMV resize order, coordinate behavior, and boundary rules.
3. Train or fine-tune using 224 x 224 tensors produced by that pipeline.
4. Include realistic camera noise, exposure, compression, and illumination
   augmentation.
5. Use representative quantization calibration data.
6. Calibrate the operating threshold on an independent validation set.
7. Evaluate once on a new held-out test set captured through the final camera
   and firmware path.

If full RGB888 preprocessing is an immutable requirement, it must be provided
by hardware or software capable of preserving RGB888 through resize, such as a
companion processor. Reconstructing RGB888 after RGB565 conversion cannot
recover the discarded color bits.

### 19.2 Fire model

The Fire preprocessing is sufficiently faithful to desktop. Further effort
should focus on model discrimination and data quality rather than additional
tensor-layout changes:

1. Expand the dataset across environments, fire sizes, smoke conditions, and
   confusing non-fire heat or light sources.
2. Review false positives and false negatives by acquisition scenario.
3. Calibrate the threshold on an independent validation set.
4. Preserve grayscale loading, `[0,1]` scaling, equal channel duplication, and
   NCHW ordering.
5. Do not route grayscale data through RGB565 unless the deployed sensor path
   genuinely produces RGB565 data.

Fire's 94% desktop agreement means its embedded implementation is faithful; it
does not mean the classifier's 68% accuracy is sufficient for a safety-critical
fire alarm. Deployment decisions require application-specific recall targets,
false-alarm costs, and a larger independently collected dataset.

### 19.3 Embedded software

1. Keep Cloud first in the sequential batch evaluator.
2. Keep the reusable Fire input tensor.
3. Maintain incremental JSON writing through a temporary file.
4. Retain tensor-contract checks and dataset-manifest validation.
5. Perform full model hashing on desktop or before a reboot, not immediately
   before allocating the model arena.
6. Record firmware, model version, threshold, sensor configuration, and
   preprocessing version for every future experimental campaign.
7. For live inference, remove fixed desktop output arrays and ground-truth
   folder assumptions while preserving preprocessing functions.

## 20. Final Conclusion

Version 6.3 is the final validated batch-evaluation implementation for the
supplied OpenMV experiment. Both models complete all 100 samples, recover their
allocated memory, generate mathematically correct JSON summaries, and reproduce
their previously validated embedded output sequences exactly.

Fire is close to desktop because its grayscale 160 x 120 source can be mapped
to the float32 NCHW input with almost no information loss or geometric change.
Its remaining six decision differences are consistent with embedded-versus-
desktop kernel and quantized-output behavior.

Native Cloud is not probability-equivalent to desktop because RGB888 data are
reduced to RGB565 before being resized by a non-identical bilinear
implementation. The exact-tensor experiment proves that the Cloud model and
runtime invocation are correct: with identical input bytes, OpenMV agrees with
desktop on 97% of decisions and differs by only one accuracy point. Therefore,
the 74% native result is the measured performance of the complete OpenMV image
pipeline, not evidence of an incorrectly scaled or incorrectly invoked model.

The final engineering decision is consequently clear: retain version 6.3 as
the reproducible native-device benchmark, retain the exact-tensor run as
evidence isolating runtime behavior, and adapt future Cloud training and
calibration to the RGB565 OpenMV deployment path.
