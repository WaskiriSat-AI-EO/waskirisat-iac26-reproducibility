# Model Optimization

This folder contains tools and workflows for optimizing trained models for performance, efficiency, and deployment.

## Folder Structure

```
optimization/
├── README.md              # This file
├── scripts/               # Optimization scripts
│   ├── hyperparameter_tuning.py  # Grid/random search over hyperparameters
│   ├── quantization.py           # Model quantization (INT8, FP16)
│   ├── pruning.py                # Weight pruning and sparsification
│   ├── distillation.py           # Knowledge distillation
│   └── __pycache__/
├── configs/               # Optimization configuration files
│   ├── tuning_config.yaml        # Hyperparameter search space
│   ├── quantization_config.yaml  # Quantization settings
│   └── pruning_config.yaml       # Pruning settings
└── results/               # Optimization results and reports
    ├── hyperparameter_tuning/
    ├── quantization_results/
    └── optimization_summary.csv
```

## Purpose

Optimize trained models for:
- **Performance**: Faster inference, lower latency
- **Efficiency**: Reduced model size, lower memory usage
- **Accuracy**: Find best hyperparameters without sacrificing accuracy
- **Deployment**: Prepare models for edge devices (satellite payload)

## Scripts

### `hyperparameter_tuning.py` - Hyperparameter Optimization
Systematically searches for the best hyperparameter combinations.

**Usage**:
```bash
python optimization/scripts/hyperparameter_tuning.py \
  --base-model training/logs/checkpoints/best_model.pth \
  --config optimization/configs/tuning_config.yaml \
  --n-trials 50 \
  --output optimization/results/hyperparameter_tuning/
```

**Search Methods**:
- Grid Search: Exhaustive over predefined values
- Random Search: Random sampling
- Bayesian Optimization: Intelligent sampling (Optuna)

**Tuned Hyperparameters**:
- Learning rate
- Batch size
- Optimizer (Adam, SGD)
- Weight decay (L2 regularization)
- Dropout rates
- Data augmentation intensity

**Output**:
- Best hyperparameter set
- Performance comparison table
- Training curves for each trial

### `quantization.py` - Model Quantization
Reduces model size by converting weights to lower precision.

**Usage**:
```bash
python optimization/scripts/quantization.py \
  --model training/logs/checkpoints/best_model.pth \
  --quantization-type int8 \
  --output optimization/results/quantization_results/
```

**Quantization Types**:
- **INT8 (8-bit integer)**: 4× smaller, ~1-5% accuracy loss
- **FP16 (half-precision float)**: 2× smaller, ~0.5% accuracy loss
- **Dynamic Quantization**: Only weights quantized, activations full precision
- **Static Quantization**: Both weights and activations quantized

**Output**:
- Quantized model (`.pth` or `.onnx`)
- Size reduction metrics
- Accuracy comparison
- Inference speed improvements

### `pruning.py` - Model Pruning
Removes unimportant weights to reduce model complexity.

**Usage**:
```bash
python optimization/scripts/pruning.py \
  --model training/logs/checkpoints/best_model.pth \
  --pruning-ratio 0.3 \
  --method structured \
  --output optimization/results/pruning/
```

**Pruning Methods**:
- **Unstructured**: Remove individual weights (sparse matrices)
- **Structured**: Remove entire channels/filters (hardware efficient)
- **Magnitude-based**: Remove small weights
- **Gradual pruning**: Progressive pruning during fine-tuning

**Output**:
- Pruned model
- Sparsity metrics
- Inference speed improvements
- Accuracy loss analysis

### `distillation.py` - Knowledge Distillation
Trains a smaller "student" model using a larger "teacher" model.

**Usage**:
```bash
python optimization/scripts/distillation.py \
  --teacher-model training/logs/checkpoints/best_model.pth \
  --student-model mobilenet_small \
  --temperature 4 \
  --alpha 0.7 \
  --output optimization/results/distillation/
```

**Process**:
1. Use large trained model as "teacher"
2. Train smaller model (student) to mimic teacher
3. Results in 50-70% smaller model with similar accuracy

**Output**:
- Student model
- Performance comparison (teacher vs. student)
- Inference speed vs. accuracy trade-off

## Configuration Files

### `tuning_config.yaml` - Hyperparameter Search Space
```yaml
search_method: bayesian  # grid, random, or bayesian
n_trials: 100
parameters:
  learning_rate:
    type: loguniform
    low: 1e-4
    high: 1e-2
  batch_size:
    type: choice
    values: [16, 32, 64]
  optimizer:
    type: choice
    values: [adam, sgd]
  weight_decay:
    type: uniform
    low: 0
    high: 0.01
```

### `quantization_config.yaml` - Quantization Settings
```yaml
quantization_type: int8  # int8, fp16, dynamic, static
calibration_data: data/val/  # Data for calibration
backend: qnnpack  # ARM optimization
```

### `pruning_config.yaml` - Pruning Settings
```yaml
pruning_method: structured  # unstructured or structured
pruning_ratio: 0.3  # Remove 30% of weights
fine_tune_epochs: 10
fine_tune_lr: 0.0001
```

## Optimization Workflow

### Scenario 1: Faster Inference (Primary Goal)
1. **Quantization** (INT8):
   ```bash
   python optimization/scripts/quantization.py \
     --model best_model.pth --quantization-type int8
   ```
   → 4× faster, minimal accuracy loss

2. **Evaluate Speed**:
   - Original: 100 ms/image
   - Quantized: 25 ms/image ✓

### Scenario 2: Smaller Model (Edge Device Deployment)
1. **Pruning** (remove 40% weights):
   ```bash
   python optimization/scripts/pruning.py \
     --model best_model.pth --pruning-ratio 0.4
   ```
   → 60% smaller, 2-3% accuracy loss

2. **Quantization** (INT8):
   ```bash
   python optimization/scripts/quantization.py \
     --model pruned_model.pth --quantization-type int8
   ```
   → Additional 4× reduction

3. **Result**: 16× smaller than original with acceptable accuracy

### Scenario 3: Best Accuracy (Research/Production)
1. **Hyperparameter Tuning**:
   ```bash
   python optimization/scripts/hyperparameter_tuning.py \
     --config tuning_config.yaml --n-trials 100
   ```
   → Find best hyperparameters

2. **Retrain** with best hyperparameters:
   ```bash
   python training/scripts/train.py --best-hyperparameters
   ```

3. **Fine Quantization** (FP16 for minimal loss):
   ```bash
   python optimization/scripts/quantization.py \
     --model best_model.pth --quantization-type fp16
   ```

## Results and Metrics

### Typical Results (ResNet50 on satellite data)

| Optimization | Model Size | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| Original | 98 MB | 1.0× | 92.5% | Baseline |
| INT8 Quantization | 25 MB | 4.0× | 92.0% | Fast inference |
| 30% Pruning | 68 MB | 1.3× | 91.8% | Balanced |
| Pruning + INT8 | 17 MB | 5.5× | 91.2% | Edge devices |
| Distillation (MobileNet) | 8 MB | 8.0× | 88.5% | Lightweight |

## Output Directory Structure

```
optimization/results/
├── hyperparameter_tuning/
│   ├── best_params.yaml
│   ├── trial_results.csv
│   └── optimization_curves.png
├── quantization_results/
│   ├── model_int8.pth
│   ├── calibration_report.txt
│   └── speed_comparison.json
├── pruning/
│   ├── pruned_model.pth
│   ├── sparsity_analysis.json
│   └── fine_tuned_model.pth
├── distillation/
│   ├── student_model.pth
│   └── teacher_vs_student.csv
└── optimization_summary.csv
```

## Performance Guidelines

### Satellite Payload Constraints
- **CPU**: ARM processor (~2 GHz)
- **Memory**: 512 MB - 2 GB
- **Latency requirement**: <100 ms per image
- **Target model size**: <50 MB

**Recommended optimization pipeline**:
1. INT8 Quantization → 4× faster
2. Structured Pruning (20-30%) → 1.5× faster
3. Knowledge Distillation → 50% smaller
4. **Final**: ~16 MB, 8-10× faster, 88-90% accuracy

## Tips

1. **Always validate** after optimization - accuracy loss is expected
2. **Start with quantization** - easiest and most effective
3. **Combine optimizations** - pruning + quantization > either alone
4. **Use calibration data** - representative data improves quantized model quality
5. **Test on target hardware** - inference speed varies by device

## Next Steps

1. Choose optimization strategy based on constraints
2. Run relevant optimization script(s)
3. Validate optimized model with `training/scripts/evaluate.py`
4. Export to deployment format (ONNX, TorchScript, etc.)
5. Test on satellite payload hardware

## Dependencies

Required packages (in `requirements.txt`):
- `torch`, `torchvision`: Deep learning
- `optuna`: Bayesian hyperparameter optimization
- `onnx`, `onnxruntime`: Model export/inference
- `numpy`, `pandas`: Data analysis
