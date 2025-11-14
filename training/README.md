# Model Training

This folder contains all code and workflows for training image classification models on satellite data.

## Folder Structure

```
training/
├── README.md              # This file
├── scripts/               # Training scripts and utilities
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Model evaluation metrics
│   ├── data_loader.py     # Data loading and augmentation
│   └── __pycache__/
├── notebooks/             # Jupyter notebooks for experimentation
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   ├── 02_model_experiments.ipynb
│   └── 03_visualization.ipynb
└── logs/                  # Training logs and checkpoints
    ├── tensorboard/       # TensorBoard event files
    ├── checkpoints/       # Model weights during training
    └── training_logs.csv  # Metrics log
```

## Purpose

Train deep learning models for image classification tasks:
- **Wildfire detection**: Binary classification (fire/no-fire)
- **Image quality control**: Multi-class classification (quality levels)

## Scripts

### `train.py` - Main Training Script
Trains a classification model with specified architecture and hyperparameters.

**Usage**:
```bash
python training/scripts/train.py \
  --data-dir /path/to/data \
  --model resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001
```

**Key features**:
- Configurable model architecture (ResNet, EfficientNet, etc.)
- Early stopping based on validation loss
- Automatic checkpoint saving
- TensorBoard logging

### `evaluate.py` - Model Evaluation
Computes classification metrics and generates evaluation reports.

**Usage**:
```bash
python training/scripts/evaluate.py \
  --model-path logs/checkpoints/best_model.pth \
  --test-data /path/to/test/data \
  --output-dir results/
```

**Metrics computed**:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC curves
- Per-class performance

### `data_loader.py` - Data Preparation
Handles loading, preprocessing, and augmentation of satellite imagery.

**Features**:
- Image normalization
- Data augmentation (rotation, flip, contrast, etc.)
- Train/val/test splitting
- Batch loading with PyTorch DataLoader

## Notebooks

- **`01_eda.ipynb`**: Exploratory analysis of training data
  - Visualize class distributions
  - Check image statistics (mean, std)
  - Identify data quality issues

- **`02_model_experiments.ipynb`**: Model prototyping and comparison
  - Try different architectures
  - Visualize training curves
  - Compare model performances

- **`03_visualization.ipynb`**: Visualization utilities
  - Plot predictions
  - Saliency maps
  - Attention mechanisms

## Training Workflow

### Step 1: Prepare Data
```bash
# Ensure data is in the correct format and split
# Expected structure:
# data/
#   ├── train/
#   │   ├── class_0/
#   │   └── class_1/
#   ├── val/
#   │   ├── class_0/
#   │   └── class_1/
#   └── test/
#       ├── class_0/
#       └── class_1/
```

### Step 2: Train Model
```bash
python training/scripts/train.py \
  --data-dir data/ \
  --model resnet50 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir training/logs/
```

### Step 3: Monitor Training
```bash
tensorboard --logdir training/logs/tensorboard/
# Open http://localhost:6006 in browser
```

### Step 4: Evaluate Model
```bash
python training/scripts/evaluate.py \
  --model training/logs/checkpoints/best_model.pth \
  --test-data data/test/ \
  --output training/logs/evaluation/
```

## Configuration

### Model Architectures Supported
- ResNet (18, 34, 50, 101)
- EfficientNet (B0-B7)
- Vision Transformer (ViT)
- MobileNet (mobile deployments)

### Training Parameters
- **Batch Size**: 16-64 (depends on GPU memory)
- **Learning Rate**: 0.0001-0.01
- **Optimizer**: Adam, SGD with momentum
- **Scheduler**: CosineAnnealingLR, StepLR
- **Augmentation**: Random rotations, flips, color jitter

## Output Structure

```
training/logs/
├── tensorboard/
│   └── events.out.tfevents.* (TensorBoard logs)
├── checkpoints/
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── best_model.pth (best validation performance)
├── training_logs.csv
└── config.yaml (training configuration used)
```

## Tips for Training

1. **GPU Usage**: Models are trained on GPU if available
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Data Augmentation**: Improves generalization
   - Random rotations (±15°)
   - Horizontal/vertical flips
   - Color jitter (brightness, contrast)
   - Mixup/Cutmix for advanced augmentation

3. **Hyperparameter Tuning**: Use `optimization/` folder for systematic tuning

4. **Early Stopping**: Training stops if validation loss doesn't improve for N epochs

5. **Class Imbalance**: If classes are imbalanced, use weighted loss or SMOTE

## Model Checkpoints

- **Best Model**: `logs/checkpoints/best_model.pth`
  - Saved when validation performance improves
  - Used for final evaluation

- **Latest Model**: `logs/checkpoints/latest.pth`
  - Last checkpoint for resuming training

- **Periodic Checkpoints**: `logs/checkpoints/epoch_*.pth`
  - Saved every N epochs for analysis

## Next Steps

1. After training: Run evaluation with `evaluate.py`
2. For optimization: Use `optimization/` folder for hyperparameter tuning
3. For deployment: Export model to ONNX or TorchScript in `optimization/`

## Dependencies

Required packages (in `requirements.txt`):
- `torch`: Deep learning framework
- `torchvision`: Computer vision utilities
- `tensorboard`: Training visualization
- `numpy`, `pillow`: Data handling
- `opencv-python`: Image processing
