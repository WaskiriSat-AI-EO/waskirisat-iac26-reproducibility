"""
Visualization utilities for satellite image datasets and model predictions.

This module provides functions for:
- Displaying sample batches from datasets
- Visualizing class distributions
- Plotting training metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple


def display_batch_samples(
    dataset: tf.data.Dataset,
    classes: Dict[int, str],
    num_samples_per_class: int = 4,
    figsize: Tuple[int, int] = None,
) -> None:
    """
    Display sample images from a dataset, organized by class.

    Args:
        dataset: TensorFlow dataset containing (images, labels) tuples
        classes: Dictionary mapping class indices to class names
        num_samples_per_class: Number of samples to display per class (default: 4)
        figsize: Figure size as (width, height). If None, auto-calculated based on num_classes

    Returns:
        None (displays matplotlib figure)

    Example:
        >>> from training.scripts.data_loader import create_dataloaders
        >>> from training.utils import display_batch_samples
        >>> train_ds, val_ds, test_ds, classes = create_dataloaders("/path/to/dataset")
        >>> display_batch_samples(train_ds, classes)
    """
    # Get one batch from dataset
    for images, labels in dataset.take(1):
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        break
    else:
        raise ValueError("Dataset is empty or could not retrieve a batch")

    num_classes = len(classes)

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (16, 4 * num_classes)

    # Create figure with subplots for each class
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=figsize)

    # Flatten axes if only one class or if num_samples_per_class is 1
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    if num_samples_per_class == 1 and num_classes > 1:
        axes = axes.reshape(num_classes, 1)

    # Get indices for each class in the batch
    for class_idx in range(num_classes):
        class_name = classes[class_idx]
        class_mask = batch_labels == class_idx
        class_indices = np.where(class_mask)[0][:num_samples_per_class]

        # Display samples for this class
        for col, img_idx in enumerate(class_indices):
            ax = axes[class_idx, col] if num_classes > 1 else axes[col]
            img = batch_images[img_idx]

            # Normalize from [0, 1] to display if needed
            if img.max() <= 1.0:
                img_display = img
            else:
                img_display = img / 255.0

            # Handle grayscale (2D) or RGB (3D)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                ax.imshow(np.squeeze(img_display), cmap="gray")
            else:
                ax.imshow(
                    img_display.astype(np.uint8) if img.max() > 1 else img_display
                )

            ax.set_title(f"{class_name}")
            ax.axis("off")

        # Hide remaining subplots in this row if fewer than num_samples_per_class
        for col in range(len(class_indices), num_samples_per_class):
            ax_hide = axes[class_idx, col] if num_classes > 1 else axes[col]
            ax_hide.axis("off")

    plt.tight_layout()
    plt.show()

    # Print batch information
    print(f"✓ Batch shape: {batch_images.shape}")
    print(f"✓ Classes: {[classes[i] for i in range(num_classes)]}")
    print(f"✓ Samples displayed: {num_samples_per_class} per class")


def display_class_distribution(
    labels: np.ndarray, classes: Dict[int, str], figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Display bar chart of class distribution.

    Args:
        labels: Array of label indices
        classes: Dictionary mapping class indices to class names
        figsize: Figure size as (width, height)

    Returns:
        None (displays matplotlib figure)
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [classes[idx] for idx in unique]

    plt.figure(figsize=figsize)
    plt.bar(class_names, counts, color="steelblue", alpha=0.8)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(f"✓ Class distribution:")
    for class_name, count in zip(class_names, counts):
        print(f"  - {class_name}: {count} samples")


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully.")
    print("Use: from training.utils import display_batch_samples")
