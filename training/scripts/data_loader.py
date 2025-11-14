"""
Data loading utilities for satellite image classification with TensorFlow.

This module provides utilities for loading satellite imagery organized in a
class-based folder structure with automatic train/val/test splitting,
augmentation, and reproducibility features.

Expected folder structure:
    dataset/
    ├── fire/
    │   ├── image_001.png
    │   ├── image_002.jpg
    │   └── ...
    ├── no_fire/
    │   ├── image_001.png
    │   └── ...
    └── cloud/
        └── ...

Classes are automatically discovered from subfolder names.
"""

import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteImageDataset:
    """TensorFlow Dataset builder.

    Automatically discovers classes from subdirectory structure.
    Supports auto-splitting into train/val/test sets with reproducibility.

    Attributes:
        image_paths: List of image file paths
        labels: List of corresponding labels (class indices)
        class_to_idx: Dictionary mapping class names to indices
        idx_to_class: Dictionary mapping indices to class names
    """

    # Standard image extensions to search for
    SUPPORTED_EXTENSIONS = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tiff",
        "*.tif",
        "*.bmp",
        "*.gif",
        "*.webp",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
        "*.TIFF",
        "*.TIF",
    )

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
        verbose: bool = True,
    ):
        """Initialize dataset builder.

        Args:
            dataset_path: Path to dataset root directory containing class subdirectories
            split: Which split to load ('train', 'val', or 'test')
            train_ratio: Proportion of data for training (0-1)
            val_ratio: Proportion of data for validation (0-1)
            test_ratio: Proportion of data for testing (0-1)
            target_size: Target image size as (height, width)
            seed: Random seed for reproducible splitting
            verbose: Whether to print dataset information

        Raises:
            ValueError: If ratios don't sum to 1, no classes found, or invalid split
            FileNotFoundError: If dataset_path doesn't exist
        """
        self.dataset_path = Path(dataset_path)
        self.split = split.lower()
        self.target_size = target_size
        self.seed = seed

        # Validate inputs
        self._validate_inputs(train_ratio, val_ratio, test_ratio, split)

        # Discover classes and images
        self.class_to_idx, self.idx_to_class = self._discover_classes()
        self.all_image_paths = self._find_all_images()

        if not self.all_image_paths:
            raise FileNotFoundError(
                f"No images found in {self.dataset_path}. "
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )

        # Split dataset
        self.image_paths, self.labels = self._split_dataset(
            train_ratio, val_ratio, test_ratio, self.split
        )

        if verbose:
            self._print_info()

    def _validate_inputs(
        self, train_ratio: float, val_ratio: float, test_ratio: float, split: str
    ) -> None:
        """Validate input parameters."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Check split ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        # Check valid split name
        valid_splits = {"train", "val", "test", "validation"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    def _discover_classes(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Discover class labels from subdirectories.

        Returns:
            Tuple of (class_to_idx, idx_to_class) dictionaries

        Raises:
            FileNotFoundError: If no subdirectories found
        """
        # Find all subdirectories
        class_dirs = sorted(
            [
                d
                for d in self.dataset_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

        if not class_dirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {self.dataset_path}"
            )

        # Create mappings
        class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        idx_to_class = {i: d.name for i, d in enumerate(class_dirs)}

        logger.info(
            f"Discovered {len(class_to_idx)} classes: {list(class_to_idx.keys())}"
        )

        return class_to_idx, idx_to_class

    def _find_all_images(self) -> list:
        """Find all images in dataset and their class labels.

        Returns:
            List of (image_path, class_index) tuples
        """
        all_images = []
        seen = set()

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.dataset_path / class_name

            # Find all image files with supported extensions
            for ext_pattern in self.SUPPORTED_EXTENSIONS:
                for img_path in sorted(class_dir.glob(ext_pattern)):
                    if img_path not in seen:
                        all_images.append((str(img_path), class_idx))
                        seen.add(img_path)

        return all_images

    def _split_dataset(
        self, train_ratio: float, val_ratio: float, test_ratio: float, split: str
    ) -> Tuple[list, list]:
        """Split dataset into train/val/test with stratification by class.

        Uses stratified splitting to maintain class distribution across splits.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            split: Which split to return ('train', 'val', or 'test')

        Returns:
            Tuple of (image_paths, labels) for the requested split
        """
        # Handle 'validation' as alias for 'val'
        split = "val" if split == "validation" else split

        # Group images by class for stratified split
        from collections import defaultdict

        images_by_class = defaultdict(list)

        for img_path, class_idx in self.all_image_paths:
            images_by_class[class_idx].append(img_path)

        # Perform stratified split
        np.random.seed(self.seed)
        train_paths, val_paths, test_paths = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for class_idx in sorted(images_by_class.keys()):
            class_images = np.array(images_by_class[class_idx])
            n = len(class_images)

            # Shuffle within class
            indices = np.random.permutation(n)
            class_images = class_images[indices]

            # Split by ratios
            train_n = int(np.ceil(n * train_ratio))
            val_n = int(np.ceil(n * val_ratio))

            train_paths.extend(class_images[:train_n].tolist())
            val_paths.extend(class_images[train_n : train_n + val_n].tolist())
            test_paths.extend(class_images[train_n + val_n :].tolist())

            # Add labels
            train_labels.extend([class_idx] * len(class_images[:train_n]))
            val_labels.extend(
                [class_idx] * len(class_images[train_n : train_n + val_n])
            )
            test_labels.extend([class_idx] * len(class_images[train_n + val_n :]))

        # Select requested split
        if split == "train":
            paths, labels = train_paths, train_labels
        elif split == "val":
            paths, labels = val_paths, val_labels
        elif split == "test":
            paths, labels = test_paths, test_labels

        logger.info(f"Split '{split}': {len(paths)} images")

        return paths, labels

    def _print_info(self) -> None:
        """Print dataset information."""
        logger.info(f"Dataset loaded from: {self.dataset_path}")
        logger.info(f"Split: {self.split}")
        logger.info(f"Total samples: {len(self.image_paths)}")
        logger.info(f"Classes: {list(self.class_to_idx.keys())}")

        # Count samples per class
        from collections import defaultdict

        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[self.idx_to_class[label]] += 1

        for class_name, count in sorted(class_counts.items()):
            pct = 100 * count / len(self.image_paths)
            logger.info(f"  {class_name}: {count} ({pct:.1f}%)")

    def _load_and_preprocess_image(
        self, image_path: str, label: int
    ) -> Tuple[tf.Tensor, int]:
        """Load and preprocess image.

        Args:
            image_path: Path to image file
            label: Class label

        Returns:
            Tuple of (image_tensor, label)
        """
        # Wrap Python function to handle file I/O
        def load_image_py(image_path):
            # Read image
            image = tf.io.read_file(image_path)
            
            # Try JPEG first
            try:
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                try:
                    # Try PNG
                    image = tf.image.decode_png(image, channels=3)
                except:
                    # Try auto-decode
                    image = tf.image.decode_image(image, channels=3)
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = tf.stack([image, image, image], axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            
            return image
        
        image = load_image_py(image_path)
        
        # Resize
        image = tf.image.resize(image, self.target_size)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        return image, label

    def get_dataset(
        self,
        batch_size: int = 32,
        augmentation: Optional[Callable] = None,
        shuffle: bool = True,
        prefetch_buffer = None,
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset.

        Args:
            batch_size: Batch size
            augmentation: Optional data augmentation function to apply
            shuffle: Whether to shuffle the dataset
            prefetch_buffer: Buffer size for prefetching (None = AUTOTUNE)

        Returns:
            tf.data.Dataset configured with batching, optionally shuffled and augmented
        """
        # Use AUTOTUNE if not specified
        if prefetch_buffer is None:
            prefetch_buffer = tf.data.AUTOTUNE
        
        # Convert to numpy arrays to avoid tensor issues
        image_paths = np.array([str(p) for p in self.image_paths])
        labels = np.array(self.labels, dtype=np.int32)
        
        # Create dataset from list
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(self.image_paths),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        
        # Define load function that works with dataset.map()
        def load_and_preprocess(image_path, label):
            # Read image file
            image_data = tf.io.read_file(image_path)
            
            # Decode - try different formats
            try:
                image = tf.image.decode_jpeg(image_data, channels=3)
            except:
                try:
                    image = tf.image.decode_png(image_data, channels=3)
                except:
                    image = tf.image.decode_image(image_data, channels=3)
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = tf.stack([image, image, image], axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            elif image.shape[-1] > 3:
                image = image[..., :3]
            
            # Resize
            image = tf.image.resize(image, self.target_size)
            
            # Normalize to [0, 1]
            image = image / 255.0
            
            return image, label

        # Load and preprocess images
        dataset = dataset.map(
            load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply augmentation if provided
        if augmentation is not None:
            dataset = dataset.map(
                lambda x, y: (augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch
        dataset = dataset.batch(batch_size)

        # Prefetch
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def get_class_weights(self) -> tf.Tensor:
        """Compute class weights for imbalanced datasets.

        Returns:
            Dictionary mapping class indices to weights

        Example:
            >>> weights = dataset.get_class_weights()
            >>> loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ...     from_logits=True,
            ...     class_weight=weights
            ... )
        """
        from collections import defaultdict

        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[label] += 1

        # Inverse frequency weighting
        weights = {}
        total = len(self.labels)
        for class_idx in range(len(self.class_to_idx)):
            if class_counts[class_idx] > 0:
                weights[class_idx] = total / (
                    len(self.class_to_idx) * class_counts[class_idx]
                )
            else:
                weights[class_idx] = 1.0

        return weights


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_size: Tuple[int, int] = (224, 224),
    train_augmentation: Optional[Callable] = None,
    val_augmentation: Optional[Callable] = None,
    test_augmentation: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
    """
    Create train, validation, and test TensorFlow datasets.

    Automatically discovers classes, splits dataset, and applies augmentation.

    Args:
        dataset_path: Path to dataset root with class subdirectories
        batch_size: Batch size for all datasets
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        target_size: Target image size (height, width)
        train_augmentation: Optional augmentation function for training
        val_augmentation: Optional augmentation for validation (usually None)
        test_augmentation: Optional augmentation for testing (usually None)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, idx_to_class)
        where idx_to_class maps class indices to class names

    Example:
        >>> from training.scripts.augmentation import get_train_augmentation
        >>>
        >>> train_aug = get_train_augmentation()
        >>>
        >>> train_ds, val_ds, test_ds, classes = create_dataloaders(
        ...     dataset_path="data/",
        ...     batch_size=32,
        ...     train_augmentation=train_aug,
        ...     seed=42
        ... )
        >>>
        >>> # Inspect first batch
        >>> for images, labels in train_ds.take(1):
        ...     print(f"Images shape: {images.shape}")  # (32, 224, 224, 3)
        ...     print(f"Labels shape: {labels.shape}")  # (32,)
    """
    # Create dataset builders
    train_dataset_builder = SatelliteImageDataset(
        dataset_path=dataset_path,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        target_size=target_size,
        seed=seed,
        verbose=True,
    )

    val_dataset_builder = SatelliteImageDataset(
        dataset_path=dataset_path,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        target_size=target_size,
        seed=seed,
        verbose=False,
    )

    test_dataset_builder = SatelliteImageDataset(
        dataset_path=dataset_path,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        target_size=target_size,
        seed=seed,
        verbose=False,
    )

    # Create datasets
    train_ds = train_dataset_builder.get_dataset(
        batch_size=batch_size, augmentation=train_augmentation, shuffle=True
    )

    val_ds = val_dataset_builder.get_dataset(
        batch_size=batch_size, augmentation=val_augmentation, shuffle=False
    )

    test_ds = test_dataset_builder.get_dataset(
        batch_size=batch_size, augmentation=test_augmentation, shuffle=False
    )

    logger.info(f"✓ TensorFlow datasets created successfully")
    logger.info(f"  Image shape: (batch_size, {target_size[0]}, {target_size[1]}, 3)")
    logger.info(f"  Batch size: {batch_size}")

    return train_ds, val_ds, test_ds, train_dataset_builder.idx_to_class


def main():
    """CLI entry point for data loading utility.

    Allows creating and inspecting datasets from terminal.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and inspect TensorFlow datasets for satellite image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create datasets with default settings
  python -m training.scripts.data_loader --dataset-path data/
  
  # Custom batch size and splits
  python -m training.scripts.data_loader --dataset-path data/ --batch-size 64 --train-ratio 0.8 --val-ratio 0.1
  
  # Inspect only training split
  python -m training.scripts.data_loader --dataset-path data/ --split train
  
  # Custom image size
  python -m training.scripts.data_loader --dataset-path data/ --target-size 256 256
        """,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset root directory with class subdirectories",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for datasets (default: 32)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for testing (default: 0.15)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (height width) (default: 224 224)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to inspect (default: all)",
    )
    parser.add_argument(
        "--inspect-batch",
        action="store_true",
        help="Show a sample batch from the dataset",
    )
    parser.add_argument(
        "--save-dataset-info",
        type=str,
        default=None,
        help="Save dataset info to JSON file",
    )

    args = parser.parse_args()

    # Validate target size
    target_size = tuple(args.target_size)

    # Create datasets
    logger.info("Creating datasets...")
    train_ds, val_ds, test_ds, classes = create_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_size=target_size,
        seed=args.seed,
    )

    logger.info("\n" + "=" * 70)
    logger.info("DATASET INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Classes: {list(classes.values())}")
    logger.info(f"Target image size: {target_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Random seed: {args.seed}")

    # Show sample batch if requested
    if args.inspect_batch:
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE BATCH INSPECTION")
        logger.info("=" * 70)

        splits = {"train": train_ds, "val": val_ds, "test": test_ds}

        splits_to_show = (
            [args.split] if args.split != "all" else ["train", "val", "test"]
        )

        for split_name in splits_to_show:
            dataset = splits[split_name]
            logger.info(f"\n{split_name.upper()} SPLIT:")

            for images, labels in dataset.take(1):
                logger.info(f"  Batch shape: {images.shape}")
                logger.info(f"  Image dtype: {images.dtype}")
                logger.info(
                    f"  Image value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]"
                )
                logger.info(f"  Labels: {labels.numpy()}")
                logger.info(
                    f"  Label classes: {[classes[int(l)] for l in labels.numpy()[:5]]}..."
                )

    # Save dataset info if requested
    if args.save_dataset_info:
        import json

        # Get dataset statistics
        dataset_builder = SatelliteImageDataset(
            dataset_path=args.dataset_path,
            split="train",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            verbose=False,
        )

        class_weights = dataset_builder.get_class_weights()

        info = {
            "dataset_path": str(args.dataset_path),
            "classes": {str(k): v for k, v in classes.items()},
            "class_weights": {str(k): float(v) for k, v in class_weights.items()},
            "target_size": list(target_size),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "splits": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
        }

        with open(args.save_dataset_info, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"\n✓ Dataset info saved to: {args.save_dataset_info}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Datasets ready for use")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
