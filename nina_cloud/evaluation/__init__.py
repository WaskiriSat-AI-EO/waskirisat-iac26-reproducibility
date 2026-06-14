"""Evaluation module for the nina_cloud cloud-detection pipeline.

Loads a trained MobileNetV2 checkpoint, runs inference over the test split of
``balanced_manifest.json``, and writes a structured ``evaluation_summary.json``
report with overall and per-degradation-profile (A/B/C) classification metrics.

Entry point::

    python -m nina_cloud.evaluation.pipelines.run_evaluate \\
        --dataset_dir data/cloud/04_dataset [--figures]
"""
