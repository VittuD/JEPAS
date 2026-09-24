"""Frozen-backbone dense-task (detection/segmentation) evaluation.

Wraps this repo's frozen JEPA encoders as torchvision detection backbones via
a ViTDet-style Simple Feature Pyramid, so token-fragmentation metrics
(experiments/token_metrics.py) can be checked against real downstream task
performance instead of only against each other. See experiments/dense_eval
scripts for the training/eval entry points.
"""
