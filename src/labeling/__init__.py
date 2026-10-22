"""Labeling, category mapping, and noise simulation module for HelpDeskNet-50K.

Provides tools for mapping source queues and ticket tags to standardized target classes,
simulating realistic label noise (symmetric, class-dependent, and worker-dependent),
computing transition matrices, and maintaining strict separation between raw source labels,
reference labels, and observed corrupted labels.
"""

from __future__ import annotations

from src.labeling.map_categories import (
    QUEUE_TO_TARGET_MAP,
    TAG_BASED_OVERRIDES,
    TARGET_CLASSES,
    apply_label_mapping,
    get_mapping_report,
    save_mapping_documentation,
)
from src.labeling.simulate_noise import (
    DEFAULT_CONFUSION_PAIRS,
    NoiseSimulator,
    build_transition_matrix,
    validate_noise_simulation,
)

__all__ = [
    # Category mapping
    "TARGET_CLASSES",
    "QUEUE_TO_TARGET_MAP",
    "TAG_BASED_OVERRIDES",
    "apply_label_mapping",
    "get_mapping_report",
    "save_mapping_documentation",
    # Noise simulation
    "DEFAULT_CONFUSION_PAIRS",
    "NoiseSimulator",
    "build_transition_matrix",
    "validate_noise_simulation",
]
