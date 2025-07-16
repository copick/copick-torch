#!/usr/bin/env python
"""
Setup script to create a minimal test environment for copick-torch tests.

This script is optional and only needed if you want to generate mock data
for testing without requiring real data.
"""

import json
import os
from pathlib import Path

import numpy as np


def setup_test_env():
    """
    Create a minimal test environment for running tests.

    This includes:
    1. A mock config file
    2. A directory structure for mock data
    """
    # Create test directory
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)

    # Create overlay directory
    overlay_dir = test_dir / "overlay"
    overlay_dir.mkdir(exist_ok=True)

    # Create a mock config file
    config = {
        "config_type": "local",
        "name": "Test Data",
        "description": "Mock data for testing",
        "version": "1.0.0",
        "pickable_objects": [
            {
                "name": "test_object_1",
                "go_id": "GO:0000001",
                "is_particle": True,
                "label": 1,
                "color": [0, 117, 220, 128],
                "radius": 30,
            },
            {
                "name": "test_object_2",
                "go_id": "GO:0000002",
                "is_particle": True,
                "label": 2,
                "color": [153, 63, 0, 128],
                "radius": 40,
            },
        ],
        "overlay_root": str(overlay_dir),
        "overlay_fs_args": {"auto_mkdir": True},
    }

    config_path = test_dir / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created test environment at {test_dir}")
    print("Set environment variable to use in tests:")
    print(f"export COPICK_TEST_CONFIG={config_path}")


if __name__ == "__main__":
    setup_test_env()
