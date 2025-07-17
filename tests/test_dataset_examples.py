#!/usr/bin/env python
"""
Test script to ensure that class names match filenames in the dataset examples.
"""

import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import zarr

# Add the script directory to the path to import generate_dataset_examples
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestDatasetExamples(unittest.TestCase):
    """
    Test class to verify the correct relationship between class names and file names
    in the dataset_examples documentation.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.docs_dir = os.path.join(cls.temp_dir, "docs")
        os.makedirs(cls.docs_dir, exist_ok=True)

        # Copy the original script output path to a backup
        cls.original_docs_dir = Path("docs/dataset_examples")
        if cls.original_docs_dir.exists():
            cls.backup_dir = Path(tempfile.mkdtemp())
            shutil.copytree(cls.original_docs_dir, cls.backup_dir / "dataset_examples")
            print(f"Backed up original docs to {cls.backup_dir}")
        else:
            cls.backup_dir = None

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        shutil.rmtree(cls.temp_dir)

        # Restore the original docs if they were backed up
        if cls.backup_dir and cls.original_docs_dir.exists():
            shutil.rmtree(cls.original_docs_dir)
            shutil.copytree(cls.backup_dir / "dataset_examples", cls.original_docs_dir)
            shutil.rmtree(cls.backup_dir)
            print("Restored original docs")

    # TODO: Uncomment and fix the test below

    # @patch("zarr.open")
    # @patch("copick.from_czcdp_datasets")
    # def test_class_name_to_filename_consistency(self, mock_from_czcdp, mock_zarr_open):
    #     """
    #     Test that each class name in the markdown file correctly corresponds to a matching
    #     filename in the directory.
    #
    #     The test patches the output directory and then run the generate_docs function
    #     with mocked dependencies to avoid actual copick operations.
    #     """
    #     # Set up mock for copick.from_czcdp_datasets
    #     mock_copick_root = MagicMock()
    #     mock_run = MagicMock()
    #     mock_vs = MagicMock()
    #     mock_tomogram = MagicMock()
    #
    #     # Configure mocks
    #     mock_from_czcdp.return_value = mock_copick_root
    #     mock_copick_root.runs = [mock_run]
    #     mock_copick_root.pickable_objects = [
    #         MagicMock(name="ferritin-complex"),
    #         MagicMock(name="virus-like-capsid"),
    #         MagicMock(name="cytosolic-ribosome"),
    #         MagicMock(name="membrane"),
    #         MagicMock(name="beta-galactosidase"),
    #         MagicMock(name="beta-amylase"),
    #         MagicMock(name="thyroglobulin"),
    #     ]
    #     for po in mock_copick_root.pickable_objects:
    #         po.name = po.name
    #
    #     mock_run.name = "test_run"
    #     mock_run.get_voxel_spacing.return_value = mock_vs
    #     mock_vs.tomograms = [mock_tomogram]
    #     mock_tomogram.tomo_type = "wbp-denoised"
    #
    #     # Mock picks for each object type
    #     mock_picks_list = []
    #     for po in mock_copick_root.pickable_objects:
    #         mock_pick = MagicMock()
    #         mock_pick.from_tool = True
    #         mock_pick.pickable_object_name = po.name
    #         mock_pick.numpy.return_value = (np.array([[100, 100, 100]]), None)
    #         mock_picks_list.append(mock_pick)
    #
    #     mock_run.get_picks.return_value = mock_picks_list
    #
    #     # Mock zarr.open to return a dummy array
    #     mock_zarr_root = MagicMock()
    #     mock_zarr_open.return_value = mock_zarr_root
    #     mock_zarr_root.__getitem__.return_value = np.random.randn(100, 100, 100)
    #     mock_tomogram.zarr.return_value = "dummy_zarr_path"
    #
    #     # We'll patch the output directory and then run the generate_docs function
    #     original_output_dir = Path("docs/dataset_examples")
    #     temp_output_dir = Path(self.docs_dir) / "dataset_examples"
    #
    #     # Monkey patch the Path class to redirect the output
    #     original_init = Path.__init__
    #
    #     def patched_init(self_path, *args, **kwargs):
    #         # Replace the output directory path with our temp directory
    #         arg_str = str(args[0]) if args else ""
    #         if arg_str == str(original_output_dir):
    #             original_init(self_path, temp_output_dir, **kwargs)
    #         else:
    #             original_init(self_path, *args, **kwargs)
    #
    #     # Apply monkey patch
    #     Path.__init__ = patched_init
    #
    #     try:
    #         # Run the document generation script
    #         from generate_dataset_examples import main as generate_docs
    #
    #         generate_docs()
    #
    #         # Verify the markdown file exists
    #         md_file = temp_output_dir / "README.md"
    #         self.assertTrue(md_file.exists(), f"README.md file not found at {md_file}")
    #
    #         # Read the markdown file
    #         with open(md_file, "r") as f:
    #             content = f.read()
    #
    #         # Extract all class names and filenames
    #         class_pattern = re.compile(r"## Class: ([^\n]+)")
    #         image_pattern = re.compile(r"!\[(.*?)\]\(\./([^)]+)\)")
    #
    #         class_names = class_pattern.findall(content)
    #         image_refs = image_pattern.findall(content)
    #
    #         # Check that all class sections have a corresponding image
    #         self.assertEqual(
    #             len(class_names),
    #             len(image_refs),
    #             f"Number of class sections ({len(class_names)}) does not match number of images ({len(image_refs)})",
    #         )
    #
    #         # Check the directory for actual image files
    #         image_files = [f for f in os.listdir(temp_output_dir) if f.endswith(".png")]
    #
    #         # Check that all referenced images exist in the directory
    #         for _, filename in image_refs:
    #             self.assertIn(filename, image_files, f"Referenced image file {filename} not found in the directory")
    #
    #         # Verify that all class names match their corresponding images
    #         for i, class_name in enumerate(class_names):
    #             # Get the image reference for this class
    #             image_alt, image_file = image_refs[i]
    #
    #             # The alt text should match the class name
    #             self.assertEqual(
    #                 class_name,
    #                 image_alt,
    #                 f"Alt text '{image_alt}' does not match class name '{class_name}'",
    #             )
    #
    #             # Generate the expected filename from the class name
    #             expected_filename = f"{class_name.lower().replace(' ', '_').replace('-', '_')}.png"
    #
    #             # Check that the actual filename matches the expected filename
    #             self.assertEqual(
    #                 expected_filename,
    #                 image_file,
    #                 f"Image filename '{image_file}' does not match expected filename '{expected_filename}' for class '{class_name}'",
    #             )
    #
    #             # Verify the file exists with the expected name
    #             self.assertTrue((temp_output_dir / image_file).exists(), f"Image file {image_file} does not exist")
    #
    #         # Check that we have examples for all the expected pickable objects
    #         expected_class_names = [po.name for po in mock_copick_root.pickable_objects] + ["background"]
    #         for expected_class in expected_class_names:
    #             self.assertIn(expected_class, class_names, f"Missing example for expected class '{expected_class}'")
    #
    #     finally:
    #         # Restore the original Path.__init__
    #         Path.__init__ = original_init


if __name__ == "__main__":
    unittest.main()
