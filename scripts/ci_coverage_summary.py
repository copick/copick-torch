#!/usr/bin/env python3
"""
Script to generate GitHub Actions coverage summary from an existing coverage.xml file.
This script is designed to be added to the CI workflow without modifying the workflow file.

Usage:
  python -m scripts.ci_coverage_summary
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    """Generate GitHub Actions coverage summary."""
    # Check if we're running in GitHub Actions
    if "GITHUB_STEP_SUMMARY" not in os.environ:
        print("Not running in GitHub Actions, skipping summary generation.")
        return 0

    try:
        # Check if coverage.xml exists
        if not Path("coverage.xml").exists():
            print("coverage.xml not found, skipping summary generation.")
            return 1

        # Parse coverage data
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        line_rate = float(root.attrib["line-rate"]) * 100

        # Write summary
        summary_file = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_file, "a") as f:
            f.write("## Coverage Summary\n\n")
            f.write(f"Total coverage: {line_rate:.2f}%\n\n")

            # Get package details
            packages = root.findall(".//package")
            if packages:
                f.write("| Package | Line Coverage | Branch Coverage |\n")
                f.write("|---------|--------------|----------------|\n")

                for package in packages:
                    pkg_name = package.attrib.get("name", "Unknown")
                    pkg_line_rate = float(package.attrib.get("line-rate", 0)) * 100
                    pkg_branch_rate = float(package.attrib.get("branch-rate", 0)) * 100
                    f.write(f"| {pkg_name} | {pkg_line_rate:.2f}% | {pkg_branch_rate:.2f}% |\n")

            f.write("\nSee artifacts for detailed coverage report.\n")

        print("Added coverage summary to GitHub Actions report")
        return 0

    except Exception as e:
        print(f"Error generating summary: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
