#!/usr/bin/env python3
"""
Generate code coverage reports with a self-contained badge.
This script can be used both locally and in CI environments.
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run_tests_with_coverage(args):
    """Run tests with coverage reporting."""
    cmd = [
        "pytest",
        f"--cov={args.package}",
        "--cov-report=xml",
        "--cov-report=html",
    ]
    if args.term:
        cmd.append("--cov-report=term")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def generate_badge(coverage_rate, output_path):
    """Generate a simple SVG coverage badge."""
    # Determine color based on coverage rate
    if coverage_rate >= 90:
        color = "brightgreen"
    elif coverage_rate >= 80:
        color = "green"
    elif coverage_rate >= 70:
        color = "yellowgreen"
    elif coverage_rate >= 60:
        color = "yellow"
    elif coverage_rate >= 50:
        color = "orange"
    else:
        color = "red"

    # SVG template
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="104" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h61v20H0z"/>
        <path fill="#{color}" d="M61 0h43v20H61z"/>
        <path fill="url(#b)" d="M0 0h104v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="30.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="30.5" y="14">coverage</text>
        <text x="82.5" y="15" fill="#010101" fill-opacity=".3">{coverage_rate:.1f}%</text>
        <text x="82.5" y="14">{coverage_rate:.1f}%</text>
    </g>
</svg>"""

    with open(output_path, "w") as f:
        f.write(svg)

    print(f"Coverage badge saved to {output_path}")


def extract_coverage_from_xml():
    """Extract coverage data from coverage.xml file."""
    try:
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        line_rate = float(root.attrib["line-rate"]) * 100
        return line_rate
    except (FileNotFoundError, KeyError) as e:
        print(f"Error extracting coverage data: {e}")
        return 0.0


def print_github_actions_summary(coverage_rate, badge_path):
    """Print summary for GitHub Actions."""
    if "GITHUB_STEP_SUMMARY" in os.environ:
        summary_file = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_file, "a") as f:
            f.write("## Coverage Summary\n\n")
            f.write(f"![Coverage]({badge_path})\n\n")
            f.write(f"Total coverage: {coverage_rate:.2f}%\n\n")
            f.write("See artifacts for detailed coverage report.\n")
        print("Added coverage summary to GitHub Actions report")


def main():
    parser = argparse.ArgumentParser(description="Generate code coverage reports and badge")
    parser.add_argument(
        "--package",
        default="copick_torch",
        help="Package to measure coverage for (default: copick_torch)",
    )
    parser.add_argument(
        "--output",
        default="coverage-badge.svg",
        help="Output path for the coverage badge (default: coverage-badge.svg)",
    )
    parser.add_argument("--term", action="store_true", help="Also generate terminal coverage report")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests (use existing coverage data)")

    args = parser.parse_args()

    # Make sure the output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        # Run tests with coverage
        ret_code = run_tests_with_coverage(args)
        if ret_code != 0:
            print("Tests failed!")
            sys.exit(ret_code)

    # Extract coverage data from xml
    coverage_rate = extract_coverage_from_xml()

    # Generate badge
    generate_badge(coverage_rate, args.output)

    # If running in GitHub Actions, add to summary
    print_github_actions_summary(coverage_rate, args.output)

    print(f"Total coverage: {coverage_rate:.2f}%")
    print("HTML report available at: htmlcov/index.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
