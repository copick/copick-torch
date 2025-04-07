#!/bin/bash
#
# Post-test steps for CI
# This script can be added as a separate step in the CI workflow
# without modifying the workflow file structure
#

set -e

# Generate GitHub Actions summary
echo "Generating GitHub Actions summary..."
python -m scripts.ci_coverage_summary

# Generate coverage badge
echo "Generating coverage badge..."
python -m scripts.coverage_report --skip-tests

# If we're in GitHub Actions, display info about where to find reports
if [ -n "$GITHUB_ACTIONS" ]; then
  echo "Coverage reports generated:"
  echo "- coverage.xml (XML report)"
  echo "- htmlcov/ (HTML report)"
  echo "- coverage-badge.svg (Coverage badge)"
  echo "These can be uploaded as GitHub Actions artifacts."
fi

exit 0
