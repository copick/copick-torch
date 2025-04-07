# GitHub Actions Coverage Reporting

This document explains how to use the included coverage reporting tools with GitHub Actions.

## Overview

The repository includes scripts to generate code coverage reports and badges that work both locally and in CI environments. These tools provide a way to visualize code coverage without relying on external services like Codecov.

## How to Add to GitHub Actions Workflow

To add coverage reporting to the existing GitHub Actions workflow, simply add the following step after the test step. This doesn't require changing the existing workflow structure.

```yaml
- name: Generate Coverage Reports
  run: |
    bash scripts/ci_post_test.sh

- name: Upload Coverage Reports
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports-${{ matrix.python-version }}
    path: |
      htmlcov/
      coverage.xml
      coverage-badge.svg
```

## What This Provides

1. **GitHub Actions Summary** - A coverage summary will be added directly to the GitHub Actions run summary page
2. **Coverage Badge** - A self-contained SVG badge displaying the coverage percentage
3. **Report Artifacts** - HTML and XML reports available as downloadable artifacts

## Local Usage

You can also use these tools locally:

```bash
# Run tests and generate all reports with a badge
python -m scripts.coverage_report --term

# View HTML report
open htmlcov/index.html

# Use the badge in your documentation
# The file will be at: coverage-badge.svg
```

## Advantages

- No external service dependencies
- Self-contained reporting
- Works identically in CI and locally
- Provides visibility directly in GitHub Actions UI
- Doesn't require changes to existing workflow structure
