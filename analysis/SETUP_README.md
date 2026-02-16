# CeliacRisks Setup Guide

**Version**: 1.0.0
**Updated**: 2026-01-30
**Author**: Andres Chousal

This guide covers environment setup for both local development and HPC production runs.

---

## Prerequisites

- Python 3.10+
- Git (for version tracking)
- ~2 GB disk space for dependencies
- Input data file: `../data/Celiac_dataset_proteomics_w_demo.parquet`
- **macOS only**: OpenMP runtime (install via `brew install libomp`)

---

## Quick Start

### Local Development (Recommended)

If you already have a conda environment:

```bash
# 1. Ensure you're in the project root
cd CeliacRisks/

# 2. Install the package in your conda environment
pip install -e analysis/

conda activate ced_ml

# 3. Verify installation
ced --help

# 4. Run pipeline (auto-discovers data, trains default models)
ced run-pipeline
```

### HPC Production

For HPC job submission with LSF (`bsub`):

```bash
# 1. Run automated setup (creates venv, installs package)
cd CeliacRisks/
bash analysis/scripts/hpc_setup.sh

source analysis/venv/bin/activate  # Required if using venv

# 2. Edit HPC config with your HPC project allocation
nano analysis/configs/pipeline_hpc.yaml

# 3. Preview jobs (dry run)
ced run-pipeline --hpc --dry-run

# 4. Submit production run
ced run-pipeline --hpc
```

---

## Detailed Setup Instructions

### Option 1: Conda Environment (Local Development)

**Best for**: Local exploration, development, testing

```bash
# Create new conda environment
conda create -n ced_ml python=3.10
conda activate ced_ml

# Navigate to project
cd analysis/

# Install package
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"

# Verify installation
ced --help
ced --version

# Run pipeline
ced run-pipeline
```

**Advantages**:
- Easy to manage multiple environments
- Works seamlessly with Jupyter notebooks
- No need for separate venv setup

### Option 2: Virtual Environment (HPC and Local)

**Best for**: HPC production runs, reproducible deployments

**Automated setup**:

```bash
cd analysis/
bash scripts/hpc_setup.sh
```

This script will:
1. Check Python version (requires 3.10+)
2. Create virtual environment in `venv/`
3. Install package and dependencies
4. Run optional test suite
5. Create output directories
6. Record package versions and git state

**Virtual environment activation**:

The setup script **cannot** activate the venv in your shell (bash subprocess limitation).

- **For interactive CLI usage**: Manually activate after setup:
  ```bash
  source analysis/venv/bin/activate
  ced --help  # Verify installation
  ```

## Verification and Testing

### Quick verification

```bash
# Check CLI is available
ced --help

# Check version
ced --version

# List available commands
ced --help
```

### Run test suite

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=ced_ml --cov-report=term-missing

# Fast tests only (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

---

**Last Updated**: 2026-01-30
**Tested On**:
- macOS (local)
- HPC cluster with LSF scheduler
