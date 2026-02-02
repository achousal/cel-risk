#!/bin/bash
#
# HPC Setup Script for CeliacRisks v1.0.0
#
# Purpose: Automated setup of Python environment and package installation
#          Works for both LOCAL and HPC environments
#
# Usage:
#   bash scripts/hpc_setup.sh
#
# For detailed setup instructions, see: SETUP_README.md
#
# Requirements:
#   - Python 3.10+
#   - Git (for version tracking)
#   - ~2 GB disk space for virtual environment
#
# Note: For local development, you can also use conda (see SETUP_README.md)
#       This script creates a venv which is required for my_run_hpc.sh
#
# Author: Andres Chousal
# Date: 2026-01-20

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

# Logging functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Print header
echo "=========================================="
echo "  CeliacRisks HPC Setup (v1.0.0)"
echo "=========================================="
echo ""

# Detect project root (CeliacRisks/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_DIR="$PROJECT_ROOT/analysis"

# Change to analysis directory (where venv will be created)
cd "$ANALYSIS_DIR" || error "Could not find analysis directory at $ANALYSIS_DIR"

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found in $ANALYSIS_DIR"
fi

info "Project root: $PROJECT_ROOT"
info "Analysis directory: $ANALYSIS_DIR"
info "venv location: $ANALYSIS_DIR/venv/"

# Check Python version
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required, found $PYTHON_VERSION. Load a newer Python module."
fi

success "Python $PYTHON_VERSION detected"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    warning "Virtual environment 'venv' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Removing existing virtual environment..."
        rm -rf venv
    else
        info "Keeping existing virtual environment."
        info "Activating environment..."
        source venv/bin/activate
        info "Upgrading pip..."
        pip install --upgrade pip setuptools wheel
        info "Installing/updating package..."
        pip install -e .
        success "Package updated successfully"
        echo ""
        echo "=========================================="
        echo "  Setup Complete!"
        echo "=========================================="
        echo ""
        echo "To activate the environment:"
        echo "  source analysis/venv/bin/activate"
        echo ""
        echo "To verify installation:"
        echo "  ced --help"
        echo ""
        exit 0
    fi
fi

# Create virtual environment
info "Creating virtual environment..."
python3 -m venv venv
success "Virtual environment created"

# Activate virtual environment
info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
success "pip upgraded to version $(pip --version | awk '{print $2}')"

# Install package
info "Installing CeliacRisks package (this may take several minutes on HPC)..."
mkdir -p ../logs/setup
SETUP_LOG="../logs/setup/install_$(date +%Y%m%d_%H%M%S).log"
pip install -e . 2>&1 | tee "${SETUP_LOG}"

if [ $? -eq 0 ]; then
    success "Package installed successfully"
else
    error "Package installation failed. Check ${SETUP_LOG} for details."
fi

# Verify installation
info "Verifying installation..."
if command -v ced &> /dev/null; then
    success "ced command is available"
else
    error "ced command not found. Installation may have failed."
fi

# Check CLI functionality
info "Testing CLI..."
ced --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    success "CLI is functional"
else
    error "CLI test failed"
fi

# Create required directories
info "Creating output directories..."
mkdir -p ../logs
mkdir -p ../splits
mkdir -p ../results
success "Output directories created"

# Check for data file
info "Checking for input data file..."
if [ -f "../data/Celiac_dataset_proteomics_w_demo.parquet" ]; then
    DATA_SIZE=$(du -h ../data/Celiac_dataset_proteomics_w_demo.parquet | cut -f1)
    success "Data file found (size: $DATA_SIZE)"
else
    warning "Data file not found at ../data/Celiac_dataset_proteomics_w_demo.parquet"
    echo "  You need to copy it from shared storage before running the pipeline."
fi

# Record environment
info "Recording package versions..."
pip freeze > requirements_frozen_$(date +%Y%m%d).txt
success "Package versions saved to requirements_frozen_$(date +%Y%m%d).txt"

# Record git state
if command -v git &> /dev/null; then
    info "Recording git state..."
    git log -1 --oneline > git_version.txt 2>/dev/null || true
    if [ -f "git_version.txt" ]; then
        success "Git version recorded: $(cat git_version.txt)"
    fi
fi

# Run optional tests
echo ""
read -p "Run test suite to verify installation? (recommended, takes ~2 min) (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    info "Running test suite..."
    mkdir -p ../logs/setup
    TEST_LOG="../logs/setup/tests_$(date +%Y%m%d_%H%M%S).log"
    if pytest tests/ -v --tb=short > "${TEST_LOG}" 2>&1; then
        TEST_PASS=$(grep -c "passed" "${TEST_LOG}" || echo "0")
        success "Tests passed: $TEST_PASS"
    else
        warning "Some tests failed. Check ${TEST_LOG} for details."
        echo "  This may not prevent pipeline execution."
    fi
fi

# Print summary
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Environment details:"
echo "  Python: $PYTHON_VERSION"
echo "  Virtual env: venv/"
echo "  Working dir: $(pwd)"
echo ""
echo "Next steps:"
echo ""
echo "  Verify installation:"
echo "    ced --help"
echo ""
echo "Option A: Run full pipeline (recommended)"
echo "  Activate environment:"
echo "    source analysis/venv/bin/activate"
echo ""
echo "  Run pipeline:"
echo "    ced run-pipeline"
echo ""
echo ""
echo ""
echo "Documentation:"
echo "  - Setup guide: SETUP_README.md"
echo "  - Project overview: ../README.md"
echo "  - Architecture: docs/ARCHITECTURE.md"
echo ""
echo "=========================================="
