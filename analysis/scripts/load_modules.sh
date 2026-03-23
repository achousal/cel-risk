#!/bin/bash
#
# Module Loading Script for cel-risk on HPC
#
# Purpose: Load required software modules for running the pipeline on HPC.
#          This script is HPC-specific and should be customized for your cluster.
#
# Usage:
#   source scripts/load_modules.sh
#
# Note: Use 'source' instead of 'bash' to load modules into current shell
#
# Author: Andres Chousal
# Date: 2026-01-19

echo "Loading modules for cel-risk pipeline..."

# ==============================================================================
# CUSTOMIZE THESE FOR YOUR HPC ENVIRONMENT
# ==============================================================================

# Python module (3.10+)
# Examples:
#   Stanford Sherlock: python/3.10.0
#   NIH Biowulf: python/3.10
#   Generic: python3/3.10.0
module load python/3.10.0

# R module (4.0+ for visualization scripts)
# Only needed if running compare_models_faith.R
# Examples:
#   Stanford Sherlock: r/4.2.0
#   NIH Biowulf: R/4.2.0
# Uncomment if needed:
# module load r/4.2.0

# Git module (usually not needed, already available)
# Uncomment if git is not in PATH:
# module load git

# Optional: GCC compiler (for some Python packages with C extensions)
# Usually not needed if using pre-built wheels
# Uncomment if you encounter compilation errors during pip install:
# module load gcc/10.2.0

# Optional: CUDA (for XGBoost GPU support)
# Only needed if using GPU-enabled XGBoost
# Uncomment if available and desired:
# module load cuda/11.7

# ==============================================================================
# VERIFICATION
# ==============================================================================

echo ""
echo "Loaded modules:"
module list 2>&1 | grep -E "python|r/|gcc|cuda" || echo "  (no relevant modules shown)"

echo ""
echo "Software versions:"
echo "  Python: $(python3 --version 2>&1 | awk '{print $2}')"
echo "  pip: $(pip3 --version 2>&1 | awk '{print $2}')"

if command -v Rscript &> /dev/null; then
    echo "  R: $(Rscript --version 2>&1 | grep -oP 'version \K[0-9.]+')"
else
    echo "  R: not loaded (OK if not running visualization)"
fi

if command -v git &> /dev/null; then
    echo "  Git: $(git --version 2>&1 | awk '{print $3}')"
fi

echo ""
echo "Environment ready for cel-risk pipeline"
echo ""
echo "Next steps:"
echo "  1. Run setup script (first time only):"
echo "     bash scripts/hpc_setup.sh"
echo ""
echo "  2. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  3. Submit jobs:"
echo "     bsub < CeD_hpc.lsf"
