"""
E2E tests for HPC configuration validation.

Tests HPC-specific configuration handling:
- HPC config structure validation
- Pipeline config HPC-readiness checks

Run with: pytest tests/e2e/test_pipeline_hpc.py -v
"""

import yaml


class TestE2EHPCConfigValidation:
    """Test HPC configuration validation."""

    def test_hpc_config_validation(self, hpc_config):
        """
        Test: HPC config loads and validates correctly.

        Validates config structure without execution.
        """
        with open(hpc_config) as f:
            config = yaml.safe_load(f)

        # Check required HPC fields
        assert "hpc" in config
        assert "project" in config["hpc"]
        assert "queue" in config["hpc"]
        assert "cores" in config["hpc"]
        assert "memory" in config["hpc"]

        # Validate types
        assert isinstance(config["hpc"]["cores"], int)
        assert config["hpc"]["cores"] > 0

    def test_hpc_pipeline_config_structure(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Pipeline config has required sections for HPC execution.

        Validates that training configs are HPC-ready.
        """
        with open(minimal_training_config) as f:
            config = yaml.safe_load(f)

        # Verify config has all required sections
        assert "cv" in config
        assert "features" in config
        assert "calibration" in config

        # Verify models are specified
        assert "lr" in config or "LR" in str(config)
