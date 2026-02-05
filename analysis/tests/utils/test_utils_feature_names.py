"""Tests for utils.feature_names module.

Coverage areas:
- Prefix stripping (num__ prefix)
- Suffix preservation (_resid)
- Plain names passed through unchanged
- Edge cases (empty string, double prefix)
"""

import pytest

from ced_ml.utils.feature_names import extract_protein_name


class TestExtractProteinName:
    """Tests for extract_protein_name()."""

    @pytest.mark.parametrize(
        "input_name, expected",
        [
            # With num__ prefix and _resid suffix
            ("num__APOE_resid", "APOE_resid"),
            ("num__IL6_resid", "IL6_resid"),
            # With num__ prefix, no suffix
            ("num__APOE", "APOE"),
            ("num__IL6", "IL6"),
            # No prefix, with _resid suffix
            ("APOE_resid", "APOE_resid"),
            ("IL6_resid", "IL6_resid"),
            # Plain name, no prefix or suffix
            ("APOE", "APOE"),
            ("IL6", "IL6"),
        ],
    )
    def test_standard_cases(self, input_name, expected):
        assert extract_protein_name(input_name) == expected

    def test_empty_string(self):
        assert extract_protein_name("") == ""

    def test_only_prefix(self):
        assert extract_protein_name("num__") == ""

    def test_double_prefix_strips_once(self):
        assert extract_protein_name("num__num__APOE") == "num__APOE"

    def test_partial_prefix_not_stripped(self):
        assert extract_protein_name("num_APOE") == "num_APOE"
        assert extract_protein_name("NUM__APOE") == "NUM__APOE"

    def test_prefix_case_sensitive(self):
        assert extract_protein_name("Num__APOE") == "Num__APOE"
