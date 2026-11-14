"""Unit tests for Data Loading, Validation, Imputation, and Encoding.

Verifies:
1. Loading both UCI ('num' target) and Kaggle ('target' target) CSV formats.
2. Accurate collapsing of multi-class target (0-4) into binary (0/1).
3. Complete elimination of NaNs through median and mode clinical imputation.
4. Correct feature dimensions after one-hot encoding.
"""

import io
import pandas as pd
import pytest
from src.data_loader import clean_and_impute_data, encode_features


@pytest.fixture
def sample_uci_data() -> pd.DataFrame:
    """Simulate raw UCI dataset containing 'num' (0-4) and '?' missing values."""
    csv_data = """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,?,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,?,4
"""
    # Parse '?' as NaN
    return pd.read_csv(io.StringIO(csv_data), na_values="?")


@pytest.fixture
def sample_kaggle_data() -> pd.DataFrame:
    """Simulate Kaggle dataset format where target is already named 'target'."""
    csv_data = """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
55.0,1.0,2.0,130.0,240.0,0.0,0.0,160.0,0.0,0.8,1.0,0.0,3.0,0
62.0,0.0,4.0,140.0,268.0,0.0,2.0,160.0,0.0,3.6,3.0,2.0,3.0,1
"""
    return pd.read_csv(io.StringIO(csv_data))


def test_uci_target_collapsing(sample_uci_data):
    """Ensure original multi-class 0-4 num column is collapsed to binary 0/1."""
    cleaned = clean_and_impute_data(sample_uci_data)
    assert "target" in cleaned.columns
    assert "num" not in cleaned.columns
    # Check binary values
    assert set(cleaned["target"].unique()).issubset({0, 1})
    # Check exact collapsing mapping
    expected_targets = [0, 1, 1, 0, 1]  # from original [0, 2, 1, 0, 4]
    assert list(cleaned["target"]) == expected_targets


def test_kaggle_format_compatibility(sample_kaggle_data):
    """Ensure data loader cleanly handles datasets with pre-existing 'target' column."""
    cleaned = clean_and_impute_data(sample_kaggle_data)
    assert "target" in cleaned.columns
    assert len(cleaned) == 2
    assert set(cleaned["target"].unique()).issubset({0, 1})


def test_zero_nans_after_imputation(sample_uci_data):
    """Ensure all missing values ('ca' and 'thal') are imputed with zero remaining NaNs."""
    assert sample_uci_data["ca"].isna().sum() > 0
    assert sample_uci_data["thal"].isna().sum() > 0

    cleaned = clean_and_impute_data(sample_uci_data)
    total_nans = cleaned.isna().sum().sum()
    assert total_nans == 0, f"Expected 0 remaining NaNs, found {total_nans}"


def test_one_hot_encoding(sample_uci_data):
    """Ensure categorical variables are properly one-hot encoded without NaNs."""
    cleaned = clean_and_impute_data(sample_uci_data)
    encoded_df, feature_cols = encode_features(cleaned, drop_first=False)

    assert "target" in encoded_df.columns
    assert "target" not in feature_cols
    # Verify dummy columns were created
    assert any("cp_" in col for col in encoded_df.columns)
    assert any("thal_" in col for col in encoded_df.columns)
    assert encoded_df.isna().sum().sum() == 0
