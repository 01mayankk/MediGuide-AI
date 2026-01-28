"""
preprocessing.py

This module contains all data preprocessing logic for the MediGuide AI project.

Responsibility:
- Transform raw healthcare data into model-ready features
- Apply medically informed cleaning rules
- Ensure consistency across training and inference

IMPORTANT:
- This file implements decisions made in `02_preprocessing.ipynb`
- No experimentation should happen here
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Constants (Single source of truth)
# ============================================================

# Target variable name
TARGET_COLUMN = "Outcome"
# Why constant?
# - Prevents typos
# - Ensures consistency across training & inference


# Columns where zero is not a valid medical measurement
INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "BMI"
]
# Defined once to avoid:
# - magic strings
# - inconsistent handling


# ============================================================
# Core preprocessing functions
# ============================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from disk.

    Parameters
    ----------
    file_path : str
        Path to raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw healthcare dataframe.

    Why this function exists:
    - Centralizes data loading
    - Makes scripts and tests reusable
    """
    return pd.read_csv(file_path)


def create_working_copy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a defensive copy of the dataset.

    Why this exists:
    - Prevents mutation of raw data
    - Ensures reproducibility
    - Avoids subtle side-effects
    """
    return df.copy()


def handle_invalid_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace medically invalid zero values with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Healthcare dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with invalid zeros replaced.

    Medical reasoning:
    - Zero glucose/BP/BMI is biologically impossible
    - Zero typically represents missing measurement
    """
    df[INVALID_ZERO_COLUMNS] = df[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median (robust strategy).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values.

    Why median?
    - Healthcare data is often skewed
    - Median is robust to extreme values
    - Safer for risk screening
    """
    for column in INVALID_ZERO_COLUMNS:
        median_value = df[column].median()

        # Explicit reassignment avoids chained-assignment bugs
        df[column] = df[column].fillna(median_value)

    return df


def split_features_and_target(df: pd.DataFrame):
    """
    Separate features and target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Why explicit separation?
    - Prevents accidental leakage
    - Required for clean ML pipelines
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split data into training and test sets.

    Why this exists:
    - Prevents data leakage
    - Ensures reproducibility via random_state
    - Stratification preserves class balance
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_numerical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Scale numerical features using StandardScaler.

    VERY IMPORTANT:
    - Scaler is fit ONLY on training data
    - Test data is transformed using the same scaler

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_test_scaled[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# End-to-end preprocessing pipeline
# ============================================================

def preprocess_training_data(file_path: str):
    """
    Full preprocessing pipeline for TRAINING data.

    This is the function that training code will call.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    raw_df = load_raw_data(file_path)
    df = create_working_copy(raw_df)

    df = handle_invalid_zero_values(df)
    df = impute_missing_values(df)

    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_numerical_features(
        X_train, X_test
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
