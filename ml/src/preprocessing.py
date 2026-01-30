"""
preprocessing.py

This module defines the COMPLETE preprocessing pipeline for MediGuide AI.

Its responsibility is to:
- Convert raw healthcare data into clean, model-ready features
- Apply medically justified data cleaning rules
- Produce deterministic, reproducible outputs for training & inference

IMPORTANT DESIGN PRINCIPLES:
- No experimentation is allowed in this file
- All decisions here are finalized in notebooks
- This file must behave identically every time it runs
"""

# ============================================================
# Imports
# ============================================================

# pandas & numpy are used for data manipulation
import pandas as pd
import numpy as np

# sklearn utilities for splitting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Constants (Single Source of Truth)
# ============================================================

# Name of the target column in the dataset
TARGET_COLUMN = "Outcome"
# Why a constant?
# - Prevents hard-coded strings across the codebase
# - Reduces typo-related bugs
# - Makes schema changes easier in the future


# Columns where a value of ZERO is medically invalid
INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "BMI",
]
# Medical reasoning:
# - Zero glucose, BP, or BMI is biologically impossible
# - In real datasets, zeros usually encode missing measurements
# - Treating them as real values would mislead the model


# ============================================================
# Core preprocessing helper functions
# ============================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw healthcare dataset from disk.

    Why this exists:
    - Centralizes file I/O logic
    - Makes testing and reuse easier
    - Keeps the pipeline modular
    """
    return pd.read_csv(file_path)


def create_working_copy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a defensive copy of the dataframe.

    Why this exists:
    - Prevents mutation of raw data
    - Ensures reproducibility
    - Avoids side-effects when chaining operations
    """
    return df.copy(deep=True)


def handle_invalid_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace medically invalid zero values with NaN.

    Why NaN?
    - Allows standard missing-value imputation
    - Makes intent explicit
    - Prevents treating zeros as real measurements
    """
    df[INVALID_ZERO_COLUMNS] = df[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using the median strategy.

    Why median (not mean)?
    - Healthcare data is often skewed
    - Median is robust to extreme outliers
    - Safer for risk prediction use-cases
    """
    for column in INVALID_ZERO_COLUMNS:
        median_value = df[column].median()

        # Explicit reassignment avoids chained-assignment bugs
        df[column] = df[column].fillna(median_value)

    return df


def split_features_and_target(df: pd.DataFrame):
    """
    Separate feature matrix (X) and target vector (y).

    Why explicit separation?
    - Prevents target leakage
    - Makes training code clearer
    - Enforces proper ML pipeline structure
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into training and test sets.

    Design choices:
    - test_size=0.2 → standard ML split
    - stratify=y → preserves outcome class balance
    - random_state fixed → reproducible results
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_numerical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Scale numerical features using StandardScaler.

    CRITICAL RULES:
    - Scaler is fit ONLY on training data
    - Test data is transformed using the same scaler
    - Column names and order are preserved

    Why preserve DataFrame?
    - Scikit-learn tracks feature names internally
    - Inference will fail if schema does not match
    """

    scaler = StandardScaler()

    # Fit scaler on training data ONLY
    # (never leak information from test data)
    scaler.fit(X_train)

    # Transform training data and keep schema
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    # Transform test data using training-time feature order
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# End-to-end preprocessing pipeline
# ============================================================

def preprocess_training_data(file_path: str):
    """
    Execute the FULL preprocessing pipeline for TRAINING.

    This is the ONLY public API of this module.
    Training code must call ONLY this function.

    Returns
    -------
    X_train : pd.DataFrame
        Scaled training features
    X_test : pd.DataFrame
        Scaled test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    scaler : StandardScaler
        Fitted scaler for inference reuse
    """

    # Step 1: Load and isolate raw data
    raw_df = load_raw_data(file_path)
    df = create_working_copy(raw_df)

    # Step 2: Clean invalid medical values
    df = handle_invalid_zero_values(df)

    # Step 3: Impute missing values safely
    df = impute_missing_values(df)

    # Step 4: Separate features and target
    X, y = split_features_and_target(df)

    # Step 5: Split into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 6: Scale numerical features
    X_train_scaled, X_test_scaled, scaler = scale_numerical_features(
        X_train, X_test
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
