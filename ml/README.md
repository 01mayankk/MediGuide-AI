# MediGuide-AI: Machine Learning System

## Overview

MediGuide-AI is a production-grade machine learning system designed to predict health risk probabilities based on structured clinical data. Unlike rule-based classification systems that rely on hardcoded thresholds and boolean logic, this ML-powered approach learns complex, nonlinear patterns from historical patient data to estimate personalized risk scores.

The system is built for healthcare applications where nuanced risk assessment is critical. It provides probabilistic predictions rather than binary classifications, enabling clinicians to make informed decisions with appropriate context.

**Core Capabilities:**
- Probabilistic health risk prediction using ensemble learning
- Feature engineering and normalization pipeline
- Model artifact versioning and schema validation
- Deterministic inference with input verification
- Post-hoc explainability for clinical interpretability

---

## Why Machine Learning Over Rule-Based Logic

Traditional healthcare risk assessment tools often use static JSON configurations with fixed thresholds (e.g., "if BMI > 30 and age > 50, risk = high"). This approach has fundamental limitations:

1. **Inability to capture feature interactions**: Real health risk emerges from complex combinations of factors that simple rules cannot encode
2. **Lack of personalization**: Fixed thresholds ignore patient-specific contexts
3. **Poor generalization**: Rules optimized for one population may fail on another
4. **No learning from data**: Cannot improve as new patient outcomes become available

**Our ML Approach:**
- **Random Forest Classifier**: Chosen for its robustness, interpretability, and ability to handle mixed feature types without extensive preprocessing
- **Ensemble learning**: Reduces variance and overfitting compared to single decision trees
- **Non-parametric**: Makes minimal assumptions about data distribution
- **Feature importance**: Naturally provides clinical insights into risk drivers

---

## ML System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Raw Healthcare Data                      │
│                    (CSV / Database Export)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
│  • Feature validation & type checking                        │
│  • Missing value imputation                                  │
│  • Outlier detection                                         │
│  • Feature scaling (StandardScaler)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Train/Test Splitting                       │
│              (Stratified, reproducible seed)                 │
└──────────────┬────────────────────────┬─────────────────────┘
               │                        │
               ▼                        ▼
        Training Set              Validation Set
               │                        │
               ▼                        │
┌──────────────────────────────┐       │
│   Random Forest Training     │       │
│   • Hyperparameter tuning    │       │
│   • Cross-validation         │       │
│   • Class balancing          │       │
└──────────────┬───────────────┘       │
               │                        │
               ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Evaluation                          │
│   • Accuracy, Precision, Recall, F1                          │
│   • ROC-AUC, PR-AUC                                          │
│   • Confusion matrix analysis                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Artifact Serialization Layer                   │
│   • random_forest_model.pkl                                  │
│   • scaler.pkl                                               │
│   • feature_schema.json (schema locking)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Inference System                          │
│   • Schema validation (feature order, types)                 │
│   • Defensive input checks                                   │
│   • Probability calibration                                  │
│   • Batch/real-time prediction                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Verification & Testing                       │
│   • Unit tests for preprocessing logic                       │
│   • Integration tests for end-to-end flow                    │
│   • Artifact integrity checks                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure & File Responsibilities

### `notebooks/`
**Purpose**: Exploratory data analysis, prototyping, and documentation of modeling decisions.

| Notebook | Responsibility |
|----------|---------------|
| `01_eda.ipynb` | Data profiling, distribution analysis, correlation studies, missingness patterns |
| `02_preprocessing.ipynb` | Prototype feature engineering logic, test scaling strategies, visualize transformations |
| `03_model_training.ipynb` | Experiment with algorithms, hyperparameter search, cross-validation studies |
| `04_explainability.ipynb` | SHAP analysis, feature importance plots, clinical interpretation of model behavior |

**What belongs here:**
- Iterative experimentation
- Visualizations and statistical tests
- Markdown narratives explaining decisions
- Draft code that may later be refactored into `src/`

**What does NOT belong here:**
- Production inference logic
- Code intended to be called by APIs or batch jobs
- Mission-critical preprocessing that must be version-controlled as code

**Key principle**: Notebooks are for human understanding, not machine execution in production.

---

### `src/`
**Purpose**: Production-ready Python modules that implement the ML pipeline as reusable, testable code.

| Module | Responsibility |
|--------|---------------|
| `preprocessing.py` | Feature validation, cleaning, scaling. Must be deterministic and idempotent. |
| `train_model.py` | Model training orchestration: loads data, calls preprocessing, trains Random Forest, saves artifacts. |
| `inference.py` | Loads trained artifacts, validates input schema, generates predictions. Zero tolerance for schema drift. |
| `evaluate_model.py` | Computes performance metrics on holdout sets. Used for model validation before deployment. |
| `explain.py` | Production explainability: computes SHAP values or feature contributions for given inputs. |

**Design principles:**
- **Modularity**: Each file has a single, well-defined responsibility
- **Testability**: Functions should be pure where possible, with minimal side effects
- **Type safety**: Use type hints throughout
- **Error handling**: Defensive programming with informative exceptions
- **Logging**: Structured logs for debugging and audit trails

**What belongs here:**
- Class definitions (e.g., `HealthRiskModel`, `PreprocessingPipeline`)
- Utility functions used across multiple scripts
- Configuration loading and validation
- Data I/O abstractions

**What does NOT belong here:**
- Hardcoded file paths (use arguments or config files)
- Print statements (use logging)
- Experimental code that hasn't been validated

---

### `scripts/`
**Purpose**: Verification and validation scripts that ensure system integrity before deployment.

| Script | Responsibility |
|--------|---------------|
| `verify_preprocessing.py` | Confirms preprocessing logic produces expected output shapes, no data leakage, handles edge cases |
| `verify_training.py` | Checks model artifacts are serialized correctly, reproducibility of training runs |
| `verify_inference.py` | End-to-end test: loads model, runs sample predictions, validates output format and ranges |

**Why these exist:**
In healthcare ML, silent failures are catastrophic. Verification scripts act as pre-deployment smoke tests:
- Detect schema drift between training and inference
- Catch serialization issues across Python versions
- Validate that model predictions fall within expected probability ranges
- Ensure preprocessing transformations are reversible/auditable

**Usage pattern:**
```bash
# Run before every deployment
python -m ml.scripts.verify_preprocessing
python -m ml.scripts.verify_training
python -m ml.scripts.verify_inference
```

**Best practices:**
- Scripts should exit with non-zero status code on failure
- Output should be machine-readable (JSON) for CI/CD integration
- Include timing benchmarks to detect performance regressions

---

### `model_artifacts/`
**Purpose**: Persistent storage for trained models and preprocessing dependencies.

| Artifact | Contents |
|----------|----------|
| `random_forest_model.pkl` | Serialized scikit-learn RandomForestClassifier with fitted parameters |
| `scaler.pkl` | Fitted StandardScaler or MinMaxScaler (must match training preprocessing) |
| `feature_schema.json` | Locked feature names, types, and order. Enforces contract between training and inference. |

**Schema locking with `feature_schema.json`:**
```json
{
  "version": "1.0.0",
  "features": [
    {"name": "age", "type": "int", "index": 0},
    {"name": "bmi", "type": "float", "index": 1},
    {"name": "blood_pressure_systolic", "type": "int", "index": 2}
  ],
  "target": "high_risk"
}
```

**Why schema locking matters:**
- Prevents silent feature reordering bugs (e.g., swapping age and BMI)
- Detects missing features at inference time before prediction
- Documents feature engineering decisions for reproducibility
- Enables safe model versioning (reject inference if schema version mismatches)

**Versioning strategy:**
- Append timestamps to artifacts: `random_forest_model_2025_01_30.pkl`
- Maintain CHANGELOG.md documenting feature additions/removals
- Use semantic versioning for schema: major changes (breaking), minor (backward compatible)

---

## How to Run

### 1. Environment Setup
```bash
cd ml/
pip install -r requirements.txt
```

### 2. Training a New Model
```bash
# Train from scratch (reads raw data, saves artifacts)
python -m ml.src.train_model --data-path ../data/healthcare_data.csv --output-dir model_artifacts/

# With hyperparameter tuning
python -m ml.src.train_model --tune --n-trials 50
```

**Expected outputs:**
- Trained model saved to `model_artifacts/random_forest_model.pkl`
- Scaler saved to `model_artifacts/scaler.pkl`
- Training metrics logged to `logs/training_YYYYMMDD_HHMMSS.json`

### 3. Model Evaluation
```bash
# Evaluate on test set
python -m ml.src.evaluate_model --model-path model_artifacts/random_forest_model.pkl --test-data ../data/test_set.csv

# Generate classification report
python -m ml.src.evaluate_model --report --output metrics/report.json
```

### 4. Running Inference
```bash
# Single prediction
python -m ml.src.inference --input '{"age": 45, "bmi": 28.3, "blood_pressure_systolic": 140}' --model-dir model_artifacts/

# Batch predictions from CSV
python -m ml.src.inference --batch --input-file ../data/new_patients.csv --output-file predictions.csv
```

**Output format:**
```json
{
  "patient_id": "12345",
  "risk_probability": 0.73,
  "risk_class": "high",
  "confidence": 0.85,
  "model_version": "1.0.0"
}
```

### 5. Verification Scripts
```bash
# Run all verification checks
python -m ml.scripts.verify_preprocessing
python -m ml.scripts.verify_training
python -m ml.scripts.verify_inference

# Expected output: All checks passed ✓
```

### 6. Explainability Analysis
```bash
# Compute SHAP values for specific prediction
python -m ml.src.explain --input '{"age": 65, "bmi": 32.1}' --model-dir model_artifacts/ --output shap_plot.png

# Batch explainability report
python -m ml.src.explain --batch --input-file high_risk_patients.csv --output-dir explanations/
```

---

## Safety and Reliability in Healthcare ML

### 1. Schema Enforcement
**Problem**: Feature order changes between training and inference can cause silent prediction errors.

**Solution**: `feature_schema.json` acts as a contract.
```python
# In inference.py
def validate_input_schema(input_data, schema_path):
    schema = load_json(schema_path)
    expected_features = [f["name"] for f in schema["features"]]
    
    # Check missing features
    missing = set(expected_features) - set(input_data.keys())
    if missing:
        raise SchemaViolationError(f"Missing features: {missing}")
    
    # Check extra features (potential data leakage)
    extra = set(input_data.keys()) - set(expected_features)
    if extra:
        logger.warning(f"Ignoring unexpected features: {extra}")
    
    # Enforce order
    return [input_data[f] for f in expected_features]
```

### 2. Deterministic Inference
**Problem**: Non-deterministic predictions undermine trust and auditability.

**Solution**:
- Fix random seeds in Random Forest (`random_state=42`)
- Pin scikit-learn version in `requirements.txt`
- Log prediction timestamps and model versions

### 3. Defensive Input Validation
```python
# Example checks in preprocessing.py
def validate_numeric_range(df, column, min_val, max_val):
    """Ensures clinical values fall within physiologically plausible ranges."""
    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
    if not out_of_range.empty:
        raise ValueError(f"Implausible {column} values detected: {out_of_range}")
```

**Checks performed:**
- Age: 0-120 years
- BMI: 10-70 kg/m²
- Blood pressure: 50-250 mmHg (systolic)

### 4. Model Monitoring Hooks
While not implemented in this version, production systems should track:
- Prediction distribution drift (KL divergence from training set)
- Feature value drift (Kolmogorov-Smirnov test)
- Calibration decay (Brier score over time)

---

## Explainability and Clinical Interpretability

### Why Explainability Matters in Healthcare
Black-box predictions are insufficient for clinical decision support. Providers need to understand:
- Which features most influenced a high-risk prediction?
- Are model predictions consistent with clinical intuition?
- Can the model explain edge cases or controversial predictions?

### Approach: SHAP (SHapley Additive exPlanations)
SHAP values provide local feature attributions that satisfy desirable mathematical properties (local accuracy, missingness, consistency).

**Notebook vs. Production Explainability:**

| Context | Tool | Use Case |
|---------|------|----------|
| `04_explainability.ipynb` | TreeExplainer, force plots, summary plots | Research, debugging, model validation |
| `src/explain.py` | Lightweight SHAP computation | Real-time explanations for clinicians |

**Example output from `explain.py`:**
```json
{
  "prediction": 0.78,
  "top_risk_factors": [
    {"feature": "age", "contribution": +0.23, "value": 68},
    {"feature": "bmi", "contribution": +0.15, "value": 31.2},
    {"feature": "smoking_history", "contribution": +0.12, "value": 1}
  ],
  "protective_factors": [
    {"feature": "exercise_frequency", "contribution": -0.08, "value": 5}
  ]
}
```

---

## Future Extensions

### 1. Multi-Disease Prediction
Current system predicts a single "high risk" outcome. Extensions:
- Multi-label classification (diabetes, hypertension, CVD simultaneously)
- Hierarchical models (predict disease category, then specific subtype)
- Disease trajectory modeling (time-to-event prediction)

### 2. Continuous Learning Pipeline
- Implement online learning with periodic retraining on new patient data
- A/B testing framework for model versions
- Automated model evaluation and rollback on performance degradation

### 3. API Integration
```python
# FastAPI endpoint example
@app.post("/predict")
async def predict_risk(patient: PatientSchema):
    input_data = patient.dict()
    prediction = inference_pipeline.predict(input_data)
    return {"risk_score": prediction["probability"]}
```

### 4. Model Monitoring Dashboard
- Real-time dashboards tracking prediction volume, latency, drift metrics
- Alerting on anomalous predictions or input distributions
- Integration with clinical EHR systems for feedback loop

### 5. Federated Learning
For multi-institution deployments, train models on decentralized data without sharing raw patient records.

---

## Requirements

See `requirements.txt` for full dependency list. Key libraries:
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
shap>=0.42.0
joblib>=1.3.0
```

**Python version**: 3.9+

---

## Contributing

This ML system follows strict code quality standards:
- All code in `src/` must have corresponding unit tests
- Run `black` and `flake8` before committing
- Update `feature_schema.json` if adding/removing features
- Document hyperparameter choices in docstrings

---

## Contact

For questions about model architecture or deployment, contact the ML Engineering team.

**Model Version**: 1.0.0  
**Last Training Date**: 2025-01-30  
**Data Version**: v2.3