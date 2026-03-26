# Model Card: MediWatch XGBoost Readmission Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | `mediwatch_xgboost` |
| **Model type** | XGBClassifier inside a scikit-learn Pipeline |
| **Framework** | scikit-learn 1.8+, XGBoost 3.2+ |
| **Task** | Binary classification — 30-day hospital readmission |
| **Training data** | UCI Diabetes 130-Hospitals (1999–2008) |
| **Version management** | MLflow Model Registry with `@champion` alias |
| **License** | Project license (see repository root) |

## Intended Use

**Primary use:** Portfolio demonstration of a champion/challenger retraining pipeline with drift monitoring. The model predicts whether a diabetic patient will be readmitted to the hospital within 30 days of discharge.

**Intended users:** Reviewers evaluating the pipeline architecture, ML engineering practices, and code quality. This model is **not intended for clinical decision-making**.

**Out-of-scope uses:**
- Clinical triage or discharge planning
- Insurance risk scoring
- Any production healthcare decision support without independent validation, regulatory review, and bias auditing

## Training Data

- **Source:** [UCI Diabetes 130-Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records:** ~100,000 hospital encounters across 130 US hospitals
- **Time span:** 1999–2008, split into 5 annual windows (2004–2008)
- **Train/eval split:** 80/20 per window (~16,000 train / ~4,000 eval)
- **Sliding window:** Challenger trains on current + previous window (2-year sliding window)
- **Positive class rate:** ~12% (readmitted within 30 days)

## Features

| Group | Count | Description |
|-------|-------|-------------|
| Numeric | 8 | Hospital stay duration, procedure counts, medication counts, prior visit history |
| Categorical | 22 | Demographics, admission metadata, medication flags — ordinal-encoded |
| Diagnosis (ICD-9) | 12 | Three diagnosis fields binned into clinical categories, collapsed into binary flags |
| **Total** | **42** | |

All feature engineering (ICD-9 binning, missing value imputation, string casting, ordinal encoding) is encapsulated in custom scikit-learn transformers serialized inside the pipeline artifact. No separate preprocessing is required at inference time.

## Model Architecture

```
Pipeline
├── MissingValueReplacer    (? → NaN)
├── ICD9Binner              (ICD-9 codes → clinical categories)
├── CategoricalStringCaster (numeric IDs → strings)
├── ColumnDropper           (remove ID columns)
├── ColumnTransformer
│   ├── num: passthrough    (8 numeric features)
│   └── cat: OrdinalEncoder (34 categorical features)
└── XGBClassifier
    ├── n_estimators=200
    ├── max_depth=7
    ├── learning_rate=0.3
    ├── subsample=0.8
    ├── scale_pos_weight=15
    └── random_state=42
```

## Performance

Performance varies across windows due to concept drift and inherently weak signal in the prediction task. ROC-AUC ranges from 0.53–0.61, which is consistent with published literature on this dataset.

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| ROC-AUC | 0.53–0.61 | Weak signal is a dataset characteristic |
| F1 | 0.20–0.35 | Used as the promotion metric |
| Accuracy | 0.55–0.70 | Misleading due to class imbalance |
| Precision | 0.15–0.25 | Low due to aggressive recall tuning (scale_pos_weight=15) |
| Recall | 0.30–0.55 | Prioritized via class weight adjustment |

**Promotion threshold:** A challenger must exceed the champion's F1 by ≥ 1% to be promoted. This prevents promotion on noise.

## Ethical Considerations

### Fairness and bias

This model was trained on hospital encounter data that includes demographic features (race, gender, age). These features are used as inputs to the model, which means the model's predictions may reflect historical biases present in the healthcare system, including:

- **Racial disparities in care:** Hospital readmission rates are known to vary by race due to systemic factors (access to follow-up care, socioeconomic status, insurance coverage) rather than clinical need alone.
- **Age-related patterns:** Older patients may have higher readmission rates due to comorbidities, but age alone should not drive clinical decisions.
- **Gender differences:** Differences in healthcare utilization patterns may be reflected in the data.

### What has NOT been done

- **Subgroup performance analysis:** Model performance has not been disaggregated by race, gender, or age group. This is a critical gap — a model with acceptable aggregate performance may perform significantly worse for underrepresented groups.
- **Fairness metric evaluation:** No equalized odds, demographic parity, or calibration analysis has been performed.
- **Bias mitigation:** No techniques (reweighting, adversarial debiasing, post-processing calibration) have been applied.

### Recommendations for production use

If this model were to be adapted for clinical use (which is explicitly out of scope for this project):

1. Perform subgroup analysis across all demographic features
2. Evaluate fairness metrics (equalized odds, predictive parity)
3. Consider removing or carefully handling demographic features
4. Conduct an independent bias audit
5. Obtain IRB approval and regulatory review
6. Implement ongoing monitoring for performance degradation across subgroups

## Limitations

- **Weak predictive signal:** The underlying prediction task is inherently difficult. The model's performance ceiling is low regardless of architecture.
- **Historical data:** The dataset covers 1999–2008 and may not reflect current clinical practices, coding standards, or patient demographics.
- **No delayed label handling:** The pipeline evaluates the challenger immediately, assuming labels are available. In production, a labeling lag window would be required.
- **No rollback automation:** While artifact versioning supports manual rollback, automated rollback on live performance degradation is not implemented.
- **Single hospital system:** The dataset comes from 130 US hospitals but may not generalize to other healthcare systems or countries.

## Citation

```
Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S.,
Cios, K.J. and Clore, J.N., 2014. Impact of HbA1c measurement on
hospital readmission rates: analysis of 70,000 clinical database
patient records. BioMed research international, 2014.
```
