# Experiments Documentation

This file documents the various experiments and their results for the ML Kaggle competition.

---

## Data Exploration & Feature Engineering

### Label Encoding
The `change_type` is the target variable, encoded as follows:
- **'Demolition'**: 0
- **'Road'**: 1
- **'Residential'**: 2
- **'Commercial'**: 3
- **'Industrial'**: 4
- **'Mega Projects'**: 5

### Feature Engineering Plan

**Categorical Features (One-Hot Encoding):**
- **urban_type**: 'Sparse Urban', 'Dense Urban', 'Rural', 'Industrial', 'Urban Slum', 'N/A'
- **geography_type**: 'Coastal', 'Dense Forest', 'Grass Land', 'Sparse Forest', 'Farms', 'Barren Land', 'Desert', 'River', 'Lakes', 'Snow', 'Hills'

**Status Encoding:**
- 'No Change': 0
- 'Demolished': 1
- 'Modified': 2
- 'New Construction': 3

**Engineered Features:**
- Count of unusual urban/geography types (handling skewed distributions)
- Interaction features between urban_type and geography_type
- Days between status changes
- Polygon features:
  - Vertices values
  - log(area) - due to skewed distribution
  - log(perimeter) - due to skewed distribution
  - Number of vertices
  - Centroid coordinates (x, y)

**Ideas to Try:**
- Quantile/rank transform for area and perimeter
- Multilayer Perceptron Classifier
- Random Forest Classifier
- Ensure dates are in chronological order

---
## First Experiment
Idea : 
HistGradientBoosting regression with 5-fold optuna parameter tuning and test on a validation set.
- Features:
    - one-hot encoding of urban_type and geography_type
    - encoded change_type as y
    - polygon features: log(area), log(perimeter), number of vertices, centroid x, centroid y
    - date features: number of days between changes
    - standard scaling of std and mean date values
- Model:
    - HistGradientBoosting Regressor with optuna tuning for learning_rate, max_iter, class_weight=‘balanced’

Results: 
    Best trial:
    F1 Score: 0.7560797002256303 ()
    Params: 
        learning_rate: 0.09816744981841224
        max_depth: 16
        l2_regularization: 0.6674834750906257
        class_weight: None
    Validation F1 Score: 0.7451502566689701 (weighted)
    Validation F1 Score (non-weighted): [0.7682059  0.71231829 0.79425646 0.67972665 0.09032258 0.        ]
Analysis:
- The minority classes are not well predicted, so the next improvements will be on the bad classes as the private board might reward better performance on the minority classes.
- An approach is to try predicting the minority classes with a different model, or to use SMOTE to balance the classes.
---

## Experiment 2: SMOTE for Class Balancing

**Approach:**
- Try SMOTE to synthetically resample minority classes
- Fix NaN values with KNNImputer
- weighted parameter in XGBoost wasn't effective enough

**Data Preprocessing:**
- KNNImputer for handling missing image data and status information

**Results:**

Best Trial:
- **F1 Score**: 0.906
- **Learning Rate**: 0.491
- **Max Depth**: 13
- **L2 Regularization**: 0.652
- **Class Weight**: None

Validation Scores:
- **Weighted F1**: 0.893
- **Per-Class F1**: [0.926, 0.946, 0.779, 0.725, 0.982, 0.999]
- **Public Leaderboard**: 0.729

**Analysis:**
- ⚠️ **Major Issue**: Applied SMOTE to validation/test data (incorrect approach)
- Minority class scores suspiciously high → overfitting to synthetic data
- Validation should maintain real-world distribution
- Public leaderboard score (0.729) much lower than validation (0.893) confirms overfitting
- Next steps: ADASYN or Borderline SMOTE, apply only to training data
---

## Experiment 3: ADASYN with "all" Parameter

**Approach:**
- Applied ADASYN with sampling_strategy="all"
- Only applied to training data (corrected from Experiment 2)

**Results:**
- ❌ No improvement over baseline
- Not worth detailed analysis
---

## Experiment 4: BOVA (Boosted One-vs-All)

**Approach:**
- Train two classifiers per class:
  - **Unbalanced**: trained on all data
  - **Balanced**: trained on undersampled balanced data
- Prediction logic:
  - If unbalanced model is confident → use its prediction
  - Else if balanced model is not confident → use balanced prediction
  - Otherwise → average probabilities

**Results:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.71      | 0.84   | **0.77** | 3,151  |
| Road           | 0.64      | 0.74   | **0.69** | 1,431  |
| Residential    | 0.80      | 0.76   | **0.78** | 14,844 |
| Commercial     | 0.67      | 0.67   | **0.67** | 10,042 |
| Industrial     | 0.15      | 0.14   | **0.15** | 132    |
| Mega Projects  | 0.03      | 0.20   | **0.05** | 15     |

**Overall:**
- **Accuracy**: 0.73
- **Macro F1**: 0.52
- **Weighted F1**: 0.73

**Analysis:**
- Majority classes perform well (0.67-0.78 F1)
- Minority classes still struggle significantly
---

## Experiment 5: AutoML with LGBM

**Approach:**
- Automated model selection and hyperparameter tuning

**Model Selected:**
- **LGBMClassifier**
- colsample_bytree: 0.959
- learning_rate: 0.069
- max_bin: 255
- n_estimators: 1104
- num_leaves: 45
- reg_alpha: 0.090
- reg_lambda: 0.034

**Results:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.74      | 0.82   | **0.78** | 3,151  |
| Road           | 0.83      | 0.69   | **0.75** | 1,431  |
| Residential    | 0.80      | 0.81   | **0.80** | 14,844 |
| Commercial     | 0.71      | 0.69   | **0.70** | 10,042 |
| Industrial     | 0.38      | 0.05   | **0.08** | 132    |
| Mega Projects  | 0.00      | 0.00   | **0.00** | 15     |

**Overall:**
- **Accuracy**: 0.76
- **Macro F1**: 0.52
- **Weighted F1**: 0.76
- **Public Leaderboard**: 0.909

**Analysis:**
- Good performance on majority classes
- Still poor on minorities (Industrial, Mega Projects)
---

## Experiment 6: Binary Models with Minority Feature

**Approach:**
- Added engineered feature indicating minority classes (4 and 5)
- Train separate binary models

**Results:**

Validation Scores:
- **Weighted F1**: 0.742
- **Per-Class F1**: [0.777, 0.728, 0.789, 0.702, 0.956, 0.500]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.70      | 0.87   | **0.78** | 3,151  |
| Road           | 0.67      | 0.79   | **0.73** | 1,431  |
| Residential    | 0.84      | 0.75   | **0.79** | 14,844 |
| Commercial     | 0.68      | 0.72   | **0.70** | 10,042 |
| Industrial     | 0.93      | 0.98   | **0.96** | 132    |
| Mega Projects  | 0.67      | 0.40   | **0.50** | 15     |

**Overall:**
- **Accuracy**: 0.75
- **Macro F1**: 0.74
- **Weighted F1**: 0.76

**Analysis:**
- ✅ Huge improvement on minority classes (Industrial: 0.96, Mega: 0.50)
- ⚠️ Slight degradation on majority classes
- Trade-off: better balance but lower overall weighted score

**Next Ideas:**
- Multiple models per frequency tier (low/medium/high)
- Ensemble: RF + LGBM + HistGradient
- Optimize ensemble weights on validation split

---

## Experiment 7: OneVsRest RandomForest

**Approach:**
- Scikit-learn's OneVsRestClassifier with RandomForestClassifier
- Includes minority class indicator feature
- Optuna hyperparameter tuning

**Best Parameters:**
- n_estimators: 29
- max_depth: 19
- criterion: log_loss
- min_samples_split: 3
- min_samples_leaf: 7
- class_weight: balanced

**Results:**

Validation Scores:
- **Weighted F1**: 0.626
- **Per-Class F1**: [0.693, 0.652, 0.775, 0.678, 0.831, 0.125]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.63      | 0.77   | **0.69** | 3,151  |
| Road           | 0.65      | 0.65   | **0.65** | 1,431  |
| Residential    | 0.81      | 0.74   | **0.78** | 14,844 |
| Commercial     | 0.66      | 0.70   | **0.68** | 10,042 |
| Industrial     | 0.91      | 0.77   | **0.83** | 132    |
| Mega Projects  | 1.00      | 0.07   | **0.12** | 15     |

**Overall:**
- **Accuracy**: 0.73
- **Macro F1**: 0.63
- **Weighted F1**: 0.73

**Analysis:**
- Minority feature helps Industrial (0.83) but Mega Projects still struggles (0.12)
- Trade-off between precision and recall for minorities

---

## Experiment 8: BOVA + RandomForest + Cutoff Optimization 🏆

**Approach:**
- BOVA (Boosted One-vs-All) approach
- Added minority class feature (for classes 4 and 5)
- Optimized prediction thresholds per class

**Cutoff Optimization Progress:**
```
F1=0.622, Industrial=0.10, Mega=0.05
F1=0.633, Industrial=0.10, Mega=0.10
F1=0.648, Industrial=0.15, Mega=0.10
F1=0.659, Industrial=0.15, Mega=0.15
F1=0.673, Industrial=0.20, Mega=0.15
F1=0.676, Industrial=0.25, Mega=0.15
F1=0.681, Industrial=0.25, Mega=0.20
F1=0.683, Industrial=0.30, Mega=0.20
F1=0.685, Industrial=0.35, Mega=0.20
F1=0.687, Industrial=0.40, Mega=0.20
F1=0.687, Industrial=0.50, Mega=0.20 ← Best
```

**Optimized Cutoffs:** [0.5, 0.5, 0.5, 0.5, 0.5, 0.2]

**Results:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.71      | 0.84   | **0.77** | 3,151  |
| Road           | 0.64      | 0.75   | **0.69** | 1,431  |
| Residential    | 0.81      | 0.76   | **0.78** | 14,844 |
| Commercial     | 0.68      | 0.67   | **0.68** | 10,042 |
| Industrial     | 0.95      | 0.83   | **0.89** | 132    |
| Mega Projects  | 0.29      | 0.60   | **0.39** | 15     |

**Overall:**
- **Accuracy**: 0.74
- **Macro F1**: 0.70 🎯
- **Weighted F1**: 0.74

**Analysis:**
- ✅ **Best macro F1 so far** (0.70)
- ✅ Strong minority performance: Industrial (0.89), Mega (0.39)
- ✅ Cutoff optimization key to success
- Balanced performance across all classes

---

## Experiment 9: Simple Mean Ensemble

**Approach:**
- Ensemble multiple models by averaging predictions

**Results:**

Per-Class F1: [0.746, 0.673, 0.781, 0.666, 0.212, 0.032]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.67      | 0.84   | **0.75** | 3,151  |
| Road           | 0.58      | 0.80   | **0.67** | 1,431  |
| Residential    | 0.82      | 0.75   | **0.78** | 14,844 |
| Commercial     | 0.68      | 0.65   | **0.67** | 10,042 |
| Industrial     | 0.14      | 0.42   | **0.21** | 132    |
| Mega Projects  | 0.02      | 0.07   | **0.03** | 15     |

**Overall:**
- **Accuracy**: 0.73
- **Macro F1**: 0.52
- **Weighted F1**: 0.73

**Analysis:**
- ❌ Simple averaging doesn't improve over best single model
- Minority classes perform poorly
---

## Experiment 10: Stacked Ensemble with Logistic Regression

**Approach:**
- Stack predictions from multiple models
- Meta-learner: Logistic Regression

**Results:**

**Meta-Learner Performance:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.72      | 0.83   | **0.78** | 3,151  |
| Road           | 0.81      | 0.67   | **0.73** | 1,431  |
| Residential    | 0.81      | 0.80   | **0.80** | 14,844 |
| Commercial     | 0.70      | 0.71   | **0.71** | 10,042 |
| Industrial     | 0.00      | 0.00   | **0.00** | 132    |
| Mega Projects  | 0.00      | 0.00   | **0.00** | 15     |

- Macro F1: 0.50
- Weighted F1: 0.76

**Final Ensemble Performance:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.65      | 0.87   | **0.74** | 3,151  |
| Road           | 0.67      | 0.75   | **0.71** | 1,431  |
| Residential    | 0.84      | 0.73   | **0.78** | 14,844 |
| Commercial     | 0.67      | 0.70   | **0.68** | 10,042 |
| Industrial     | 0.18      | 0.39   | **0.25** | 132    |
| Mega Projects  | 0.00      | 0.00   | **0.00** | 15     |

**Overall:**
- **Accuracy**: 0.73
- **Macro F1**: 0.53
- **Weighted F1**: 0.74

**Analysis:**
- ❌ Stacking doesn't improve over single best model (Experiment 8)
- Meta-learner fails completely on minorities

---

## Experiment 11: AutoML FLAML with Train/Val Split

**Approach:**
- FLAML AutoML with proper train/validation split
- Then trained on all data (potential overfitting)

### Phase 1: With Train/Val Split

**Model Selected: LGBMClassifier**
- colsample_bytree: 0.970
- learning_rate: 0.220
- max_bin: 127
- min_child_samples: 18
- n_estimators: 143
- num_leaves: 29
- reg_alpha: 0.003
- reg_lambda: 0.361

**Results:**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.76      | 0.82   | **0.79** | 3,263  |
| Road           | 0.77      | 0.69   | **0.73** | 1,416  |
| Residential    | 0.79      | 0.81   | **0.80** | 14,774 |
| Commercial     | 0.70      | 0.67   | **0.68** | 10,020 |
| Industrial     | 0.94      | 0.98   | **0.96** | 129    |
| Mega Projects  | 0.71      | 0.38   | **0.50** | 13     |

- **Accuracy**: 0.76
- **Macro F1**: 0.74
- **Weighted F1**: 0.75

### Phase 2: Trained on All Data

**Model Selected: XGBClassifier**
- colsample_bylevel: 0.959
- learning_rate: 0.189
- max_leaves: 35
- min_child_weight: 1.132
- n_estimators: 154

**Results (Training Set):**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|--------|
| Demolition     | 0.76      | 0.85   | **0.80** | 31,509 |
| Road           | 0.84      | 0.76   | **0.80** | 14,305 |
| Residential    | 0.81      | 0.82   | **0.81** | 148,435|
| Commercial     | 0.74      | 0.70   | **0.72** | 100,422|
| Industrial     | 1.00      | 1.00   | **1.00** | 1,324  |
| Mega Projects  | 1.00      | 1.00   | **1.00** | 151    |

- **Accuracy**: 0.78
- **Macro F1**: 0.86
- **Weighted F1**: 0.78

**Analysis:**
- ⚠️ Perfect scores on minorities (1.00) → likely overfitting
- Phase 1 results more realistic and generalizable