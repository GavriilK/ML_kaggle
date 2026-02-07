# Purpose
This files serves as a documentation of the various experiments done

# Experiments
## Data exploration:
TODO after the exploration of data:
The change_type is the label to predict, so i will encode it as follows: 
  - 'Road':5, 'Demolition':4, 'Commercial':3, 'Residential':2, 'Industrial':1, 'Mega Projects' : 0
  
- the urban_type column needs to be one-hot encoded for these categories: 
    - 'Sparse Urban', 'Dense Urban', 'Rural', 'Industrial', 'Dense Urban', 'Urban Slum', 'N,A'
- do one hot encoding for data_train.geography_type :
  - 'Coastal', 'Dense Forest', 'Grass Land', 'Sparse Forest', 'Farms', 'Barren Land', 'Desert','River', 'Lakes', 'Snow', 'Hills'
- The dates are not well ordered, and also i should convert the statuses into numerical values:
     - 'No Change' : 0
     - 'Demolished' : 1
     - 'Modified' : 2
     - 'New Construction' : 3
- As the distribution of the urban type and geography is very skewed towards one class, it might be interesting to make features that count how many unusual types are present or about interactions between the two features.
  
- For the changes in date, could be interesting to have the number of days between the changes
- for the polygons:
  - value of each vertice
  - area (log because of distribution)
  - perimeter (log because of distribution)
  - number of vertices
  - centroid

-To try:
    - quantile / rank transform for area or perimeter
    - Multilayer Perceptron Classifier
    - Random Forest Classifier
    - make date in right order
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
## Experiment 2: SMOTE Resampling

### Idea
We already used the weighted parameter in xgboost but it didn't improve the score on the minority classes which have not enough data, so now we will try SMOTE which resamples the minority classes synthetically so it has almost the same count of data as the majority classes.
We could also try imputing our own weights and optimize to find the best weights

First we fix the NaN values with a knnimputer because we might need the data and it seems either image data is missing, either time / change status data is missing.

### Results
**Best Trial:**
- **F1 Score:** 0.9064262214406363
- **Params:**
  - learning_rate: 0.4909390231840453
  - max_depth: 13
  - l2_regularization: 0.6519286101082055
  - class_weight: None

**Validation Performance:**
- **Weighted F1:** 0.8929361045508605
- **Per-class F1:** [0.926, 0.946, 0.779, 0.725, 0.982, 0.999]
- **Public Leaderboard:** 0.72871

### Analysis
- The score we get are a bit too much on the minority classes and the majority classes have not gotten better results, we might be overfitting. We should probably try to use ADASYN (Adaptive Synthetic Sampling) a variant of SMOTE that focuses on hard to learn classes or we could try Borderline SMOTE.
- Uploading on the leaderboard we get a score of : 0.72871 which means the model is either overfitting or the public board gives a high weight to the class encoded as 3 (value of 0.72516 is close to public 0.72871)
- Also the problem is that im modifying the data distribution with smote on the val test while it shouldn't because the val/test should represent the real world distribution.
- I might have not used properly smote because i applied on everything so the final evaluation value i had was wrong
## Experiment 3: ADASYN Resampling

### Idea
I applied ADASYN and with the "all" parameter, and only on data used to train the model.

### Results
Not worth mentioning, not better

---

## Experiment 4: BOVA (Boosted One-vs-All)

### Idea
I tried using bova (boosted one vs all) the idea is to train two classifiers one on all the data and the other adapted for minority classes. Then if the unbalanced model is confident we follow its prediction, otherwise we follow the balanced model. If neither are confident we take the mean probability.

### Results

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition     | 0.71      | 0.84   | 0.77     | 3,151    |
| Road           | 0.64      | 0.74   | 0.69     | 1,431    |
| Residential    | 0.80      | 0.76   | 0.78     | 14,844   |
| Commercial     | 0.67      | 0.67   | 0.67     | 10,042   |
| Industrial     | 0.15      | 0.14   | 0.15     | 132      |
| Mega Projects  | 0.03      | 0.20   | 0.05     | 15       |
| **Accuracy**   |           |        | **0.73** | 29,615   |
| **Macro Avg**  | 0.50      | 0.56   | **0.52** | 29,615   |
| **Weighted Avg**| 0.74     | 0.73   | 0.73     | 29,615   |
## Experiment 5: AutoML (FLAML)

### Model
LGBMClassifier with FLAML-optimized parameters:
- colsample_bytree: 0.959
- learning_rate: 0.069
- max_bin: 255
- n_estimators: 1104
- num_leaves: 45
- reg_alpha: 0.090
- reg_lambda: 0.034

### Results

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.74      | 0.82   | 0.78     | 3,151    |
| Road (1)       | 0.83      | 0.69   | 0.75     | 1,431    |
| Residential (2)| 0.80      | 0.81   | 0.80     | 14,844   |
| Commercial (3) | 0.71      | 0.69   | 0.70     | 10,042   |
| Industrial (4) | 0.38      | 0.05   | 0.08     | 132      |
| Mega Projects (5)| 0.00    | 0.00   | 0.00     | 15       |
| **Accuracy**   |           |        | **0.76** | 29,615   |
| **Macro Avg**  | 0.57      | 0.51   | **0.52** | 29,615   |
| **Weighted Avg**| 0.76     | 0.76   | 0.76     | 29,615   |
| **Public Leaderboard** |  |        | **0.90937** |      |
## Experiment 6: Binary Models + Minority Feature

### Idea
Added a feature representing the minority class (4 and 5)

### Results
**Validation Performance:**
- **Weighted F1:** 0.742020229404521
- **Per-class F1:** [0.777, 0.728, 0.789, 0.702, 0.956, 0.500]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.70      | 0.87   | 0.78     | 3,151    |
| Road (1)       | 0.67      | 0.79   | 0.73     | 1,431    |
| Residential (2)| 0.84      | 0.75   | 0.79     | 14,844   |
| Commercial (3) | 0.68      | 0.72   | 0.70     | 10,042   |
| Industrial (4) | 0.93      | 0.98   | 0.96     | 132      |
| Mega Projects (5)| 0.67    | 0.40   | 0.50     | 15       |
| **Accuracy**   |           |        | **0.75** | 29,615   |
| **Macro Avg**  | 0.75      | 0.75   | **0.74** | 29,615   |
| **Weighted Avg**| 0.76     | 0.75   | 0.76     | 29,615   |

### Analysis
With a feature representing the minority class (4 and 5), the f1 score of these two minority classes is way higher but the overall score hasnt not improved a lot, the other f1 score have gone down.

**Idea after:** do multiple models : per medium high low freq, whole datasaet and then rf + lgbm + hist then on val split optimize param so biggestf1 score overall

## Experiment 7: OneVsRest + RandomForest

### Model
Scikit's OneVsRest with RandomForestClassifier

### Best Trial
- **F1 Score:** 0.6340419797020355
- **Params:**
  - n_estimators: 29
  - max_depth: 19
  - criterion: log_loss
  - min_samples_split: 3
  - min_samples_leaf: 7
  - class_weight: balanced

### Results
**Validation Performance:**
- **Weighted F1:** 0.6258216409686566
- **Per-class F1:** [0.693, 0.652, 0.775, 0.678, 0.831, 0.125]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.63      | 0.77   | 0.69     | 3,151    |
| Road (1)       | 0.65      | 0.65   | 0.65     | 1,431    |
| Residential (2)| 0.81      | 0.74   | 0.78     | 14,844   |
| Commercial (3) | 0.66      | 0.70   | 0.68     | 10,042   |
| Industrial (4) | 0.91      | 0.77   | 0.83     | 132      |
| Mega Projects (5)| 1.00    | 0.07   | 0.12     | 15       |
| **Accuracy**   |           |        | **0.73** | 29,615   |
| **Macro Avg**  | 0.78      | 0.62   | **0.63** | 29,615   |
| **Weighted Avg**| 0.73     | 0.73   | 0.73     | 29,615   |

### Analysis
This is just sickit's onevsrest with randomforestclassifier, we can see the feature indicating the minority classes make them bettter but worse than the other.

## Experiment 8: BOVA + RF + Cutoff Optimization ✅ BEST

### Idea
Bova with the new rf feature for minority classes 4 and 5

### Cutoff Optimization Search
Searching for optimal cutoffs...
- New best! F1=0.6224, Industrial=0.10, Mega=0.05
- New best! F1=0.6328, Industrial=0.10, Mega=0.10
- New best! F1=0.6477, Industrial=0.15, Mega=0.10
- New best! F1=0.6589, Industrial=0.15, Mega=0.15
- New best! F1=0.6733, Industrial=0.20, Mega=0.15
- New best! F1=0.6764, Industrial=0.25, Mega=0.15
- New best! F1=0.6805, Industrial=0.25, Mega=0.20
- New best! F1=0.6832, Industrial=0.30, Mega=0.20
- New best! F1=0.6849, Industrial=0.35, Mega=0.20
- New best! F1=0.6867, Industrial=0.40, Mega=0.20
- New best! F1=0.6873, Industrial=0.50, Mega=0.20

**Final best cutoffs:** [0.5, 0.5, 0.5, 0.5, 0.5, 0.2]
**Final best F1:** 0.6873

### Results

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition     | 0.71      | 0.84   | 0.77     | 3,151    |
| Road           | 0.64      | 0.75   | 0.69     | 1,431    |
| Residential    | 0.81      | 0.76   | 0.78     | 14,844   |
| Commercial     | 0.68      | 0.67   | 0.68     | 10,042   |
| Industrial     | 0.95      | 0.83   | **0.89** | 132      |
| Mega Projects  | 0.29      | 0.60   | **0.39** | 15       |
| **Accuracy**   |           |        | **0.74** | 29,615   |
| **Macro Avg**  | 0.68      | 0.74   | **0.70** | 29,615   |
| **Weighted Avg**| 0.74     | 0.74   | 0.74     | 29,615   |

## Experiment 9: Ensemble with Mean Averaging

### Idea
First ensemble with just means

### Results
**Per-class F1:** [0.746, 0.673, 0.781, 0.666, 0.212, 0.032]

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.67      | 0.84   | 0.75     | 3,151    |
| Road (1)       | 0.58      | 0.80   | 0.67     | 1,431    |
| Residential (2)| 0.82      | 0.75   | 0.78     | 14,844   |
| Commercial (3) | 0.68      | 0.65   | 0.67     | 10,042   |
| Industrial (4) | 0.14      | 0.42   | 0.21     | 132      |
| Mega Projects (5)| 0.02    | 0.07   | 0.03     | 15       |
| **Accuracy**   |           |        | **0.73** | 29,615   |
| **Macro Avg**  | 0.48      | 0.59   | **0.52** | 29,615   |
| **Weighted Avg**| 0.74     | 0.73   | 0.73     | 29,615   |
## Experiment 10: Ensemble with Logistic Regression Meta-Learner

### Idea
with mean / logistic regression at the end

### Results - Logistic Regression Stacking

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.72      | 0.83   | 0.78     | 3,151    |
| Road (1)       | 0.81      | 0.67   | 0.73     | 1,431    |
| Residential (2)| 0.81      | 0.80   | 0.80     | 14,844   |
| Commercial (3) | 0.70      | 0.71   | 0.71     | 10,042   |
| Industrial (4) | 0.00      | 0.00   | 0.00     | 132      |
| Mega Projects (5)| 0.00    | 0.00   | 0.00     | 15       |
| **Accuracy**   |           |        | **0.76** | 29,615   |
| **Macro Avg**  | 0.51      | 0.50   | **0.50** | 29,615   |
| **Weighted Avg**| 0.76     | 0.76   | 0.76     | 29,615   |

### Results - Ensemble Predictions

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.65      | 0.87   | 0.74     | 3,151    |
| Road (1)       | 0.67      | 0.75   | 0.71     | 1,431    |
| Residential (2)| 0.84      | 0.73   | 0.78     | 14,844   |
| Commercial (3) | 0.67      | 0.70   | 0.68     | 10,042   |
| Industrial (4) | 0.18      | 0.39   | 0.25     | 132      |
| Mega Projects (5)| 0.00    | 0.00   | 0.00     | 15       |
| **Accuracy**   |           |        | **0.73** | 29,615   |
| **Macro Avg**  | 0.50      | 0.57   | **0.53** | 29,615   |
| **Weighted Avg**| 0.75     | 0.73   | 0.74     | 29,615   |

## Experiment 11: FLAML with Train/Val Split

### Model - LGBM (with validation split)
LGBMClassifier with FLAML-optimized parameters:
- colsample_bytree: 0.970
- learning_rate: 0.220
- max_bin: 127
- min_child_samples: 18
- n_estimators: 143
- num_leaves: 29
- reg_alpha: 0.003
- reg_lambda: 0.361

### Results - With Validation Split

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.76      | 0.82   | 0.79     | 3,263    |
| Road (1)       | 0.77      | 0.69   | 0.73     | 1,416    |
| Residential (2)| 0.79      | 0.81   | 0.80     | 14,774   |
| Commercial (3) | 0.70      | 0.67   | 0.68     | 10,020   |
| Industrial (4) | 0.94      | 0.98   | 0.96     | 129      |
| Mega Projects (5)| 0.71    | 0.38   | 0.50     | 13       |
| **Accuracy**   |           |        | **0.76** | 29,615   |
| **Macro Avg**  | 0.78      | 0.73   | **0.74** | 29,615   |
| **Weighted Avg**| 0.75     | 0.76   | 0.75     | 29,615   |

### Model - XGBoost (on all data)
Now on all the data (probably overfitting but let's see the results)

XGBClassifier with FLAML-optimized parameters:
- colsample_bylevel: 0.959
- learning_rate: 0.189
- max_leaves: 35
- min_child_weight: 1.132
- n_estimators: 154

### Results - All Data (Overfitting Check)

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|----------|
| Demolition (0) | 0.76      | 0.85   | 0.80     | 31,509   |
| Road (1)       | 0.84      | 0.76   | 0.80     | 14,305   |
| Residential (2)| 0.81      | 0.82   | 0.81     | 148,435  |
| Commercial (3) | 0.74      | 0.70   | 0.72     | 100,422  |
| Industrial (4) | 1.00      | 1.00   | 1.00     | 1,324    |
| Mega Projects (5)| 1.00    | 1.00   | 1.00     | 151      |
| **Accuracy**   |           |        | **0.78** | 296,146  |
| **Macro Avg**  | 0.86      | 0.86   | **0.86** | 296,146  |
| **Weighted Avg**| 0.78     | 0.78   | 0.78     | 296,146  |