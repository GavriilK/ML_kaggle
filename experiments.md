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
## Second Experiment
We already used the weighted parameter in xgboost but it didn't improve the score on the minority classes which have not enough data, so now we will try SMOTE which resamples the minority classes synthetically so it has almost the same count of data as the majority classes.
We could also try imputing our own weights and optimize to find the best weights

First we fix the NaN values with a knnimputer because we might need the data and it seems either image data is missing, either time / change status data is missing. 

Results : Best trial:
  F1 Score: 0.9064262214406363
  Params: 
    learning_rate: 0.4909390231840453
    max_depth: 13
    l2_regularization: 0.6519286101082055
    class_weight: None
Training final model with best hyperparameters on full training data...
Validation F1 Score: 0.8929361045508605, 
 Validation F1 Score (non-weighted): [0.92594435 0.94601835 0.77859966 0.72516003 0.98246665 0.99942759]
- Analysis: 
  - The score we get are a bit too much on the minority classes and the majority classes have not gotten better results, we might be overfitting. We should probably try to use ADASYN (Adaptive Synthetic Sampling) a variant of SMOTE that focuses on hard to learn classes or we could try Borderline SMOTE.
  - Uploading on the leaderboard we get a score of : 0.72871 which means the model is either overfitting or the public board gives a high weight to the class encoded as 3 (value of 0.72516 is close to public 0.72871)
  Also the problem is that im modifying the data distribution with smote on the val test while it shouldn't because the val/test should represent the real world distribution.
  I might have not used properly smote because i applied on everything so the final evaluation value i had was wrong
### Third Experiment
I applied ADASYN and with the "all" parameter, and only on data used to train the model. 
results : not worth mentioning, not better
### Fourth Experiment
I tried using bova (boosted one vs all) the idea is to train two classifiers one on all the data and the other adapted for minority classes. Then if the unbalanced model is confident we follow its prediction, otherwise we follow the balanced model. If neither are confident we take the mean probability.
i get these results : 
               precision    recall  f1-score   support

   Demolition       0.71      0.84      0.77      3151
         Road       0.64      0.74      0.69      1431
  Residential       0.80      0.76      0.78     14844
   Commercial       0.67      0.67      0.67     10042
   Industrial       0.15      0.14      0.15       132
Mega Projects       0.03      0.20      0.05        15

     accuracy                           0.73     29615
    macro avg       0.50      0.56      0.52     29615
 weighted avg       0.74      0.73      0.73     29615
# Fifth Experiment
AutoML result : 
LGBMClassifier(colsample_bytree=np.float64(0.9588740520120813),
               learning_rate=np.float64(0.06940593814641595), max_bin=255,
               n_estimators=1104, n_jobs=-1, num_leaves=45,
               reg_alpha=np.float64(0.0902804651294653),
               reg_lambda=np.float64(0.03411127868780876), verbose=-1)

              precision    recall  f1-score   support

           0       0.74      0.82      0.78      3151
           1       0.83      0.69      0.75      1431
           2       0.80      0.81      0.80     14844
           3       0.71      0.69      0.70     10042
           4       0.38      0.05      0.08       132
           5       0.00      0.00      0.00        15

    accuracy                           0.76     29615
   macro avg       0.57      0.51      0.52     29615
weighted avg       0.76      0.76      0.76     29615
On the public i get 0.90937
# Sixth Experiment
Validation F1 Score: 0.742020229404521, 
 Validation F1 Score (non-weighted): [0.77722702 0.72779923 0.78921917 0.7023204  0.95555556 0.5       ]
              precision    recall  f1-score   support

           0       0.70      0.87      0.78      3151
           1       0.67      0.79      0.73      1431
           2       0.84      0.75      0.79     14844
           3       0.68      0.72      0.70     10042
           4       0.93      0.98      0.96       132
           5       0.67      0.40      0.50        15

    accuracy                           0.75     29615
   macro avg       0.75      0.75      0.74     29615
weighted avg       0.76      0.75      0.76     29615
With a feature representing the minority class (4 and 5), the f1 score of these two minority classes is way higher but the overall score hasnt not improved a lot, the other f1 score have gone down.

idea after : do multiple models : per medium high low freq, whole datasaet and then rf + lgbm + hist then on val split optimize param so biggestf1 score overall

# 7th Experiment
Best trial:
  F1 Score: 0.6340419797020355
  Params: 
    n_estimators: 29
    max_depth: 19
    criterion: log_loss
    min_samples_split: 3
    min_samples_leaf: 7
    class_weight: balanced
Validation F1 Score: 0.6258216409686566, 
 Validation F1 Score (non-weighted): [0.69297621 0.65227985 0.77509843 0.67829965 0.83127572 0.125     ]
              precision    recall  f1-score   support

           0       0.63      0.77      0.69      3151
           1       0.65      0.65      0.65      1431
           2       0.81      0.74      0.78     14844
           3       0.66      0.70      0.68     10042
           4       0.91      0.77      0.83       132
           5       1.00      0.07      0.12        15

    accuracy                           0.73     29615
   macro avg       0.78      0.62      0.63     29615
weighted avg       0.73      0.73      0.73     29615

This is just sickit's onevsrest with randomforestclassifier, we can see the feature indicating the minority classes make them bettter but worse than the other.

# 8th Experiment
Bova with the new rf feature for minority classes 4 and 5:
Searching for optimal cutoffs...
  New best! F1=0.6224, Industrial=0.10, Mega=0.05
  New best! F1=0.6328, Industrial=0.10, Mega=0.10
  New best! F1=0.6477, Industrial=0.15, Mega=0.10
  New best! F1=0.6589, Industrial=0.15, Mega=0.15
  New best! F1=0.6733, Industrial=0.20, Mega=0.15
  New best! F1=0.6764, Industrial=0.25, Mega=0.15
  New best! F1=0.6805, Industrial=0.25, Mega=0.20
  New best! F1=0.6832, Industrial=0.30, Mega=0.20
  New best! F1=0.6849, Industrial=0.35, Mega=0.20
  New best! F1=0.6867, Industrial=0.40, Mega=0.20
  New best! F1=0.6873, Industrial=0.50, Mega=0.20

Final best cutoffs: [0.5 0.5 0.5 0.5 0.5 0.2]
Final best F1: 0.6873
Optimized cutoffs: [0.5 0.5 0.5 0.5 0.5 0.2]
               precision    recall  f1-score   support

   Demolition       0.71      0.84      0.77      3151
         Road       0.64      0.75      0.69      1431
  Residential       0.81      0.76      0.78     14844
   Commercial       0.68      0.67      0.68     10042
   Industrial       0.95      0.83      0.89       132
Mega Projects       0.29      0.60      0.39        15

     accuracy                           0.74     29615
    macro avg       0.68      0.74      0.70     29615
 weighted avg       0.74      0.74      0.74     29615

 # 9th Experiment
First ensemble with just means

Ensemble F1 Score Per Class: [0.74569147 0.6731506  0.78088463 0.66646375 0.21212121 0.03174603]
              precision    recall  f1-score   support

           0       0.67      0.84      0.75      3151
           1       0.58      0.80      0.67      1431
           2       0.82      0.75      0.78     14844
           3       0.68      0.65      0.67     10042
           4       0.14      0.42      0.21       132
           5       0.02      0.07      0.03        15

    accuracy                           0.73     29615
   macro avg       0.48      0.59      0.52     29615
weighted avg       0.74      0.73      0.73     29615
# 10th Experiment
with mean / logistic regression at the end :               
precision    recall  f1-score   support

           0       0.72      0.83      0.78      3151
           1       0.81      0.67      0.73      1431
           2       0.81      0.80      0.80     14844
           3       0.70      0.71      0.71     10042
           4       0.00      0.00      0.00       132
           5       0.00      0.00      0.00        15

    accuracy                           0.76     29615
   macro avg       0.51      0.50      0.50     29615
weighted avg       0.76      0.76      0.76     29615

Ensemble F1 Score Per Class: 
              precision    recall  f1-score   support

           0       0.65      0.87      0.74      3151
           1       0.67      0.75      0.71      1431
           2       0.84      0.73      0.78     14844
           3       0.67      0.70      0.68     10042
           4       0.18      0.39      0.25       132
           5       0.00      0.00      0.00        15

    accuracy                           0.73     29615
   macro avg       0.50      0.57      0.53     29615
weighted avg       0.75      0.73      0.74     29615