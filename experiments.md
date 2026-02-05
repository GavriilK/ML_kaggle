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