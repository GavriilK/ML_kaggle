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