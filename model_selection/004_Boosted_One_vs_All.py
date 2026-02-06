from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from random import sample, shuffle, seed
import numpy as np
import pandas as pd
import datetime

class BOVA:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = []
    
    def fit(self,X,Y,type="hist"):
        seed(self.random_state)
        N = len(Y[0])
        for category_idx in range(N):
            assert len(self.models)==category_idx
            if type=="logit":
                self.models.append({
                    "unbalanced": LogisticRegression(random_state=self.random_state),
                    "balanced" : LogisticRegression(random_state=self.random_state)
                })
            if type=='hist':
                self.models.append({
                    "unbalanced": HistGradientBoostingClassifier(random_state=self.random_state),
                    "balanced" : HistGradientBoostingClassifier(random_state=self.random_state)
                })
            if type=="mlp":
                self.models.append({
                    "unbalanced": MLPClassifier(random_state=self.random_state,hidden_layer_sizes=5),
                    "balanced" : MLPClassifier(random_state=self.random_state,hidden_layer_sizes=5)
                })
            yt = Y[:, category_idx]

            # positive and negative examples
            idx_1 = [i for i,y in enumerate(yt) if y == 1]
            idx_0 = [i for i,y in enumerate(yt) if y != 1]

            # Create a balanced subset by undersampling negatives
            if len(idx_0) > len(idx_1):
                #minority class => undersample negatives
                balanced_idx = idx_1 + sample(idx_0, len(idx_1))
            elif len(idx_0) < len(idx_1):
                #majority class => undersample positives
                balanced_idx = idx_0 + sample(idx_1, len(idx_0))
            else :
                balanced_idx = idx_1 + idx_0

            # train (un)balanced models 
            self.models[category_idx]['unbalanced'].fit(X,yt)
            self.models[category_idx]['balanced'].fit(X.iloc[balanced_idx], yt[balanced_idx])
    
    @staticmethod
    def __proba(unbalanced_proba, balanced_proba, cutoff):
        if unbalanced_proba > cutoff:
            return unbalanced_proba
        elif balanced_proba <= cutoff:
            return balanced_proba
        else:
            return (unbalanced_proba + balanced_proba)/2

    def __predict_proba(self, X, cutoff):
        probas = [[0.0 for _ in X] for _ in self.models]

        for i, cat_models in enumerate(self.models):
            y_unbalanced_proba = cat_models['unbalanced'].predict_proba(X)[:,1]

            y_balanced_proba = cat_models['balanced'].predict_proba(X)[:,1]

            probas[i] = [BOVA.__proba(yup, ybp, cutoff) for yup,ybp in zip(y_unbalanced_proba,y_balanced_proba)]

        return list(zip(*probas))
    
    def predict_proba(self, X, cutoff=0.5):
        return np.array(self.__predict_proba(X,cutoff))

    def predict(self, X, multilabel=False, cutoff=0.5):
        if multilabel:
            return pd.DataFrame(self.__predict_proba(X,cutoff)).apply(lambda x: [1 if xx>cutoff else 0 for xx in x]).to_numpy()
        else :
            return pd.DataFrame(self.__predict_proba(X,cutoff)).apply(lambda x: (x == np.max(x)).astype(int),axis=1).to_numpy()

    def predict_with_cutoffs(self, X, cutoffs):
        probas = self.predict_proba(X)

        adjusted_probas = probas - cutoffs

        predictions = np.argmax(adjusted_probas, axis=1)

        onehot = np.zeros((len(predictions),6))
        onehot[np.arange(len(predictions)), predictions] = 1
        return onehot

    def optimize_cutoffs(self, X_val, y_val):
        """Grid search on minority classes jointly"""
        best_f1 = 0
        best_cutoffs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        print("Searching for optimal cutoffs...")
        
        # Search lower cutoffs for minorities (make it easier to predict them)
        cutoff_options_industrial = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        cutoff_options_mega = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        
        for cut_ind in cutoff_options_industrial:
            for cut_mega in cutoff_options_mega:
                cutoffs = np.array([0.5, 0.5, 0.5, 0.5, cut_ind, cut_mega])
                
                y_pred = self.predict_with_cutoffs(X_val, cutoffs)
                f1 = f1_score(np.argmax(y_val, axis=1), 
                             np.argmax(y_pred, axis=1), 
                             average='macro')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_cutoffs = cutoffs.copy()
                    print(f"  New best! F1={f1:.4f}, Industrial={cut_ind:.2f}, Mega={cut_mega:.2f}")
        
        print(f"\nFinal best cutoffs: {best_cutoffs}")
        print(f"Final best F1: {best_f1:.4f}")
        return best_cutoffs

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y_1D = pd.read_parquet('processing/y_train.parquet').values.ravel()

#adapt y for this training
lb = LabelBinarizer()
y = lb.fit_transform(y_1D)


# Three-way split: train (80%), val (10%), test (10%)
X_temp, X_test, y_temp, y_test = train_test_split(train, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp)  # 0.111 * 0.9 ≈ 0.1

# Fit on training set
bova = BOVA()
bova.fit(X_train, y_train)

# Optimize cutoffs on validation set
print("Optimizing cutoffs...")
best_cutoffs = bova.optimize_cutoffs(X_val, y_val)
print(f"Optimized cutoffs: {best_cutoffs}")

# Evaluate on test set with optimized cutoffs
y_bova = bova.predict_with_cutoffs(X_test, best_cutoffs)

y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_bova, axis=1)

print(classification_report(
    y_true=y_test_labels, y_pred=y_pred_labels, 
    target_names = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects'])
    )

# Predict on submission set
submission_pred = bova.predict_with_cutoffs(submission, best_cutoffs)
submission_pred_labels = np.argmax(submission_pred, axis=1)

date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred_labels, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/bova_submission_{date}.csv', index=True, index_label='Id')