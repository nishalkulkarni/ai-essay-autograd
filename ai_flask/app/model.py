import nltk
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('app/features/features_set_3.csv')
data


X = data.iloc[:, 3:]
y = data.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(class_weight='balanced_subsample')
rf_params = {'n_estimators': list(range(20, 200, 10)),
             'max_depth': list(range(2, 14, 1))}


scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='macro')
rf_random = GridSearchCV(estimator=rf, param_grid=rf_params,
                         cv=5, verbose=2,  n_jobs=2, scoring=scorer)

rf_random.fit(X_train, y_train)

rf_final = RandomForestClassifier(random_state=0, n_estimators=rf_random.best_params_[
                                  'n_estimators'], max_depth=rf_random.best_params_['max_depth'], class_weight='balanced_subsample')
rf_final.fit(X_train, y_train)
X_pred = rf_final.predict(X_test)

report = classification_report(X_pred, y_test, digits=3)
print(report)

pickle.dump(rf_final, open('model.pkl', 'wb'))
