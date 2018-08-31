from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

def run_models(X_train, X_test, y_train, y_test):
    '''
    Runs two classification models and prints results.
    '''

    clf = RandomForestClassifier(bootstrap=True, class_weight={0:.1, 1:.9}, criterion='gini',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"{clf} Results",confusion_matrix(y_test, preds))
    print('\n')
    print("Feature importance: {clf.feature_importances_}")
    print(classification_report(y_test, preds))

    logreg = linear_model.LogisticRegression(class_weight='balanced')
    result = logreg.fit(X_train, y_train)
    preds = result.predict(X_test)
    print(f"{result} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
