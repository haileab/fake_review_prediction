from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import pandas as pd

def balanced_split(df):
    df.reset_index(drop=True, inplace=True)
    features = ['review_text','rating','filtered','review_length','biz_rvws','user_rvws', 'avg_rating', 'past_filt', 'percent_filt', 'avg_rev_len']
    df = df[features]
    X_train, X_test = train_test_split(df, test_size=0.15, random_state=0)
    classN = X_train[X_train.filtered == 0]
    classY = X_train[X_train.filtered == 1]
    classY_count = len(classY)
    sub_classN = resample(classN, n_samples=classY_count)
    frames = [sub_classN,classY]
    X_train = pd.concat(frames)
    X_train = resample(X_train, n_samples=classY_count*2)
    y_train = X_train.pop('filtered')
    X_train_tfidf = X_train.pop('review_text')
    y_test = X_test.pop('filtered')
    X_test_tfidf = X_test.pop('review_text')
    return X_train, X_train_tfidf, X_test, X_test_tfidf, y_train, y_test

def unbalanced_split(df):
    df.reset_index(drop=True, inplace=True)
    features = ['review_text','rating','filtered','review_length','biz_rvws','user_rvws', 'avg_rating', 'past_filt', 'percent_filt', 'avg_rev_len']
    df = df[features]
    X_train, X_test = train_test_split(df, test_size=0.15, random_state=0)
    # classN = X_train[X_train.filtered == 0]
    # classY = X_train[X_train.filtered == 1]
    # classY_count = len(classY)
    # sub_classN = resample(classN, n_samples=classY_count)
    # frames = [sub_classN,classY]
    # X_train = pd.concat(frames)
    # X_train = resample(X_train, n_samples=classY_count*2)
    y_train = X_train.pop('filtered')
    X_train_tfidf = X_train.pop('review_text')
    y_test = X_test.pop('filtered')
    X_test_tfidf = X_test.pop('review_text')
    return X_train, X_train_tfidf, X_test, X_test_tfidf, y_train, y_test

def run_models(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(bootstrap=True, class_weight={0:.1, 1:.9}, criterion='gini',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    F = clf.feature_importances_
    print(f"{clf} Results",confusion_matrix(y_test, preds))
    print('\n')
    #print(f"Feature importance: {[ '%.5f' % elem for elem in F ]}")
    print(classification_report(y_test, preds))
    print(accuracy_score(y_test, preds))

    logreg = linear_model.LogisticRegression(class_weight='balanced')
    result = logreg.fit(X_train, y_train)
    preds = result.predict(X_test)
    print(f"{result} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
    print(accuracy_score(y_test, preds))

    gbc = GradientBoostingClassifier(n_estimators=200)
    gbc.fit(X_train, y_train)
    preds = gbc.predict(X_test)
    print(f"{gbc} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
    print(accuracy_score(y_test, preds))
