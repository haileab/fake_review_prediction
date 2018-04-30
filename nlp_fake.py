from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
import numpy as np


def balanced_class(df):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=1000)
    classN = df_train[df_train.filtered == 0]
    classY = df_train[df_train.filtered == 1]
    classY_count = len(classY)
    sub_classN = resample(classN, n_samples=classY_count)
    frames = [sub_classN,classY]
    balanced = pd.concat(frames)
    balanced = resample(balanced, n_samples=classY_count*2)
    y_train = balanced.pop('filtered')
    X_train = balanced['review_text']
    #print(type(X_train), type(y_train))
    y_test = df_test.pop('filtered')
    X_test = df_test['review_text']
    #print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test


def tfidfed(X_train, X_test, y_train, y_test):
    stop_words = set(stopwords.words("english"))
    tfidf = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', analyzer = 'word', stop_words=stop_words, max_features=5000)
    review_transformer = tfidf.fit(X_train)
    X_train_tfidf = review_transformer.transform(X_train)
    X_test_tfidf = review_transformer.transform(X_test)
    tfidf_columns = review_transformer.get_feature_names()
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_columns


def count_vectored(X_train, X_test, y_train, y_test):
    stop_words = set(stopwords.words("english"))
    tfidf = CountVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', analyzer = 'word', stop_words=stop_words, max_features=5000)
    review_transformer = tfidf.fit(X_train)
    X_trained = review_transformer.transform(X_train)
    X_tested = review_transformer.transform(X_test)
    print(review_transformer.get_feature_names()[:30])
    return X_trained, X_tested, y_train, y_test

###
def run_vect_models(X_trained, X_tested, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import LinearSVC
    #from sklearn import preprocessing

    model = MultinomialNB()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))


    model = BernoulliNB()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))

    model = LinearSVC()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))

def run_tfidf_models(X_trained, X_tested, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import LinearSVC
    #from sklearn import preprocessing

    # model = MultinomialNB()
    # model.fit(X_trained, y_train)
    # preds = model.predict(X_tested)
    # print(f"{model} Results",confusion_matrix(y_test, preds))
    # print('\n')
    # print(classification_report(y_test, preds))

    # model = GaussianNB()
    # X_array = (X_trained).toarray()
    # print(X_trained.shape, X_array.shape)
    # model.fit(X_array, y_train)
    # X_test_arr = (X_tested).toarray()
    # preds = model.predict(X_test_arr)
    # print(f"{model} Results",confusion_matrix(y_test, preds))
    # print('\n')
    # print(classification_report(y_test, preds))

    # model = BernoulliNB()
    # model.fit(X_trained, y_train)
    # preds = model.predict(X_tested)
    # print(f"{model} Results",confusion_matrix(y_test, preds))
    # print('\n')
    # print(classification_report(y_test, preds))

    model = LinearSVC()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))


def SGD(X_trained, X_tested, y_train, y_test):
    model = SGDClassifier()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))


def NB(X_trained, X_tested, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    #from sklearn import preprocessing
    nb = MultinomialNB()
    nb.fit(X_trained, y_train)
    preds = nb.predict(X_tested)
    return preds

if __name__ == '__main__':
    pass
