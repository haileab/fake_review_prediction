from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np


def tfidfed(X_train, X_test, y_train, y_test):
    '''
    Creates a term frequency/inverse document frequency vector for the review texts.
    '''
    stemmer = PorterStemmer()

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    stop_words = set(stopwords.words("english"))
    vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', tokenizer=tokenize,  analyzer = 'word', stop_words=stop_words, max_features=2000)
    review_transformer = vectorizer.fit(X_train)
    X_train_tfidf = review_transformer.transform(X_train)
    X_test_tfidf = review_transformer.transform(X_test)
    tfidf_columns = review_transformer.get_feature_names()
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_columns


def count_vectored(X_train, X_test, y_train, y_test):
    '''
    Creates a word count vector for the review text.
    '''
    stop_words = set(stopwords.words("english"))
    vectorizer = CountVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', analyzer = 'word', stop_words=stop_words, max_features=2000)
    review_transformer = vectorizer.fit(X_train)
    X_trained = review_transformer.transform(X_train)
    X_tested = review_transformer.transform(X_test)
    vect_columns = review_transformer.get_feature_names()
    print(review_transformer.get_feature_names()[:30])
    return X_trained, X_tested, y_train, y_test, vect_columns


def run_tfidf_models(X_trained, X_tested, y_train, y_test):
    '''
    Runs three classification models and prints out the results.
    '''
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import LinearSVC

    model = MultinomialNB()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
    print(f"Accuaracy: {accuracy_score(y_test, preds)}")

    model = BernoulliNB()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
    print(f"Accuaracy: {accuracy_score(y_test, preds)}")

    model = LinearSVC()
    model.fit(X_trained, y_train)
    preds = model.predict(X_tested)
    print(f"{model} Results",confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))
    print(f"Accuaracy: {accuracy_score(y_test, preds)}")





if __name__ == '__main__':
    pass
