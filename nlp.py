from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report


def balanced_class(df):
    class5 = df[df.stars == 5]
    class1 = df[df.stars == 1]
    class1_count = len(class1)
    sub_5 = resample(class5, n_samples=class1_count)
    frames = [sub_5,class1]
    balanced = pd.concat(frames)
    balanced = resample(balanced, n_samples=class1_count*2)
    return balanced

def vectored(df):
    print(df.head())
    y = df.stars
    X = df.text
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
    stop_words = set(stopwords.words("english"))
    tfidf = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,1), max_features=500, min_df = 5, use_idf = True)
    review_transformer = tfidf.fit(X_train)
    X_trained = review_transformer.transform(X_train)
    X_tested = review_transformer.transform(X_test)
    return X_trained, X_tested, y_train, y_test

def run_models(X_trained, X_tested, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    #from sklearn import preprocessing
    model = MultinomialNB()
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


def NB(X_trained, X_tested, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    #from sklearn import preprocessing
    nb = MultinomialNB()
    nb.fit(X_trained, y_train)
    preds = nb.predict(X_tested)
    return preds

if __name__ == '__main__':
    pass
