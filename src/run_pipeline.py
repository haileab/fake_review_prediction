import multiprocessing as mp
import pandas as pd
pool = mp.Pool(processes=4)

import cleaner
import classifiers
import data_split
import nlp_models

def go():
    '''
    Runs the data cleaning/prep and the classification and NLP models.
    '''

    df = cleaner.merge_files()
    df = cleaner.cumulator(df)
    print(df.head())
    X_train, X_train_tfidf, X_test, X_test_tfidf, y_train, y_test = data_split.unbalanced(df)
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_columns = nlp_models.tfidfed(X_train_tfidf, X_test_tfidf, y_train, y_test)
    X_train_mid = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_columns)
    X_test_mid = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_columns)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, X_test_mid], axis=1)
    X_train = pd.concat([X_train, X_train_mid], axis=1)
    print(X_train.shape)
    print(X_test.shape)
    classifiers.run_models(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    go()
