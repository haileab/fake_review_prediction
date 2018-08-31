from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import pandas as pd


#sentiment analysis
def sentiment():
    '''
    Used for creating a chart to compare sentiment analysis feature importance
    with the results of Fake/genuine feature importance.
    '''
    df = pd.read_csv("Data/train_set_2.csv" , sep="\t")[0:10000]
    df.reset_index(drop=True, inplace=True)
    features = ['review_text','rating','filtered','review_length','biz_rvws','user_rvws', 'avg_rating', 'past_filt', 'percent_filt', 'avg_rev_len', 'word_count', 'avg_word', 'stopwords', 'numbers', 'upper', 'Friday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    df = df[features]
    #mask = (data['value2'] == 'A') & (data['value'] > 4)
    mask = (df['rating'] > 4) | (df['rating'] < 2)
    df = df[mask]
    X_train, X_test = train_test_split(df, test_size=0.15, random_state=0)
    y_train = X_train.pop('rating')
    X_train_tfidf = X_train.pop('review_text')
    y_test = X_test.pop('rating')
    X_test_tfidf = X_test.pop('review_text')
    return X_train, X_train_tfidf, X_test, X_test_tfidf, y_train, y_test
