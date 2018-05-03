#grid search modeling 
from __future__ import print_function
import csv
import cleaner
import pandas as pd
from pprint import pprint
from time import time
import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import nlp_fake


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')



# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(class_weight= 'balanced')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None),
    'vect__ngram_range': ( (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2'),
    #'clf__n_iter': (10, 50, 80),

}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier

    df = pd.read_csv('Data/train_set.csv' , sep="\t")
    from sklearn.utils import resample
    df_test = resample(df, n_samples=1000)
    X = df_test['review_text']
    y = df_test['filtered']
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring ='roc_auc')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    #writing results to csv
    with open('grid_results.csv', 'a', newline='\n') as csvfile:
         fieldnames = ['model', 'score', 'paramaters']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         writer.writerow({'model': f'{grid_search.best_estimator_}', 'score': f'{grid_search.best_score_}', 'paramaters': f'{parameters.keys()}'})
