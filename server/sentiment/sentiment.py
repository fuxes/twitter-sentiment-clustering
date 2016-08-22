import os
import time
import numpy as np
from pprint import pprint

from sklearn import datasets
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB


class SentimentClf(object):
    pkl_path = os.path.join(
        os.path.dirname(__file__), '/pkls/sentiment_clf.pkl')
    dataset_path = os.path.join(os.path.dirname(__file__), 'data')

    def __init__(self):
        self.dataset = datasets.load_files(SentimentClf.dataset_path)
        self.clf = None

    def get_target_name(self, category):
        return self.dataset.target_names[category]

    def get_clf(self):
        t0 = time.time()
        if not self.clf:
            try:
                clf = joblib.load('sentiment.pkl')
            except IOError:
                clf = self.train_clf()
                joblib.dump(clf, 'sentiment.pkl')
            finally:
                self.clf = clf
        print("Sentiment classifier returned in : %2.fs" % (time.time() - t0))
        return self.clf

    def train_clf(self):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("svc", LinearSVC(C=100))
        ])
        pipeline.fit(self.dataset.data, self.dataset.target)
        return pipeline

    def gs(self, clf, params):
        gs = GridSearchCV(
            clf, params, verbose=3, n_jobs=2)
        gs.fit(self.dataset.data, self.dataset.target)

        print(gs.best_estimator_)
        print(gs.best_score_)
        pprint(gs.grid_scores_)
        # import pdb; pdb.set_trace()

    def run_gs(self):
        # Run gridsearch for LinearSVC Classifier
        # params = {
        #     "tfidf__ngram_range": [(1, 2)],
        #     "svc__C": [.01, .1, 1, 10, 100],
        #     "svc__loss": ('squared_hinge', 'hinge')
        # }
        # clf = Pipeline([
        #     ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        #     ("svc", LinearSVC())
        # ])
        # self.gs(clf, params)

        # # Run gridsearch for DecisionTree Classifier
        # params = {
        #     "tfidf__ngram_range": [(1,1), (1, 2)],
        #     'tree__criterion': ('gini', 'entropy'),
        #     'tree__splitter': ("best", "random"),
        #     'tree__max_depth': np.arange(3, 10),
        # }
        # clf = Pipeline([
        #     ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        #     ("tree", DecisionTreeClassifier(random_state=42))
        # ])
        # self.gs(clf, params)

        # Run gridsearch for MultinomialNB Classifier
        params = {
            "tfidf__ngram_range": [(1,1), (1, 2)],
            'mnb__alpha': [0, 0.1, 0.5, 1.0],
        }
        clf = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("mnb", MultinomialNB())
        ])
        self.gs(clf, params)

if __name__ == '__main__':
    SentimentClf().run_gs()
