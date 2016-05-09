from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

data = load_files('sentiment/txt_sentoken')

def run_tests():
	clf = get_clf
	scores = cross_val_score(clf, data.data, data.target, cv=10, verbose=1)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def get_clf():
	clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
                ("svc", LinearSVC(C=100))])
	clf.fit(data.data, data.target)

	return clf