import string
import time
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from sklearn.pipeline import Pipeline

import sanders

################################################################################
# Load Sentiment Lexicon

with open('lexicons/positive-words.txt', 'r') as fp:
	positive_words = fp.read().split('\n')

with open('lexicons/negative-words.txt', 'r') as fp:
	negative_words = fp.read().split('\n')

################################################################################
# Preprocessing
stop_words = ENGLISH_STOP_WORDS.union(list(string.punctuation))
stop_words = stop_words.difference(positive_words)
stop_words = stop_words.difference(negative_words)
stop_words = stop_words.union(['RT', 'rt', 'via', 'red', 'carpet', 'redcarpet', '#oscars'])
# few emoticons --[u'\u2026', u'\ud83d', u'\ude0d', '#oscars']

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    # r'(?:[^A-Za-z0-9]+)'
    # r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
	tokenization = [token for token in tokens_re.findall(s) if token not in stop_words]

	# Stemmer
	# stemmer = PorterStemmer()
	# tokenization = [stemmer.stem(token) for token in tokenization]

	return tokenization

def preprocess(s, lowercase=True):
    return [token if emoticon_re.search(token) else token.lower() for token in tokenize(s)]

################################################################################
# Load train data

sanders_train = sanders.get_tweets()
categories = sanders.target_names

################################################################################
# Vectorization
def get_clf():
    vectorizer_params = {
    	"ngram_range": (1,2),
    	"token_pattern": ur'\b\w+\b',
    	"tokenizer": preprocess,
    	"max_features": 10000,
    }

    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()

    text_clf = Pipeline([('vect', CountVectorizer(**vectorizer_params)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    text_clf.fit(sanders_train["data"], sanders_train["target"])

    return text_clf

if __name__ == '__main__':
    text_clf = get_clf()
    new_docs = ['awesome movie cool good great', 'what the hell is this shit!?']
    predicted = text_clf.predict(new_docs)

    for doc, category in zip(new_docs, predicted):
        print('%r => %s' % (doc, sanders.target_names[category]))
