from pprint import pprint
from collections import Counter, defaultdict
import re
import string
import logging
import json
from optparse import OptionParser
import time
import sys
import constants

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GMM
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
from sentiment.sentiment import SentimentClf

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Parse commandline arguments
op = OptionParser()
op.add_option("--data",
			  dest="filename", default="data/oscars_trim.json",
			  help="JSON file that contains the tweets to be clustered")

op.add_option("--no_kmeans",
			  action="store_false", dest="run_kmeans", default=True,
			  help="Disable K-Means Clustering")

op.add_option("--no_clf",
			  action="store_false", dest="train_clf", default=False,
			  help="Disable Classifier training.")

op.add_option("--min_df", dest="min_df", default=1,
			  help="Minimum frequency for documents")

op.add_option("--max_df", dest="max_df", default=1.0,
			  help="Maximum frequency for documents")

op.add_option("--n_clusters", dest="n_clusters", default="8",
			  help="Maximum number of clusters to extract K-Means.")

op.add_option("--n_features", dest="n_features", default=None,
			  help="Maximum number of features to extract.")

op.add_option("--n_grams", dest="ngrams", default="(1,1)")

op.add_option("--sentiment",
			  action="store_true", dest="use_sentiment", default=True,
			  help="Evaluate sentiment polarity of cluster words")

op.add_option("--idf",
			  action="store_true", dest="use_idf", default=False,
			  help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--stem",
			  action="store_true", dest="use_stem", default=False,
			  help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--bin",
			  action="store_true", dest="use_bin", default=False,
			  help="Disable Inverse Document Frequency feature weighting.")

(opts, args) = op.parse_args()
opts.ngrams = eval(opts.ngrams)
opts.n_clusters = eval(opts.n_clusters)

if len(args) > 0:
    print(args)
    op.error("this script takes no arguments.")
    sys.exit(1)

print

################################################################################
# Load Sentiment Lexicon

with open('lexicons/positive-words.txt', 'r') as fp:
	positive_words = fp.read().split('\n')

with open('lexicons/negative-words.txt', 'r') as fp:
	negative_words = fp.read().split('\n')

################################################################################
# Preprocessing
stop_words = text.ENGLISH_STOP_WORDS.union(list(string.punctuation))
stop_words = stop_words.difference(positive_words)
stop_words = stop_words.difference(negative_words)

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
	tokenization = [token for token in tokens_re.findall(s) if token not in stop_words and not token.startswith('http')]

	# Stemmer
	if opts.use_stem:
		stemmer = PorterStemmer()
		tokenization = [stemmer.stem(token) for token in tokenization]

	return tokenization

def preprocess(s, lowercase=True):
    return [token if emoticon_re.search(token) else token.lower() for token in tokenize(s)]

################################################################################
# Load dataset

tt = time.time()
print("Loading %s dataset" % opts.filename)

t0 = time.time()

def to_time(timestamp):
	return time.mktime(time.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y"))

with open(opts.filename, 'r') as fp:
	raw_dataset = json.load(fp)
	tweets_list = raw_dataset['data']
	tweets_dict = dict((tweet['id_str'], {
		'id_str': tweet['id_str'],
		'created_at': tweet['created_at'],
		'text': tweet['text'],
		'user': {
			'profile_image_url': tweet['user']['profile_image_url'],
			'name': tweet['user']['name'],
			'screen_name': tweet['user']['screen_name']
			},
		}) for tweet in tweets_list)
	corpus = map(lambda t: t['text'], tweets_list)
	corpus.sort()
	corpus = list(set(corpus)) ## Should remove retweets from raw_dataset

oldest = time.ctime(min(to_time(t['created_at']) for t in tweets_list))
newest = time.ctime(max(to_time(t['created_at']) for t in tweets_list))

print("done in %fs" % (time.time() - t0))
print("%s tweets loaded: from %s to %s" % (len(tweets_list), oldest, newest))
print

################################################################################
# Vectorization

vectorizer_params = {
	"ngram_range": opts.ngrams,
	"token_pattern": ur'\b\w+\b',
	"tokenizer": preprocess,
	"min_df": eval(opts.min_df),
	"max_df": eval(opts.max_df),
	"max_features": opts.n_features,
	"binary": opts.use_bin
}
print vectorizer_params

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time.time()
count_vect = CountVectorizer(**vectorizer_params)
X = count_vect.fit_transform(corpus)

if opts.use_idf:
	tfidf_transformer = TfidfTransformer()
	X = tfidf_transformer.fit_transform(X)

print("done in %fs" % (time.time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print


################################################################################
# Clustering

if opts.run_kmeans:
	n_clusters_range = []
	if type(opts.n_clusters) is int:
		n_clusters_range.append(opts.n_clusters)
	else:
		n_clusters_range = range(opts.n_clusters[0], opts.n_clusters[1])

	for clusters in n_clusters_range:
		# km = KMeans(n_clusters=clusters,
		km = MiniBatchKMeans(n_clusters=clusters,
					init='k-means++',
					max_iter=300,
					n_init=20,
					# n_jobs=2,
					# verbose=1,
					random_state=42)

		print("Clustering sparse data with %s" % km)
		t0 = time.time()
		km.fit_predict(X)
		print("done in %0.3fs" % (time.time() - t0))
		print

		print("Top terms per cluster:")
		order_centroids = km.cluster_centers_.argsort()[:, ::-1]
		terms = count_vect.get_feature_names()

		# top terms for a cluster
		try:
			for i in range(clusters):
			    print("Cluster %d (%s):" % (i, len([t for t in km.labels_ if t == i])))
			    for ind in order_centroids[i, :10]:
			        print(terms[ind])
			    print
		except:
			print "Unexpected error:", sys.exc_info()[0]

################################################################################
# Sentiment analysis classifier training

""" DESIGN
{"clusters": [] list of clusters
	{
		size: amount of words
		centroids: most common words
		words: [] list of wordclouds
		{
			text: term
			frequency: volume of this word
			polarity: sentiment polarity
			sentiment: positive or negative
			related_tweets: [] list of tweets ids that has this word
		}
		### position: in an X, Y plane # YET TO BE IMPLEMENTED
	}
"clusters_size": number of clusters
"""

if opts.use_sentiment:
	sentiment = SentimentClf()
	sentiment_clf = sentiment.get_clf()
	predicted = sentiment_clf.predict(corpus)

	tweets_by_text = dict((d['text'], dict(d, index=index)) for (index, d) in enumerate(tweets_list))

	for i, category in enumerate(predicted):
		tweet_id = tweets_by_text[corpus[i]]['id_str']
		tweets_dict[tweet_id]['sentiment'] = sentiment.get_target_name(category)

################################################################################
# Cloud of words

def get_term_polarity(tweets_ids):
	c = 0
	for id in tweets_ids:
		c += (1 if tweets_dict[id]['sentiment'] == 'positive' else
			-1 if tweets_dict[id]['sentiment'] == 'negative' else
			0)
	return 'positive' if c > 0 else 'negative' if c < 0 else 'neutral'

if opts.use_sentiment:
	# Counter for each cluster
	counters = [Counter() for _ in range(opts.n_clusters)]
	idcounters = [defaultdict(list) for _ in range(opts.n_clusters)]

	# Generar un counter de palabras para cada cluster
	for i, cluster_idx in enumerate(km.labels_):
		tweet_tokens = preprocess(tweets_list[i]['text'])
		for token in tweet_tokens:
			idcounters[cluster_idx][token].append(tweets_list[i]['id_str'])
		counters[cluster_idx].update(tweet_tokens)

	clusters = []
	for i, counter in enumerate(counters):
		terms = []
		for word, count in counter.items():
			if count > 5 and word not in constants.STOPWORDS:
				sentiment_polarity = get_term_polarity(idcounters[i][word])
				terms.append({
					"term": word,
					"count": count,
					"tweets": idcounters[i][word],
					"sentiment": sentiment_polarity
				})

		clusters.append({
			"terms": terms,
			"cluster_size": len(terms)
		})

if opts.use_sentiment:
	# Save JSON with Tweets Sentiment
	with open('vis/tweets.json', 'w') as fp:
		json.dump(tweets_dict, fp, indent=4)
		print "SENTIMENT SAVED ON: %s TWEETS" % len(tweets_dict.keys())

	with open('vis/bubbles.json', 'w') as fp:
		json.dump({"data": clusters}, fp, indent=4)
		print "SAVED: %s Clusters" % len(clusters)

################################################################################
# Classifier Training

if opts.train_clf:

	# Split data into training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, km.labels_, test_size=0.33, random_state=42)

	clf = LinearSVC()

	# 10-fold ross validation
	cv = KFold(y_train.size, 10)
	scores = cross_val_score(clf, X_train, y_train, cv=cv)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	# Accuaracy on test data
	clf.fit(X_train, y_train)
	y_true, y_pred = y_test, clf.predict(X_test)
	print(metrics.classification_report(y_true, y_pred))



print ("Total time is: %0.3fs" % (time.time() - tt))