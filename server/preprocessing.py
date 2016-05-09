from pprint import pprint
from collections import Counter, defaultdict
import re
import string
import logging
import json
from optparse import OptionParser
import time
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud

import matplotlib.pyplot as plt
from sentiment import sentiment

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Parse commandline arguments
op = OptionParser()
op.add_option("--data",
			  dest="filename", default="oscars.json",
			  help="JSON file that contains the tweets to be clustered")

op.add_option("--no-kmeans",
			  action="store_false", dest="run_kmeans", default=True,
			  help="Disable K-Means Clustering")

op.add_option("--no-idf",
			  action="store_false", dest="use_idf", default=True,
			  help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--no-clf",
			  action="store_false", dest="train_clf", default=True,
			  help="Disable Classifier training.")

op.add_option("--min-df", dest="min_df", type=float, default=1,
			  help="Minimum frequency for documents")

op.add_option("--max-df", dest="max_df", type=float, default=1.0,
			  help="Maximum frequency for documents")

op.add_option("--n-clusters", dest="n_clusters", type=int, default=8,
			  help="Maximum number of clusters to extract K-Means.")

op.add_option("--n-features", dest="n_features", type=int, default=5000,
			  help="Maximum number of features to extract.")

op.add_option("--n-grams", dest="ngrams", default="(2,3)")

op.add_option("--wordcloud",
			  action="store_true", dest="use_wordcloud", default=False,
			  help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--sentiment",
			  action="store_true", dest="use_sentiment", default=False,
			  help="Evaluate sentiment polarity of cluster words")

op.add_option("--stem",
			  action="store_true", dest="use_stem", default=False,
			  help="Disable Inverse Document Frequency feature weighting.")

op.print_help()

(opts, args) = op.parse_args()
opts.ngrams = eval(opts.ngrams)

if len(args) > 0:
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
	tweets_dict = dict((tweet['id_str'], tweet) for tweet in tweets_list)
	corpus = map(lambda t: t['text'], tweets_list)
	# corpus = list(set(corpus)) # delete retweets ## Should remove retweets from raw_dataset

oldest = time.ctime(min(to_time(t['created_at']) for t in tweets_list))
newest = time.ctime(max(to_time(t['created_at']) for t in tweets_list))

print("done in %fs" % (time.time() - t0))
print("%s tweets loaded: from %s to %s" % (len(tweets_list), oldest, newest))
print

################################################################################
# Vectorization

vectorizer_params = {
	# "ngram_range": opts.ngrams,
	"token_pattern": ur'\b\w+\b',
	"tokenizer": preprocess,
	# "min_df": opts.min_df,
	# "max_df": opts.max_df,
	# "max_features": opts.n_features,
	# "binary": True
}
print vectorizer_params

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time.time()

count_vect = CountVectorizer(**vectorizer_params)
X = count_vect.fit_transform(corpus)

import pdb; pdb.set_trace();

if opts.use_idf:
	tfidf_transformer = TfidfTransformer()
	X = tfidf_transformer.fit_transform(X)

print("done in %fs" % (time.time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print


################################################################################
# Clustering

if opts.run_kmeans:
	n_clusters = opts.n_clusters
	km = KMeans(n_clusters=n_clusters,
				init='k-means++',
				max_iter=300,
				n_init=20, #  @WRITE: Nro de veces que arranca
				n_jobs=-1,
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

	# Cloud of words for a cluster
	try:
		for i in range(n_clusters):
		    print("Cluster %d (%s):" % (i, len([t for t in km.labels_ if t == i])))
		    for ind in order_centroids[i, :10]:
		        print(terms[ind])
		    print
	except:
		print "Unexpected error:", sys.exc_info()[0]

	silhouette_avg = metrics.silhouette_score(X.toarray(), km.labels_, sample_size=1000)
	print("For n_clusters =", n_clusters,
	          "The average silhouette_score is :", silhouette_avg)

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
	sentiment_clf = sentiment.get_clf()
	predicted = sentiment_clf.predict(corpus)
	for i, category in enumerate(predicted):
		tweet_id = tweets_list[i]['id_str']
		tweets_dict[tweet_id]['sentiment'] = sentiment.data.target_names[category]

################################################################################
# Cloud of words

def get_term_polarity(tweets_ids):
	c = 0
	for id in tweets_ids:
		c += (1 if tweets_dict[id]['sentiment'] == 'positive' else
			-1 if tweets_dict[id]['sentiment'] == 'negative' else
			0)
	return 'positive' if c > 0 else 'negative' if c < 0 else 'neutral'

if opts.use_wordcloud or opts.use_sentiment:
	# Counter for each cluster
	counters = [Counter() for _ in range(n_clusters)]
	idcounters = [defaultdict(list) for _ in range(n_clusters)]

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
			if count > 1:
				sentiment_polarity = get_term_polarity(idcounters[i][word])
				'positive' if word in positive_words else 'negative' if word in negative_words else 'neutral'
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
		# clusters.append([(w, c, idcounters[i][w]) for w, c in counter.items() if c > 10])

	if opts.use_wordcloud:
		wc = WordCloud()
		for wcounts in clusters:
			wc.fit_words(wcounts)
			plt.imshow(wc)
			plt.show()

if opts.use_sentiment:
	# Save JSON with Tweets Sentiment
	with open('tweets_sent.json', 'w') as fp:
		json.dump(tweets_dict, fp)
		print "SENTIMENT SAVED ON: %s TWEETS" % len(tweets_dict.keys())

	with open('clusters_wordcloud.json', 'w') as fp:
		json.dump({"data": clusters}, fp)
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
