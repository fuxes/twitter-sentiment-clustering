import tweepy
import time
import json
import datetime
from tweepy import OAuthHandler

"""
This module is used to crawl twitter hasthtags

Notes:
 - Remove tokens to a different module
 - Queries should be setted by console params
 - Now everything is hardcoded
"""

#config
consumer_key = "zq7rlKRXw09F12wNkOGOvc09Y"
consumer_secret = "VdG7CBvgrJ9EaYqiBtsYcuf3x4grIFPu7eEF4P7btDWIYjrS13"
access_token = "590204184-lNiLkqD8yKOmbrFjHTUhdEOl3N3Tl3tNa8rJzAwK"
access_secret = "07hEAULtXYF5lUQWo6SRQgzAGEUrb5kjVrUUXO1wJ0gMV"

#api setup
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# Workaround for twitter limitations
def limited_handled(cursor):
	while True:
		try:
			yield cursor.next()
		except tweepy.TweepError:
			print 'Error! Waiting 15 minutes! Pause started at: %s' % str(datetime.datetime.now())
			time.sleep(15*60) # sleep 15 mins
		except StopIteration:
			print '\nYUP! FINISH!'
			break
		except KeyboardInterrupt:
			print '\nAlright...'
			break


# Twitter crawler
def get_tweets():
	query = '#oscars'

	print ('Downloading %s' % query)
	dates = [
		'2016-03-07',
		'2016-03-06',
		'2016-03-05',
		'2016-03-04',
		'2016-03-03',
		'2016-03-02',
		'2016-03-01',
		'2016-02-29'
	]

	total_tweets = 0
	partial_tweet = 0

	with open('oscars_inf.json', 'a') as file:
		for date in dates:
			partial_tweet = 0

			for tweet in limited_handled(tweepy.Cursor(api.search, q=query, lang="en", until=date).items()):
				file.write(json.dumps(tweet._json))
				partial_tweet += 1
				total_tweets += 1
				if (partial_tweet % 150 == 0):
					print total_tweets, str(tweet.created_at), tweet.user.name, tweet.text

			print '%s tweets from %r!' % (partial_tweet, date)
			print 'For a total of %s mf\'ing tweets!' % total_tweets

if __name__ == '__main__':
	get_tweets()
