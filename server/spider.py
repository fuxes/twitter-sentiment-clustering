import time
import json
import datetime
import optparse

import constants
import tweepy

"""
This module is used to crawl twitter hasthtags

Notes:
 - Remove tokens to a different module
 - Queries should be setted by console params
 - Now everything is hardcoded
"""


class TwitterSerializer:

    def __init__(self, filename, new=True):
        self.tweet_count = 0
        self.filename = filename
        self.__dirty = False
        self.__open_method = 'w'

    def write(self, tweet):
        fp = open(self.filename, self.__open_method)

        if self.__dirty:
            fp.write(', ')
        else:
            fp.write('{"data":[')
            self.__dirty = True
            self.__open_method = 'a'

        fp.write('\n' + json.dumps(self.min_tweet(tweet)))
        self.tweet_count += 1

        if (self.tweet_count % 300 == 0):
            print(self.tweet_count)

        fp.close()

    def close(self):
        fp=open(self.filename, 'a')
        fp.write('\n]}')
        fp.close()

    def get_count(self):
        return self.tweet_count

    def min_tweet(self, tweet):
        return {
        'id_str': tweet['id_str'],
        'created_at': tweet['created_at'],
        'text': tweet['text'],
        'user': {
            'profile_image_url': tweet['user']['profile_image_url'],
            'name': tweet['user']['name'],
            'screen_name': tweet['user']['screen_name']
            },
        }

def get_tweeter_api():
    auth=tweepy.OAuthHandler(constants.CONSUMER_KEY, constants.CONSUMER_SECRET)
    auth.set_access_token(constants.ACCESS_TOKEN, constants.ACCESS_SECRET)
    api=tweepy.API(auth)

    return api

# Workaround for twitter limitations
def limited_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.TweepError:
            print('Error! Waiting 15 minutes! Pause started at: %s' %
                  str(datetime.datetime.now()))
            time.sleep(15*60)  # sleep 15 mins
        except (StopIteration, KeyboardInterrupt):
            break

def main():
    TwitterApi=get_tweeter_api()
    query='#oscars' #HERE GOES THE SEARCH QUERY
    BASE='oscars/'
    # date = '2016-05-20'
    print('Downloading %s' % query)

    total_tweets=0
    partial_tweet=0

    serializer=TwitterSerializer('oscars_crawled.json')
    for tweet in limited_handled(tweepy.Cursor(
        TwitterApi.search, q=query, lang="en").items()):

        tweet_json = tweet._json
        if not tweet_json['text'].startswith('RT'):
            serializer.write(tweet_json)

    serializer.close()
    print('Total of %s tweets!' % serializer.get_count())

if __name__ == '__main__':
    main()
