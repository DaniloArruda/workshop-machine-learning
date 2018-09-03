# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:42:17 2018

@author: mathe
"""

import pandas as pd
import tweepy

consumer_key = 'eICGRZ3xMVGAS2LZtW2HZ8ESP'
consumer_secret = 'uaaibUdbfNujAJbaXJ6fZCDoqTPZVapf6iLMSuOl7zvEATfRky'
access_token = '988578283922100225-kGZrxiEFmMNPVlIupkBz45BzCf5n4Dk'
access_token_secret = 'toqhW7r8FfmWQwyf0ebUsQY9tfa2QfUiqHAJYuarET3Br'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
tweets = list() 
sentimentos = list()

for tweet in tweepy.Cursor(api.search,q='#Lula  -filter:retweets',tweet_mode='extended',count=100,lang="pt").items(100):
    if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
        tweets.append(tweet.full_text)
        
print('Coletados tweets sobre Lula')
tweets_Dataframe = pd.DataFrame({'Text':tweets})
tweets_Dataframe.to_csv('../bases/tweets_lula_teste.csv', encoding='utf-8',index = False)

print(api.rate_limit_status()['resources']['search'])