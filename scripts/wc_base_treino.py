# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:52:48 2018

@author: Danilo
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
count = 0           
tweets = list()
sentimentos  = list() 

tweets_retorno = tweepy.Cursor(api.search,q='#LulaLivre  -filter:retweets',tweet_mode='extended',count=200,lang="pt").items(200)

for tweet in tweets_retorno:
    count+=1
    if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
        tweets.append(tweet.full_text)
        sentimentos.append('A FAVOR')
    
print('Coletados tweets a favor')    
        
for tweet in tweepy.Cursor(api.search,q='#LulaPreso  -filter:retweets',tweet_mode='extended',count=200,lang="pt").items(200):
    count+=1
    if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
        tweets.append(tweet.full_text)
        sentimentos.append('CONTRA')       
                     
print('Coletados tweets contra')

tweets_Dataframe = pd.DataFrame({'Text':tweets, 'Sentimento':sentimentos})
tweets_Dataframe.to_csv('../bases/tweets_lula_treino.csv', encoding='utf-8', index = False)