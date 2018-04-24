import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from datetime import datetime, date

dataset = pd.read_csv('../bases/tweets_wc_lula_treino.csv')

tweets = dataset['Text']
classes = dataset['Sentimento']

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
bow = vectorizer.fit_transform(tweets)

print(bow.shape)

#normalização de ocorrências. frequência das palavras 
tfidf_transformer = TfidfTransformer()
bow = tfidf_transformer.fit_transform(bow)

##seleção dos K recursos mais significativos
#selector = SelectKBest(chi2, k=20).fit(bow, classes)
#bow = selector.transform(bow)
#
##para exibir os mais cotados. Pega os indexs dos K+ e pega o Bag of Words Gerado inicialmente
#idxs_selected = selector.get_support(indices=True)
#features_dataframe = np.asarray(vectorizer.get_feature_names())[idxs_selected]
#print(features_dataframe)

modelo = MultinomialNB()
modelo.fit(bow,classes)

dataset_test = pd.read_csv('../bases/tweets_wc_lula_teste.csv')
tweets_tests = dataset_test['Text']

bow_tests = vectorizer.transform(tweets_tests)
bow_tests = tfidf_transformer.transform(bow_tests)
#bow_tests = selector.transform(bow_tests)

predicoes = modelo.predict(bow_tests)

resultado_predicao = pd.DataFrame({'Text':tweets_tests,'Classificacao':predicoes})

#Exibindo em grafico de barras
fig = plt.figure(figsize=(8,6))
resultado_predicao.groupby('Classificacao').count().plot.bar(ylim=0)
plt.show()

#name_file = '../predicoes/predicao{}.csv'.format(date.today())
#resultado_predicao.to_csv(name_file, encoding='utf-8', index=False)
