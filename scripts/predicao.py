import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from datetime import datetime, date

def criar_arquivo(dataFrame):
    name_file = '../predicoes/predicao{}.csv'.format(date.today())
    dataFrame.to_csv(name_file, encoding='utf-8', index=False)

dataset = pd.read_csv('../bases/tweets_lula_treino.csv')

tweets = dataset['Text']
classes = dataset['Sentimento']

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
bow = vectorizer.fit_transform(tweets)

print(bow.shape)

#normalização de ocorrências. frequência das palavras 
tfidf_transformer = TfidfTransformer()
bow = tfidf_transformer.fit_transform(bow)

modelo = MultinomialNB()
modelo.fit(bow,classes)

dataset_test = pd.read_csv('../bases/tweets_lula_teste.csv')
tweets_tests = dataset_test['Text']

bow_tests = vectorizer.transform(tweets_tests)
bow_tests = tfidf_transformer.transform(bow_tests)

predicoes = modelo.predict(bow_tests)

resultado_predicao = pd.DataFrame({'Text':tweets_tests,'Classificacao':predicoes})

#Exibindo em grafico de barras
fig = plt.figure(figsize=(8,6))
resultado_predicao.groupby('Classificacao').count().plot.bar(ylim=0)
plt.show()

criar_arquivo(resultado_predicao)

