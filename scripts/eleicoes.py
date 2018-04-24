import nltk
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import os
import matplotlib.pyplot as plt

dataset = pd.read_csv('../bases/tweets_wc_lula_treino.csv')

sentimentos=['A FAVOR','CONTRA']

tweets = dataset['Text']
classes = dataset['Sentimento']

#Gerando o Data Frame
tweets_Dataframe = pd.DataFrame({'Text':tweets,'Classificacao':classes})
#Analisar quantos tweets a base possui
print(len(tweets_Dataframe))

#Analisar quantos dados de cada classificação existem
print(tweets_Dataframe.Classificacao.value_counts())
#Exibindo em grafico de barras
fig = plt.figure(figsize=(8,6))
tweets_Dataframe.groupby('Classificacao').count().plot.bar(ylim=0)
plt.show()
#tecnicas de normalização. Oversampling e undersampling
vectorizer = CountVectorizer(ngram_range=(1,2))
bow = vectorizer.fit_transform(tweets)


#Exibindo o Bag of Words. Lembrar de pegar o vetor com menos dados [:50]
#bow_data_frame = pd.DataFrame(bow.A,columns = vectorizer.get_feature_names())

#normalização de ocorrências. frequência das palavras 
tfidf_transformer = TfidfTransformer()
bow = tfidf_transformer.fit_transform(bow)

#Exibindo o Bag of Words após o TDIDF. Lembrar de pegar o vetor com menos dados [:50]
#bow_data_frame = pd.DataFrame(bow.A,columns = vectorizer.get_feature_names())

#Exibindo apenas valores de uma determinada classificação
#tweets_Dataframe.loc[tweets_Dataframe['Classificacao'] == 'Neutro']
#listClass =(tweets_Dataframe.index[tweets_Dataframe['Classificacao'] == sentimento].tolist())
    


#criar seleção k = quantidade de termos!
# =============================================================================
# for sentimento in sentimentos:    
#     selector = SelectKBest(chi2, k=5).fit(bow, classes == sentimento)
#     #dataTreinoBag = selector.transform(dataTreinoBag)
#     #para exibir os mais cotados. Pega os indexs dos K+ e pega o Bag of Words Gerado inicialmente
#     idxs_selected = selector.get_support(indices=True)
#     features_dataframe = np.asarray(vectorizer.get_feature_names())[idxs_selected]
#     print("Sentimento: ", sentimento + "\n Top Word: ", features_dataframe)
#     nomearquivo = '../bases/' + sentimento + '.csv'
#     tweets_Dataframe.loc[tweets_Dataframe['Classificacao'] == sentimento].to_csv(nomearquivo, encoding='utf-8',index = False)
# =============================================================================
    
modelo = MultinomialNB()
modelo.fit(bow,classes)

resultados = cross_val_predict(modelo, bow, classes, cv=10)

metrics.accuracy_score(classes,resultados)

print (metrics.accuracy_score(classes, resultados))
print (metrics.classification_report(classes,resultados,sentimentos),'')
print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
