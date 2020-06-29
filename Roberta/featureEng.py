import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import itertools
import xgboost as xgb
import re
import operator
from collections import Counter
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.metrics import confusion_matrix

from featureExt import *
sentiment_mapping = {'positive':0,'negative':1,'neutral':2}
#nltk.download('stopwords')
def featureEng(train, name = 'NB1', n_comp = 50, ngram_cv = 3, ngram_tfidf= 3, training=True):
    eng_stopwords = set(stopwords.words('english'))

    cls = [(RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=10), "Random forest"),
        (LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3),"SVC-L1"),
        (LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3),"SVC-L2"),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='l1'),'SGD-L1'),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='l2'),'SGD-L2'),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='elasticnet'),'SGD-ElasticNet'),
        (NearestCentroid(),'Nearest neighbor'),
        (MultinomialNB(alpha=.1),'NB1'),
        (BernoulliNB(alpha=.1),'NB2')]

    train['num_words'] = train['text'].apply(lambda x:len(str(x).split()))
    train['num_unique_words'] = train['text'].apply(lambda x:len(set(str(x).split())))    
    train['num_chars'] = train['text'].apply(lambda x:len(str(x)))
    train['num_stopwords'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))    
    train['num_punctions'] = train['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))
    train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    ## Number of title case words in the text ##
    train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    ## Average length of the words in the text ##
    train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    all_text_without_sw = ''
    for i in train.itertuples():
        all_text_without_sw = all_text_without_sw +  str(i.text)
    #getting counts of each words:
    counts = Counter(re.findall(r"[\w']+", all_text_without_sw))
    #deleting ' from counts
    del counts["'"]
    #getting top 50 used words:
    sorted_x = dict(sorted(counts.items(), key=operator.itemgetter(1),reverse=True)[:300])

    #Feature-5: The count of top used words.
    train['num_top'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]) )
    
    #Similarly lets identify the least used words:
    reverted_x = dict(sorted(counts.items(), key=operator.itemgetter(1))[:10000])
    #Feature-6: The count of least used words.
    train['num_least'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]) )
    train['unique_word_fraction'] = train['text'].apply(lambda row: unique_word_fraction(row))
    train['stopwords_count'] = train['text'].apply(lambda row: stopwords_count(row))
    train['punctuations_fraction'] = train['text'].apply(lambda row: punctuations_fraction(row))
    train['char_count'] = train['text'].apply(lambda row: char_count(row))
    train['fraction_noun'] = train['text'].apply(lambda row: fraction_noun(row))
    train['fraction_adj'] = train['text'].apply(lambda row: fraction_adj(row))
    train['fraction_verbs'] = train['text'].apply(lambda row: fraction_verbs(row))
    train['sentiment_id'] = train['sentiment'].apply(lambda row: sentiment_mapping[row])    
    if training:
        train['y1'] = train.apply(lambda row: process_data(row.text,row.selected_text)[0],axis=1) 
        train['y2'] = train.apply(lambda row: process_data(row.text,row.selected_text)[1],axis=1) 
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,ngram_tfidf))
    train_tfidf = tfidf_vec.fit_transform(train['text'].values.tolist())
    
    ###SVD on word TFIDF
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='randomized')
    svd_obj.fit(train_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

    train_svd.columns = ['svd_wordtfidf_'+str(i) for i in range(n_comp)]
    train = pd.concat([train, train_svd], axis=1)
    del train_tfidf, train_svd

    ### Fit transform the count vectorizer ###
    wordcv_vec = CountVectorizer(stop_words='english', ngram_range=(1,ngram_cv))
    train_vec = wordcv_vec.fit_transform(train['text'].values.tolist())

    ###SVD on Character TFIDF
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='randomized')
    svd_obj.fit(train_vec)
    train_svd = pd.DataFrame(svd_obj.transform(train_vec))
        
    train_svd.columns = ['svd_wordcv_'+str(i) for i in range(n_comp)]
    train = pd.concat([train, train_svd], axis=1)
    del train_vec, train_svd

    charcv_vec = CountVectorizer(ngram_range=(1,ngram_cv), analyzer='char')
    train_vec = charcv_vec.fit_transform(train['text'].values.tolist())

    ###SVD on Character TFIDF
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='randomized')
    svd_obj.fit(train_vec)
    train_svd = pd.DataFrame(svd_obj.transform(train_vec))
        
    train_svd.columns = ['svd_charcv_'+str(i) for i in range(n_comp)]
    train = pd.concat([train, train_svd], axis=1)
    del train_vec, train_svd
    
    return train