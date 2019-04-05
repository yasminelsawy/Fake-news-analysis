from os.path import join
import pandas as pd

import re
import multiprocessing

import nltk
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import random

import matplotlib.pyplot as plt

# # Functions

# In[ ]:

# ## Language Processing Functions

# In[6]:


def removeLinks(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',
                  text)


def removeLinksDF(tweets):

    # Apply preprocess function to given tweets dataframe
    tweets['text'] = tweets.apply(lambda row: removeLinks(row.text), axis=1)

    # Return tweets dataframe after preprocessing
    return tweets


def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text, stemmer):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(
                token) > 3:
            result.append(lemmatize_stemming(token, stemmer))
    return result


def preprocessDF(tweets):

    # Initialise stemmer
    stemmer = SnowballStemmer('english')

    # Apply preprocess function to given tweets dataframe
    tweets['preprocess'] = tweets.apply(
        lambda row: preprocess(row.text, stemmer), axis=1)

    # Return tweets dataframe after preprocessing
    return tweets


def parallelize_dataframe(df, func):

    print("parallelize_dataframe ??????")

    # Number of cores on your machine
    num_cores = multiprocessing.cpu_count() - 1

    print("num_cores", num_cores)

    # Number of partitions to split dataframe
    num_partitions = num_cores

    # Split dataframe
    df_split = np.array_split(df, num_partitions)

    # Create multiprocesing pool using number of cores
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))

    # Close and join pool
    pool.close()
    pool.join()

    # Return merged dataframe
    return df


def topicChoice(sortedTopics):
    index, score = sortedTopics[0]
    return index


# In[23]:
if __name__ == '__main__':

    # Read fakeNews files
    data_path1 = 'fakenews200000A.csv'
    tweets1DF = pd.read_csv(data_path1, engine='python')

    data_path2 = 'fakenews200000B.csv'
    tweets2DF = pd.read_csv(data_path2, engine='python')

    data_path3 = 'fakenews200000C.csv'
    tweets3DF = pd.read_csv(data_path3, engine='python')

    frames = [tweets1DF, tweets2DF, tweets3DF]

    fakeNewsDataset = pd.concat(frames)
    #fakeNewsDataset = tweets1DF
    print("fakeNewsDataset.shape", fakeNewsDataset.shape)
    print("COncatenation is DONE ??????")

    ######################################################################################################

    # LDA ANALYSIS

    tweets1 = parallelize_dataframe(fakeNewsDataset, removeLinksDF)

    # Get preprocessed tweets
    processed_docs = parallelize_dataframe(tweets1, preprocessDF)['preprocess']

    # Bag of words representation for the dataset
    dictionary = gensim.corpora.Dictionary(processed_docs)
    print("dictionary")
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print("bow_corpus")

    # LDA Analysis
    lda_model = gensim.models.LdaMulticore(
        bow_corpus,
        num_topics=4,
        id2word=dictionary,
        passes=15,
        workers=multiprocessing.cpu_count() - 1)

    print("lda_model DONE !!")

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    print("After for loop")

    unseen_document = 'Muslim report crime'
    stemmer = SnowballStemmer('english')
    bow_vector = dictionary.doc2bow(preprocess(unseen_document, stemmer))

    print("unseen_document", unseen_document)

    print('\nStatement:')

    for index, score in sorted(
            lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
        print('\nTopic', index)
        print("Score: {}\t Topic: {}".format(score,
                                             lda_model.print_topic(index, 5)))
''''
    fakeNewsDataset['Topic'] = fakeNewsDataset.apply(
        lambda row: (topicChoice(sorted(lda_model[dictionary.doc2bow(preprocess(row.text, stemmer))], key=lambda tup: -1 * tup[1]))),axis=1
    )

    fakeNewsDataset.to_csv('fakeNewsDatasetTopic.csv')
'''
