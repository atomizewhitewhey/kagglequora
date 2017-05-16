#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:07:26 2017

@author: matthewyeozhiwei
"""

import pandas as pd
train = pd.read_csv('/Users/matthewyeozhiwei/Downloads/train2.csv')
questions1 = train['question1'].tolist()
questions2 = train['question2'].tolist()

import string
sp = string.punctuation
questions1 = list(map(lambda t: ''.join(["" if c.isdigit() else c for c in t]), questions1))
questions1 = list(map(lambda t: ''.join(["" if c in sp else c for c in t]), questions1))
questions1 = list(map(str.lower, questions1))

'''
questions2 = list(map(lambda t: ''.join(["" if c.isdigit() else c for c in t]), questions2))
questions2 = list(map(lambda t: ''.join(["" if c in sp else c for c in t]), questions2))
questions2 = list(map(str.lower, questions2)) 
'''

def to_TF(questions):
    import pandas as pd
    import nltk
    questions = list(map(lambda t: nltk.regexp_tokenize(t, r'\S+'), questions))
    questions = [w for l in questions for w in l]

    ## Compute the frequency distribution of the words as a dictionary
    fdist = nltk.FreqDist(questions) 
    ## Convert the dictionary to a dataframe contaning the words and
    ## counts indexed by the words, and then take the transpose.
    count_frame = pd.DataFrame(fdist, index =[0]).T
    count_frame.columns = ['Count']
    return(count_frame.sort_values('Count', ascending = False))
    
wf = to_TF(questions1)

def wf_bar(wf):
    import matplotlib.pyplot as plt
    ## Barplot of the most fequent words.   
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca()    
    wf['Count'][:60].plot(kind = 'bar', ax = ax)
    ax.set_title('Frequency of the most common words')
    ax.set_ylabel('Frequency of word')
    ax.set_xlabel('Word')
    plt.show()
    return 'Done'
##wf_bar(wf)

def plot_cfd(wf):
    import matplotlib.pyplot as plt
    ## Compute the relative cumulative frequency of the words in 
    ## descending order of frequency and add the dataframe.   
    word_count = float(wf['Count'].sum(axis = 0))   
    wf['Cum'] = wf['Count'].cumsum(axis = 0)
    wf['Cum'] = wf['Cum'].divide(word_count)
    
    ## Barplot the cumulative frequency for the most frequent words.   
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca()    
    wf['Cum'][:60].plot(kind = 'bar', ax = ax)
    ax.set_title('Cumulative fraction of total words vs. words')
    ax.set_ylabel('Cumulative fraction')
    ax.set_xlabel('Word')
    plt.show()
    return 'Done'
##plot_cfd(wf)


stop_words = pd.read_csv('/Users/matthewyeozhiwei/Downloads/stopwords2.csv')
stop_words = [w for w in stop_words.words if w in stop_words.words.unique() ]
print(stop_words[:20])

temp = [question.split() for question in questions1] ## Split tweets into tokens
questions1 = [' '.join([word for word in question if word not in set(stop_words)]) for question in temp]
'''            
wf = to_TF(questions1)
wf_bar(wf)
plot_cfd(wf)
'''
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
temp = [question.split() for question in questions1] ## Split questions into tokens
temp = map(lambda t: [porter_stemmer.stem(w) for w in t], temp)
questions1 = [' '.join(question) for question in temp] ## Join the words of the question string
'''
wf = to_TF(questions1)
wf_bar(wf)
plot_cfd(wf)
'''

print(questions1[:10])