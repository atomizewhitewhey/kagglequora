#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:07:26 2017

@author: matthewyeozhiwei
"""

import pandas as pd
train = pd.read_csv('/Users/matthewyeozhiwei/repos/KaggleQuora/train.csv')
questions1 = train['question1'].tolist()
questions2 = train['question2'].tolist()


import string
sp = string.punctuation

def normalize(questions):
    questions = list(map(lambda t: ''.join(["" if c.isdigit() else c for c in str(t)]), questions))
    questions = list(map(lambda t: ''.join(["" if c in sp else c for c in str(t)]), questions))
    questions = list(map(str.lower, questions))
    return questions

questions1 = normalize(questions1)
questions2 = normalize(questions2)



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
    
## wf = to_TF(questions1)

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



stop_words = pd.read_csv('/Users/matthewyeozhiwei/repos/KaggleQuora/stopwords.csv')
stop_words = [w for w in stop_words.words if w in stop_words.words.unique() ]

def rmstop(questions):
    temp = [question.split() for question in questions]
    questions = [' '.join([word for word in question if word not in set(stop_words)]) for question in temp]
    return questions

questions1 = rmstop(questions1)
questions2 = rmstop(questions2)
           
## wf = to_TF(questions1)
## wf_bar(wf)
## plot_cfd(wf)

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

def stem(questions):
    temp = [question.split() for question in questions] 
    temp = map(lambda t: [porter_stemmer.stem(w) for w in t], temp)
    questions = [' '.join(question) for question in temp] 
    return questions

questions1 = stem(questions1)
questions2 = stem(questions2)

##wf = to_TF(questions1)
##wf_bar(wf)
##plot_cfd(wf)
questions1n = normalize(questions1)
questions2n = normalize(questions2)
questions1n = rmstop(questions1n)
questions2n = rmstop(questions2n)
questions1n = stem(questions1n)
questions2n = stem(questions2n)

def numwordf(questions1n, questions2n):
    temp1 = [question.split() for question in questions1n] 
    temp2 = [question.split() for question in questions2n]
    numwords = []
    for i in range(len(temp1)):
        temp3 = 0
        for z in temp1[i]:
            if z in temp2[i]:
                temp3 = temp3 + 1
        numwords.append(temp3)
    return numwords

numwords = numwordf(questions1n, questions2n)

def stopwords(questions):
    temp = [question.split() for question in questions]
    questions = [' '.join([word for word in question if word in set(stop_words)]) for question in temp]
    return questions

questions1stop = normalize(questions1)
questions2stop = normalize(questions2)
questions1stop = stopwords(questions1stop)
questions2stop = stopwords(questions2stop)

def numstopf(questions1stop, questions2stop):  
    temp1 = [question.split() for question in questions1stop] 
    temp2 = [question.split() for question in questions2stop]
    numstop = []
    for i in range(len(temp1)):
        temp3 = 0
        for z in temp1[i]:
            if z in temp2[i]:
                temp3 = temp3 + 1
        numstop.append(temp3)
    return numstop

numstop = numstopf(questions1stop, questions2stop)
print(numstop[:5])

def firstword(questions):
    questions = [question.split() for question in questions]
    interog = []
    for question in questions:
        if len(question) > 0:
            interog.append(question[0])
        else:
            interog.append('0')
    return interog

def sameqf(questions1, questions2):
    interog1 = normalize(questions1)
    interog2 = normalize(questions2)
    interog1 = firstword(interog1)
    interog2 = firstword(interog2)
    sameq = []
    for i in range(len(interog1)):
        if interog1[i] == interog2[i]:
            sameq.append(1)
        else:
            sameq.append(0)
    return sameq

sameq = sameqf(questions1, questions2)
print(sameq[:5])