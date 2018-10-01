#!/usr/bin/env python
# coding: utf-8

# ### Natural Language Processing ###
#      A Naive Bayes Classifier 
#      Name: Nana Ama Atombo-Sackey
#    

# In[1]:


import re
from math import *
#reading the file.
def readFile(files):
    MainClassification=[]
    data = {
        0:[],
        1:[]}
    for file in files:
        for line in open(file):
            # getting rid of all regular expressions not needed
            lineCleaner = re.sub(r"[,/?!-()*&^}:;{=$%]","",line)
            lineCleaner2 = re.sub(r"[.']"," ",lineCleaner.lower())
            final = MainClassification.append(lineCleaner2)
            review = line.split('\t')
            features= review[0].split()
            label = int(review[1])
            if label == 0:
                data[0].append(features)
            else:
                data[1].append(features)
            
    print("The total words in the negative class is: " , len(data[0])) 
    print("The total words in the positive class is: " , len(data[1])) 
    return data
corpus= readFile(['amazon_cells_labelled.txt',"imdb_labelled.txt","yelp_labelled.txt"])
    


# #### Calculating log prior and loglikelihood ###
# 

# In[3]:


def train(doc):
    #Initializng the logprior and Loglikehood
    classes= [0,1]
    prior = dict()
    likelihood= {
        0:{},
        1:{}
    }
    numOfDocD = len(doc[0])+len(doc[1])
    wordCount = {
        0:{},
        1:{}
    }
    print("The number of documents in D(corpus) is :", numOfDocD)
    #calculating log prior
    for c in classes:
        numOfDocClass = len(doc[c])
        prior[c] = log((numOfDocClass/numOfDocD))
        #print(prior[c])
        
    print("The number of documents from Corpus in the class is: ",numOfDocClass)
    #print("The logpriority is: ", logprior)
   
    for c in doc:
        for reviews in doc[c]:

            for words in reviews:
                if words in wordCount[c]:
                    wordCount[c][words]+=1
                else:
                    wordCount[c][words]=1
    #print(wordCount)
    vocabulary = []
    for c in classes:
        vocabulary += list(wordCount[c].keys()) 
    vocabulary = set(vocabulary)
    for words in vocabulary:
        for c in classes:
            if words in wordCount[c]:
                likelihood[c][words] = log( ((wordCount[c][words] + 1)/(sum(wordCount[c].values())+len(vocabulary))) )
            else:
                 likelihood[c][words] = log(((1)/(sum(wordCount[c].values())+len(vocabulary))) )
   # print (likelihood[0]['good'], likelihood[1]['good'])
   # print (wordCount[0]['good'], wordCount[1]['good'])
    return prior,likelihood,vocabulary
prior,likelihood,vocabulary=train(corpus)


# #### Implementing Test Function ####

# In[4]:


def test(doc, prior, likelihood,vocabulary):
    summ = dict()
    for c in [0,1]:
        summ[c]= prior[c]
        for word in doc:
            if word in vocabulary:
                summ[c] = summ[c] + likelihood[c][word]
    print(summ)
    if summ[0]> summ[1]:
        return 0 
    else:
        return 1
    
print(test("awesome", prior,likelihood,vocabulary))







